from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import shapely as shp
import torch
from rasterio import features as rio_features
from torch.utils.data import Dataset


class BinaryLandcoverDataset(Dataset):
    def __init__(
        self,
        aoi_filepath: Path,
        projected_crs: str,
        landcover_polygons_filepath: Path,
        # landcover_lookup_filepath: Path,
        tile_width_range: tuple[int | float, int | float],
        tile_raster_size: int = 32,
    ):
        """Set up a dataset for binary landcover classification

        Args:
            aoi_filepath:
                Path to the area of interest GeoJSON file. Must have the crs
                EPSG:4326, which is what Earth Engine exports, and will be
                converted to projected_crs for calculations.
            projected_crs:
                A projected CRS that is appropriate for the area of interest.
                This is needed to convert the horizontal units to meters
                (instead of degrees) for spatial calculations.
            landcover_polygons_filepath:
                Path to the GeoJSON file containing polygons representing the
                landcover classes. The file must have a crs of EPSG:4326 and
                will be converted to the projected_crs for calculations.
            tile_width_range:
                A tuple containing the minimum and maximum width of the tiles
                to extract. The width is in meters.
            tile_raster_size:
                The size (in pixels) of the final square raster images. Once a
                tile area has been selected, the tile is resampled to this size.

        """

        # NOTE: landcover lookup doesn't seem necessary. Leaving out for now
        # landcover_lookup_filepath:
        #     Path to a CSV file that contains information on how to interpret
        #     the landcover data. This files should contain the columns:
        #         - value: The landcover class id found in the data
        #         - name: The landcover name to use for labels
        #         - description: An optional description of the landcover
        #         class.

        self.projected_crs = projected_crs
        self.tile_width_range = tile_width_range
        self.tile_raster_size = tile_raster_size

        # TODO: Approximate tile width range from the aoi dimensions?

        self.aoi_raw_gdf = gpd.read_file(aoi_filepath)
        self.aoi_shp = self.aoi_raw_gdf.to_crs(projected_crs).union_all()

        # # Load landcover lookup table
        # self.landcover_lookup_df = pd.read_csv(landcover_lookup_filepath)
        # name_lookup_dict = (
        #     self.landcover_lookup_df
        #     .set_index("value")
        #     .drop(columns=["description"])
        # ).to_dict()["name"]

        # Load landcover data and do some basic cleaning
        self.landcover_data_gdf = (
            gpd.read_file(landcover_polygons_filepath)
            # .replace({"label": name_lookup_dict})
            .drop(columns=["id", "count"])
            .dissolve(by="label")
            .to_crs(projected_crs)
        )

    def __len__(self):
        return 10**6

    def get_random_point_in_polygon(
        self,
        target_zone_shp: shp.Polygon,
        rng: np.random.Generator,
    ) -> shp.Point:
        """Get a random point inside a polygon

        Args:
            target_zone_shp:
                The shapely polygon to sample from.
            rng:
                A numpy random number generator object

        Returns:
            A shapely Point object that is inside the target zone
        """

        # Pick a random point inside the target zone
        # This will be the center of the tile
        # Surprisingly, guess-and-check appears to be the best way to do it
        x_range = target_zone_shp.bounds[::2]
        y_range = target_zone_shp.bounds[1::2]
        rand_point = None
        max_attempts = 100
        n_attempts = 0
        while rand_point is None:
            candidate_point = shp.geometry.Point(
                rng.uniform(*x_range),
                rng.uniform(*y_range),
            )
            if target_zone_shp.contains(candidate_point):
                rand_point = candidate_point

            # Track number of attempts as a failsafe
            n_attempts += 1
            if n_attempts == max_attempts:
                raise RuntimeError(
                    "Failed to find a random point in the target zone after "
                    f"{max_attempts} attempts. It is highly probable something "
                    "is wrong with the aoi/target zone."
                )

        return rand_point

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        # Set up a random number generator using the index as a seed
        rng = np.random.default_rng(idx)

        # Pick a random tile width
        tile_width = rng.integers(*self.tile_width_range)

        # Figure out a target zone that will fit a tile with any rotation
        # This is a polygon inside the aoi that is at least max_tile_distance
        # away from the edges of the aoi
        max_tile_distance = np.sqrt(2) * tile_width / 2
        target_zone_shp = self.aoi_shp.buffer(-max_tile_distance)

        # Check that there is a target zone to sample from
        if target_zone_shp.is_empty:
            raise ValueError("Target zone is empty. Tile width too large.")

        # Pick a random point in the target zone
        target_point = self.get_random_point_in_polygon(target_zone_shp, rng)

        # Pick a random rotation angle
        rotation_angle = rng.uniform(0, 360)

        # Make a square around the target point and rotate it
        square = shp.geometry.box(*target_point.buffer(tile_width / 2).bounds)
        square_bounds = square.bounds
        rotated_square = shp.affinity.rotate(
            square,
            rotation_angle,
            origin=target_point,
        )

        # Clip the landcover polygons to the rotated square
        landcover_gdf = self.landcover_data_gdf
        rotated_square_landcover = landcover_gdf.intersection(rotated_square)

        # Unrotate the square data to get the tile data
        tile_polygons = rotated_square_landcover.rotate(
            -rotation_angle,
            origin=target_point,
        )

        # Rasterize the tile polygons and transform to the raster size
        tile_size = self.tile_raster_size
        rio_transform = rio.transform.from_bounds(
            *square_bounds,
            tile_size,
            tile_size,
        )
        geom_mapping = [
            (geom, value)
            for value, geom in tile_polygons.items()
            if not geom.is_empty
        ]
        if not geom_mapping:
            raise ValueError("No landcover data in tile")
        tile_raster = rio_features.rasterize(
            geom_mapping,
            out_shape=(tile_size, tile_size),
            transform=rio_transform,
            all_touched=False,
            fill=0,
            default_value=1,
            dtype="uint8",
        )

        # Calculate a metric for the tile
        # In this case the percent of the tile that is land
        land_percent = tile_raster.sum() / tile_raster.size * 100

        # Convert the tile raster to a tensor
        tile_tensor = torch.tensor(tile_raster, dtype=torch.uint8)

        return tile_tensor, land_percent
