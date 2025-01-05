# Developing a method for generating a training image


# %%
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapely as shp
import rasterio as rio
from rasterio import features as rio_features

import training_dataset as td

# from importlib import reload

################################################################################
# %%
# Config

# Note: All files exported from earth engine seem to be in EPSG:4326?

# Data directories
root_data_dir = Path("data")
data_dir = root_data_dir / "greece"

# Area of interest in EPSG:4326
aoi_filepath = data_dir / "aoi.geojson"

# Projected CRS appropriate for the area of interest
projected_crs = "EPSG:2100"

# Layers in EPSG:4326
landcover_polygons_filepath = data_dir / "landcover_100m_binary_vector.geojson"

# # Landcover lookup table
# # Contains information on how to interpret the landcover data
# # landcover_lookup_filepath = data_dir / "landcover_lookup_CGLS-LC100_V3.csv"
# landcover_lookup_filepath = root_data_dir / "landcover_lookup_binary.csv"

# Tile width range in meters
# TODO: Approximate tile width range from the aoi dimensions?
tile_width_range = (10**2, 10**5) # meters

# Final raster size in pixels
tile_raster_size = 64

# Pick a random seed (would be idx in __getitem__)
rand_seed = 0

########################################

# Check input filepaths
to_check = [
    aoi_filepath,
    # landcover_lookup_filepath,
    landcover_polygons_filepath,
]
for path in to_check:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

################################################################################
# %%
# Initialization (preprocessing)

aoi_raw_gdf = gpd.read_file(aoi_filepath)
aoi_shp = aoi_raw_gdf.to_crs(projected_crs).union_all()

# # Load landcover lookup table
# landcover_lookup_df = pd.read_csv(landcover_lookup_filepath)
# name_lookup_dict = (
#     landcover_lookup_df
#     .set_index("value")
#     .drop(columns=["description"])
# ).to_dict()["name"]


# Load landcover data and do some basic cleaning
landcover_data_gdf = (
    gpd.read_file(landcover_polygons_filepath)
    # .replace({"label": name_lookup_dict})
    .drop(columns=["id", "count"])
    .dissolve(by="label")
    .to_crs(projected_crs)
)

# %%
# Plot the data
def plot_basic_map(ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    landcover_data_gdf.plot(
        ax=ax,
        legend=True,
        figsize=(10, 10),
        legend_kwds={"loc": "upper left"},
        color=["lightblue", "lightgreen"],
        edgecolor="black",
        linewidth=0.1,
    )
    ax.plot(*aoi_shp.exterior.xy, "r", linewidth=5)

    return ax

def plot_polygons(
        geometries: shp.geometry.base.BaseGeometry|None,
        ax: mpl.axes.Axes,
):
    """ Plot a shapely MultiPolygon or Polygon """
    if geometries is None:
        return

    if isinstance(geometries, shp.geometry.MultiPolygon):
        geoms_iter = geometries.geoms
    elif isinstance(geometries, shp.geometry.Polygon):
        geoms_iter = [geometries]
    else:
        raise ValueError(f"Unknown geometry type {type(geometries)}")

    for geom in geoms_iter:
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys)

plot_basic_map()
plt.show()
# %%
################################################################################
# Generate a random tile

# Set up a random number generator using the index as a seed
idx = rand_seed
rng = np.random.default_rng(idx)

# Pick a random tile width
tile_width = rng.integers(*tile_width_range)

# %%
# Figure out a target zone that will fit a tile with any rotation
# This is a polygon inside the aoi that is at least max_tile_distance
# away from the edges of the aoi
max_tile_distance = np.sqrt(2) * tile_width / 2
target_zone_shp = aoi_shp.buffer(-max_tile_distance)

# %%
# Pick a random point in the target zone
# Pick a random point inside the target zone
# This will be the center of the tile
# Surprisingly, guess-and-check appears to be the best way to do it
x_range = target_zone_shp.bounds[::2]
y_range = target_zone_shp.bounds[1::2]
target_point = None
max_attempts = 100
n_attempts = 0
while target_point is None:
    candidate_point = shp.geometry.Point(
        rng.uniform(*x_range),
        rng.uniform(*y_range),
    )
    if target_zone_shp.contains(candidate_point):
        target_point = candidate_point

    # Track number of attempts as a failsafe
    n_attempts += 1
    if n_attempts == max_attempts:
        raise RuntimeError(
            "Failed to find a random point in the target zone after "
            f"{max_attempts} attempts. It is highly probable something "
            "is wrong with the aoi/target zone."
        )

# %%
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

# %%
# Clip the landcover polygons to the rotated square
landcover_gdf = landcover_data_gdf
rotated_square_landcover = landcover_gdf.intersection(rotated_square)

# %%
# Unrotate the square data to get the tile data
tile_polygons = rotated_square_landcover.rotate(
    -rotation_angle,
    origin=target_point,
)

# %%
# Rasterize the tile polygons and transform to the raster size
tile_size = tile_raster_size
rio_transform = rio.transform.from_bounds(
    *square_bounds,
    tile_size,
    tile_size,
)
tile_raster = rio_features.rasterize(
    [(geom, value) for value, geom in tile_polygons.items()],
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

# %%
################################################################################
# Using pytorch dataset

# Initialize the dataset object
# reload(td)
dataset = td.BinaryLandcoverDataset(
    aoi_filepath=aoi_filepath,
    projected_crs=projected_crs,
    landcover_polygons_filepath=landcover_polygons_filepath,
    # landcover_lookup_filepath=landcover_lookup_filepath,
    tile_width_range=tile_width_range,
    tile_raster_size=tile_raster_size,
)

# Get the random tile
pytorch_tile, pytorch_land_percent = dataset[rand_seed]


# %%
################################################################################
# Plot stuff

# %%
# Plot entire map area
ax = plot_basic_map()
# fig, ax = plt.subplots(figsize=(10, 10))
plot_polygons(target_zone_shp, ax)
plot_polygons(rotated_square, ax)
landcover_data_gdf.plot(ax=ax, facecolor="none", linewidth=0.5)
rotated_square_landcover.plot(ax=ax, facecolor="none", linewidth=0.5)

ax.plot(*target_point.xy, "ro", markersize=10)

ax.set_title(
    f"rng seed: {rand_seed}, tile width: {tile_width} m, "
    f"rotation angle: {rotation_angle}"
)

plt.show()

# %%
# Plot only the final tile and associated polygons

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.flipud(tile_raster), cmap="gray", origin="lower")

# Scale and translate the polygons to line up with raster
tile_polygons_normalized = (
    tile_polygons
    .translate(-square_bounds[0], -square_bounds[1])
    .scale(tile_size / tile_width, tile_size / tile_width, origin=(0, 0))
    .translate(-0.5, -0.5) # so polys line up with imshow pixels
)
tile_polygons_normalized.plot(
    ax=ax,
    facecolor="none",
    edgecolor="red",
    linewidth=0.5,
)

ax.set_title(
    "Script tile\n"
    f"rng seed: {rand_seed}, tile width: {tile_width} m,\n"
    f"final resolution: {tile_raster_size} px, "
    f"land percent: {land_percent:.2f}"
)

plt.show()

# %%
# Plot the pytorch tile for comparison
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.flipud(pytorch_tile), cmap="gray", origin="lower")

ax.set_title(
    "Pytorch tile\n"
    f"rng seed: {rand_seed}, tile width: {tile_width} m,\n"
    f"final resolution: {tile_raster_size} px, "
    f"land percent: {land_percent:.2f}"
)

plt.show()

# %%