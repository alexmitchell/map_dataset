# Developing a method for generating a training image


# %%
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapely as shp
import rasterio as rio
from rasterio import features

################################################################################
# %%
# Config

# Note: All files exported from earth engine seem to be in EPSG:4326?

data_dir = Path("data")
aoi_filepath = Path("aoi/greece_aoi.geojson")

# define projected CRS for greece
proj_crs = "EPSG:2100"

# Layers
landcover_100m_binary_filepath = data_dir / "landcover_100m_binary_vector.geojson"

# Output
target_zone_filepath = data_dir / "greece_aoi_target_zone.geojson"

# Set up a random number generator
rand_seed = 0
rng = np.random.default_rng(rand_seed)

# Tile width range
tile_width_range = (10**2, 10**5) # meters
tile_width = rng.integers(*tile_width_range)

tile_raster_size = 64

################################################################################
# %%
# Load and clean data

# Load the aoi
aoi_raw = gpd.read_file(aoi_filepath)
aoi = aoi_raw.to_crs(proj_crs)

# Load landcover data
landcover_100m_binary = (
    gpd.read_file(landcover_100m_binary_filepath)
    .replace({"label": {1: "land", 0: "water"}})
    .drop(columns=["id", "count"])
    .dissolve(by="label")
    .to_crs(proj_crs)
)

# %%
# Plot the data
def plot_basic_map(ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    landcover_100m_binary.plot(
        ax=ax,
        legend=True,
        figsize=(10, 10),
        legend_kwds={"loc": "upper left"},
        color=["lightgreen", "lightblue"],
        edgecolor="black",
        linewidth=0.1,
    )
    aoi.plot(ax=ax, edgecolor="red", linewidth=5, facecolor="none")

    return ax

plot_basic_map()
# %%
################################################################################
# shapely method

# %%
# Extract the shapely geometries
aoi_geom = aoi.geometry.iloc[0]
land_geom = landcover_100m_binary.loc["land", "geometry"]
water_geom = landcover_100m_binary.loc["water", "geometry"]

# Buffer
max_distance = np.sqrt(2) * tile_width / 2
target_zone = aoi_geom.buffer(-max_distance)

# %%
# Pick a random point in the target zone
# Surprising as it is, guess-and-check appears to be the standard way to do this
rand_point = None
while rand_point is None:
    candidate_point = shp.geometry.Point(
        rng.uniform(*target_zone.bounds[::2]),
        rng.uniform(*target_zone.bounds[1::2]),
    )
    print(f"Trying point {candidate_point}")
    if target_zone.contains(candidate_point):
        rand_point = candidate_point

# %%
# Pick a rotation angle
rotation_angle = rng.uniform(0, 360)

# Make a square around rand_point and rotate it
square = shp.geometry.box(*rand_point.buffer(tile_width / 2).bounds)
square_center_pt = shp.geometry.Point(square.centroid)
square_bounds = square.bounds
square = shp.affinity.rotate(square, rotation_angle, origin=square_center_pt)

# %%
# Clip the landcover data to the square
square_land = land_geom.intersection(square)
square_water = water_geom.intersection(square)

# %%
# Plot
ax = plot_basic_map()
# fig, ax = plt.subplots(figsize=(10, 10))
for geoms in [aoi_geom, land_geom, water_geom]:
    for geom in geoms.geoms:
        # facecolor = {
        #     aoi_geom: "none",
        #     land_geom: "lightgreen",
        #     water_geom: "lightblue",
        # }
        xs, ys = geom.exterior.xy
        # ax.fill(xs, ys, alpha=0.5, ec="none", fc=facecolor[geoms])
        ax.plot(xs, ys)

ax.plot(*target_zone.exterior.xy)
ax.plot(*rand_point.xy, "ro", markersize=10)

for geoms in [square_land, square_water]:
    for geom in geoms.geoms:
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys)

ax.plot(*square.exterior.xy)

ax.set_title(f"rng seed: {rand_seed}, tile width: {tile_width} m, rotation angle: {rotation_angle}")
    
# %%
# Unrotate the square data
land_unrotated = shp.affinity.rotate(square_land, -rotation_angle, origin=square_center_pt)
water_unrotated = shp.affinity.rotate(square_water, -rotation_angle, origin=square_center_pt)

# %%
# Rasterize
t_size = tile_raster_size
lookup = {
    1: water_unrotated,
    2: land_unrotated,
}
tile_data_raster = features.rasterize(
    [(g, l) for l, g in lookup.items()],
    out_shape=(t_size, t_size),
    transform=rio.transform.from_bounds(*square_bounds, t_size, t_size),
    all_touched=False,
    fill=0,
    default_value=1,
    dtype="uint8",
)

# plt.imshow(tile_data_raster, cmap="gray")

# %%
# Plot unrotated
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.flipud(tile_data_raster), cmap="gray", origin="lower")
for geoms in [land_unrotated, water_unrotated]:
    for geom in geoms.geoms:
        # Translate to (0, 0)
        geom = shp.affinity.translate(geom, xoff=-square_bounds[0], yoff=-square_bounds[1])

        # Scale the coordinates to the raster size
        geom = shp.affinity.scale(geom, xfact=t_size / tile_width, yfact=t_size / tile_width, origin=(0, 0))

        # Translate again half a pixel to line up with the raster
        geom = shp.affinity.translate(geom, xoff=-0.5, yoff=-0.5)

        xs, ys = geom.exterior.xy
        # print(type(ys))
        # ys = t_size - ys
        ax.plot(xs, ys)
        # ax.fill(xs, ys, alpha=0.5, ec='none', fc=facecolor)

ax.set_title(f"rng seed: {rand_seed}, tile width: {tile_width} m, final resolution: {t_size} px")


# %%

# %%
################################################################################
# Geopandas method

# %%

# # Buffer
# # tile_width = 10**5
# target_zone = aoi.buffer(-tile_width // 2)

# # # Save the target zone
# # target_zone.to_file(target_zone_filepath)

# # ax = plot_basic_map()
# # target_zone.plot(ax=ax, edgecolor="orange", linewidth=5, facecolor="none")

# # %%
# # Pick a random point in the inner aoi
# rand_points = target_zone.sample_points(1, random_state=rng)
# rand_point = rand_points.iloc[0]

# # %%
# # Pick a rotation angle
# rotation_angle = rng.uniform(0, 360)

# # %%
# # Make a square around rand_point and rotate it
# square = shp.geometry.box(*rand_point.buffer(tile_width / 2).bounds)
# square = shp.affinity.rotate(square, rotation_angle, origin="center")

# # %%
# # clip the landcover data to the square
# square_data_vector = gpd.clip(landcover_100m_binary, square)

# # %%
# # Plot
# ax = plot_basic_map()
# target_zone.plot(ax=ax, edgecolor="orange", linewidth=5, facecolor="none")
# gpd.GeoSeries([square]).plot(ax=ax, edgecolor="purple", linewidth=5, facecolor="none")
# gpd.GeoSeries([rand_point]).plot(ax=ax, edgecolor="red", linewidth=5, facecolor="none")

# square_data_vector.plot(ax=ax, legend=True, color=["lightgreen", "lightblue"], edgecolor="black", linewidth=0.5)

# # %%
# # Unrotate the geodataframe
# tile_vectors = gpd.GeoDataFrame(
#     {
#         "geometry": square_data_vector.geometry.rotate(-rotation_angle, origin="center"),
#         "label": square_data_vector.index,
#     }
# )

# # Convert the square to a raster
# t_size = tile_raster_size
# lookup = {
#     "land": 2,
#     "water": 1,
# }
# vector_list = [(g, lookup[l]) for g, l in zip(tile_vectors.geometry, tile_vectors.index)]
# tile_data_raster = features.rasterize(
#     vector_list,
#     out_shape=(t_size, t_size),
#     transform=rio.transform.from_bounds(*square.bounds, t_size, t_size),
#     all_touched=True,
#     fill=0,
#     default_value=1,
#     dtype="uint8",
# )

# # Plot the raster
# plt.imshow(tile_data_raster, cmap="gray")

# # %%