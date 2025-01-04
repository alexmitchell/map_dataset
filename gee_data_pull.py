# %%
import ee
import geemap
from pathlib import Path
import time

# %%
# Config
# Note: CRS must be EPSG:4326
# geemap and ee assume it and do not read the crs from the file
aoi_path = Path("aoi/greece_aoi.geojson")

gdrive_dirname = "map_dataset_tests"


# %%
# Connect to Google Earth Engine
ee.Authenticate()
ee.Initialize()

# %%
# Load the aoi
aoi_ee = geemap.geojson_to_ee(aoi_path.as_posix())

# Display aoi to double check
map = geemap.Map()
map.centerObject(aoi_ee, 7)
map.addLayer(aoi_ee.style(**{'color': 'red', 'width': 2, 'fillColor': '00000000'}), {}, 'AOI')
map


# %%
# Load landcover data
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global
landcover_100m_raw = (
    ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
    .select('discrete_classification')
)

# # Make an image where any holes are filled with the nearest value
# landcover_100m_filled = landcover_100m_raw.focal_mode(radius=2.5)

# Clean up the landcover data
landcover_100m = (
    landcover_100m_raw
    # .unmask(landcover_100m_filled)
    .clip(aoi_ee)
)

# %%
# Load 30m DEM
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30
dem_30m_ic = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")

# Mosiac the DEM and fix the projection
# Note, the GLO30 dataset is a composite that requires reprojection after mosaicking
dem_proj = dem_30m_ic.first().projection()
dem_30m = dem_30m_ic.mosaic().setDefaultProjection(dem_proj)

# Fill no data values with 0 and clip to the aoi
dem_30m = dem_30m.unmask(0).clip(aoi_ee)

# Generate a hillshade
hillshade = ee.Terrain.hillshade(dem_30m)


# %%
# Identify land vs not land
# Not land is ocean (200) or other permanent water bodies (80)
# For now just exclude ocean because other water bodies mixes marshes and lakes
landcover_100m_binary = landcover_100m.neq(200)#.And(landcover_100m.neq(80))

# Polygonize the landcover 100m binary
landcover_100m_binary_vector = landcover_100m_binary.reduceToVectors(
    geometry=aoi_ee,
    geometryType='polygon',
    scale=100,
    maxPixels=1e13,
    reducer=ee.Reducer.countEvery(),

)


# %%
# Display all layers
map = geemap.Map()
map.centerObject(aoi_ee, 7)
# map.addLayer(dem_30m, {'min': 0, 'max': 3000}, 'DEM 30m')
map.addLayer(hillshade, {}, 'Hillshade')

# Add landcover data with opacity 50%
map.addLayer(landcover_100m, {}, 'Landcover')
map.addLayer(landcover_100m_binary, {}, 'Landcover (binary)')

map.addLayer(landcover_100m_binary_vector, {}, 'Landcover (vector)')

# Display AOI as an empty box
map.addLayer(aoi_ee.style(**{'color': 'red', 'width': 2, 'fillColor': '00000000'}), {}, 'AOI')
map

# %%
# Define a function to export a vector to Google Drive
def export_vector(vector, filename, wait=True):
    if gdrive_dirname is None or gdrive_dirname == "":
        print("No drive folder specified, skipping export")
        return

    # Export the vector data to a geojson on Google Drive
    task = ee.batch.Export.table.toDrive(
        collection=vector,
        description=filename,
        folder=gdrive_dirname,
        fileNamePrefix=filename,
        fileFormat='GeoJSON',
    )

    task.start()

    if wait:
        wait_for_tasks([task])

    return task

# Define a function to wait for tasks to complete
def wait_for_tasks(task_ids: list[str]):
    # Wait for the task to complete
    print(f"Exporting {len(task_ids)} item(s) to Google Drive")
    all_status = ee.data.getTaskStatus(task_ids)
    running_states = ['READY', 'RUNNING']
    while any(task['state'] in running_states for task in all_status):
        print("Checking task status")
        all_status = ee.data.getTaskStatus(task_ids)
        remaining_tasks = [s for s in all_status if s['state'] in running_states]
        print(f"Remaining tasks: {len(remaining_tasks)}")
        time.sleep(20)
    
    print("All tasks completed")

# %%
# Export the landcover 100m binary vector
task = export_vector(landcover_100m_binary_vector, "landcover_100m_binary_vector")


# %%

# Get a list of all tasks
all_tasks = ee.batch.Task.list()
all_tasks


# %%
