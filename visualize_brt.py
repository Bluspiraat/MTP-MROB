import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import json
import numpy as np

# Input GML
gml_wegdeel = ("Data/BRT/top10nl_wegdeel.gml", "verhardingstype")
gml_terrein = ("Data/BRT/top10nl_terrein.gml", "typeLandgebruik")
gml_waterdeel = ("Data/BRT/top10nl_waterdeel.gml", "typeWater")
gml_spoorbaandeel = ("Data/BRT/top10nl_spoorbaandeel.gml", "typeSpoorbaan")

# This order determines solving class conflicts, the last in order will the most important class
gmls = [gml_terrein, gml_waterdeel, gml_spoorbaandeel, gml_wegdeel]

# Input dictionaries
class_map_file = "Data/BRT/class_map.json"
class_grouping_file = "Data/BRT/class_grouping.json"

# Open the class map and grouping dictionaries
with open(class_map_file, 'r') as f:
    class_map = json.load(f)

with open(class_grouping_file, 'r') as f:
    class_grouping = json.load(f)

# Bounding box (RD New coordinates)
minx, miny = 251000, 471000  # 251000, 471000
maxx, maxy = 252000, 472000  # 259000, 478000
clip_geom = box(minx, miny, maxx, maxy)

# Load dataframes into list, ensure CRS (RD-New), clip to specified bounding box and group classes
gdf_list = []
for gml_file, attribute in gmls:
    gdf_new = gpd.read_file(gml_file)
    if gdf_new.crs is None:
        gdf_new = gdf_new.set_crs("EPSG:28992")
    gdf_new = gdf_new.clip(clip_geom)
    attribute_count_old = len(gdf_new[attribute].unique())
    gdf_new[attribute] = gdf_new[attribute].map(class_grouping)
    attribute_count_new = len(gdf_new[attribute].unique())
    print("For file: " + str(gml_file) + " New bounds after clipping: " + str(
        gdf_new.total_bounds) + ". The total number of distinct classes was: " + str(
        attribute_count_old) + " and is now: " + str(attribute_count_new))
    gdf_list.append(gdf_new)

# (5 cm = 0.05 m)
res = 0.05
width = int((maxx - minx) / res)
height = int((maxy - miny) / res)

# Affine transform
transform = from_origin(minx, maxy, res, res)

# prepare shapes by combining geometry and type of land usage, the result is a tuple with geometry object and class index.
shapes_list = []
for gdf, gml in zip(gdf_list, gmls):
    shapes_temp = ((geom, class_map[class_value]) for geom, class_value in zip(gdf.geometry, gdf[gml[1]]))
    shapes_list.append(shapes_temp)

print("Converted all shapes to integer representations")

# Function to rasterize
def rasterize_gdf(shapes, transform, width, height):
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    return raster


final_raster = np.zeros((height, width), dtype=np.uint8)
for shapes in shapes_list:
    raster_layer = rasterize_gdf(shapes, transform, width, height)
    final_raster = np.where(raster_layer > 0, raster_layer, final_raster)

unique_indices = np.unique(final_raster)
class_map_inverted = {v: k for k, v in class_map.items()}
labels_present = [class_map_inverted[index] for index in unique_indices]

print("Rasterized and merges all provided layers")
print("Present labels are: " + str(labels_present))

# Save raster
out_tif = "rasterized_BRT_5cm.tif"
meta = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": final_raster.dtype,
    "crs": "EPSG:28992",
    "transform": transform,
}

with rasterio.open(out_tif, "w", **meta) as dst:
    dst.write(final_raster, 1)

print(f"Raster saved to {out_tif}")
