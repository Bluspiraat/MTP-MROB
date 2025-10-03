import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import json
import numpy as np


def _get_dictionaries():
    # Input dictionaries
    class_map_file = "Data/BRT/class_map.json"
    class_grouping_file = "Data/BRT/class_grouping.json"

    # Open the class map and grouping dictionaries
    with open(class_map_file, 'r') as f:
        class_map = json.load(f)

    with open(class_grouping_file, 'r') as f:
        class_grouping = json.load(f)

    return class_map, class_grouping


# Load dataframes into list, ensure CRS (RD-New), clip to specified bounding box and group classes
def _create_gdf_list(gmls, clip_geometry, class_grouping):
    gdf_list = []
    for gml_file, attribute in gmls:
        gdf_new = gpd.read_file(gml_file)
        if gdf_new.crs is None:
            gdf_new = gdf_new.set_crs("EPSG:28992")
        gdf_new = gdf_new.clip(clip_geometry)
        attribute_count_old = len(gdf_new[attribute].unique())
        gdf_new[attribute] = gdf_new[attribute].map(class_grouping)
        attribute_count_new = len(gdf_new[attribute].unique())
        print("For file: " + str(gml_file) + " New bounds after clipping: " + str(
            gdf_new.total_bounds) + ". The total number of distinct classes was: " + str(
            attribute_count_old) + " and is now: " + str(attribute_count_new))
        gdf_list.append(gdf_new)
    return gdf_list


# Prepare shapes by combining geometry and type of land usage, the result is a tuple with geometry object and class
# index.
def _obtain_shapes(gdf_list, gmls, class_map):
    shapes_list = []
    for gdf, gml in zip(gdf_list, gmls):
        shapes_temp = ((geom, class_map[class_value]) for geom, class_value in zip(gdf.geometry, gdf[gml[1]]))
        shapes_list.append(shapes_temp)
    print("Converted all shapes to integer representations")
    return shapes_list


# Function to rasterize
def _rasterize_gdf(shapes, transform, width, height):
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    return raster


def _create_flattened_raster(shapes_list, transform, height, width):
    final_raster = np.zeros((height, width), dtype=np.uint8)
    for shapes in shapes_list:
        raster_layer = _rasterize_gdf(shapes, transform, width, height)
        final_raster = np.where(raster_layer > 0, raster_layer, final_raster)
    return final_raster


def _get_meta_data(height, width, final_raster, transform):
    meta_data = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": final_raster.dtype,
        "crs": "EPSG:28992",
        "transform": transform,
    }
    return meta_data


# gml_files order: terrein, waterdeel, spoorbaandeel, wegdeel.
def create_BRT_export(gml_files, resolution, output_name, minx, miny, maxx, maxy):
    gml_terrein = (gml_files[0], "typeLandgebruik")
    gml_waterdeel = (gml_files[1], "typeWater")
    gml_spoorbaandeel = (gml_files[2], "typeSpoorbaan")
    gml_wegdeel = (gml_files[3], "verhardingstype")
    gmls = [gml_terrein, gml_waterdeel, gml_spoorbaandeel, gml_wegdeel]

    # Get dictionaries
    class_map, class_grouping = _get_dictionaries()

    # Create clip geometry
    clip_geom = box(minx, miny, maxx, maxy)

    # Create geodataframes
    gdf_list = _create_gdf_list(gmls, clip_geom, class_grouping)

    # Create transform based on resolution given in meters
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Affine transform
    transform = from_origin(minx, maxy, resolution, resolution)

    # Obtain shapes list
    shapes_list = _obtain_shapes(gdf_list, gmls, class_map)

    # Create final raster
    final_raster = _create_flattened_raster(shapes_list, transform, height, width)
    meta_data = _get_meta_data(height, width, final_raster, transform)

    # Export raster
    with rasterio.open(output_name + ".tif", "w", **meta_data, compress="lzw") as dst:
        dst.write(final_raster, 1)

    print(f"Raster saved to {output_name + ".tif"}")

    # Report on the number of unique labels present in the final export
    unique_indices = np.unique(final_raster)
    class_map_inverted = {v: k for k, v in class_map.items()}
    labels_present = [class_map_inverted[index] for index in unique_indices]

    print("Rasterized and merges all provided layers")
    print("Present labels are: " + str(labels_present))







