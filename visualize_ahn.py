import glob
from itertools import product
import rasterio
import numpy as np
import rasterio
from scipy.ndimage import zoom

class AHN_Tile():
    def __init__(self, ahn_file_location):
        self.minx = 0
        self.miny = 0
        self.maxx = 0
        self.maxy = 0
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.file_location = ahn_file_location
        self._extract_ahn_information(ahn_file_location)

    def _extract_ahn_information(self, ahn_file_location):
        with rasterio.open(ahn_file_location) as src:
            self.width = src.width
            self.height = src.height
            self.resolution = src.res[0]
            self.minx = int(src.bounds[0])
            self.miny = int(src.bounds[1])
            self.maxx = int(src.bounds[2])
            self.maxy = int(src.bounds[3])

    def get_ahn_data(self):
        with rasterio.open(self.file_location) as src:
            elevation = src.read(1)
            nodata_entries = src.nodata

            # If there are values with no data, then fill those entries with nan
            if nodata_entries is not None:
                elevation[elevation == nodata_entries] = np.nan
        return elevation

def _get_ahn_tiles(ahn_file_folder):
    ahn_tiles = []
    for file in glob.glob(ahn_file_folder + '*.tif'):
        ahn_tiles.append(AHN_Tile(file))
    # print("Number of AHN tiles found in folder and thus created: " + str(len(ahn_tiles)))
    return ahn_tiles

def _find_origin(ahn_tiles):
    origin_x = ahn_tiles[0].minx
    origin_y = ahn_tiles[0].miny
    for ahn_tile in ahn_tiles:
        if ahn_tile.minx < origin_x or ahn_tile.miny < origin_y:
            origin_x = ahn_tile.minx
            origin_y = ahn_tile.miny
    return origin_x, origin_y

def _get_ahn_grid(ahn_tiles, minx, miny, maxx, maxy):
    width_m = ahn_tiles[0].width*ahn_tiles[0].resolution
    height_m = ahn_tiles[0].height*ahn_tiles[0].resolution
    origin_x, origin_y = _find_origin(ahn_tiles)

    ahn_grid = {}
    for tile in ahn_tiles:
        if not (tile.maxx <= minx or tile.minx >= maxx or
                tile.maxy <= miny or tile.miny >= maxy):
            dict_x = int((tile.minx - origin_x) / width_m)
            dict_y = int((tile.miny - origin_y) / height_m)
            ahn_grid[dict_x, dict_y] = tile
    # print("AHN grid size created with number of tiles: " + str(len(ahn_grid)))
    return ahn_grid


def _get_stitched_ahn(ahn_grid):
    max_i = max(i for i, j in ahn_grid.keys())
    max_j = max(j for i, j in ahn_grid.keys())

    # Calculate height and width of a patch for all patches/tiles
    ahn_tile_example = next(iter(ahn_grid.values()))
    width = int((max_i + 1) * ahn_tile_example.width)
    height = int((max_j + 1) * ahn_tile_example.height)

    stitched_image = np.zeros((height, width), dtype=np.float32)

    # place tiles: invert vertical index so j=0 (bottom) maps to bottom of array
    for (i, j), ahn_tile in ahn_grid.items():
        tile_data = ahn_tile.get_ahn_data()  # shape: (tile_height_px, tile_width_px)

        row_start = (max_j - j) * tile_data.shape[0]  # rows
        col_start = i * tile_data.shape[1]  # columns
        stitched_image[row_start:row_start + tile_data.shape[0], col_start:col_start + tile_data.shape[1]] = tile_data

    return stitched_image

def _export_ahn_subset(output_name, ahn_grid, minx, maxy, resolution):
    crs = "EPSG:28992"
    height, width = ahn_grid.shape
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

    with rasterio.open(
            output_name + ".tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=ahn_grid.dtype,
            crs=crs,
            transform=transform,
            compress="ZSTD"
    ) as dst:
        dst.write(ahn_grid.astype('float16'), 1)

    # print("AHN tile exported as " + output_name + ".tif")

def get_ahn_data(ahn_file_folder, output_name, minx, miny, maxx, maxy):
    ahn_tiles = _get_ahn_tiles(ahn_file_folder)
    ahn_grid = _get_ahn_grid(ahn_tiles, minx, miny, maxx, maxy)
    stitched_image = _get_stitched_ahn(ahn_grid)

    # Calculate pixels per meters
    ahn_tile_example = ahn_tiles[0]
    m_to_p = int(1/ahn_tile_example.resolution)
    res = ahn_tile_example.resolution
    tile_width_m, tile_height_m = ahn_tile_example.width*res, ahn_tile_example.height*res

    # Determine bounds (meters conversion to pixels count) and get vertical length for flipping
    lb_hor = (minx // tile_width_m) * tile_width_m
    lb_ver = (miny // tile_height_m) * tile_height_m
    vertical_length = stitched_image.shape[0]

    # Subset image
    minx_subset, maxx_subset = int((minx - lb_hor) * m_to_p), int((maxx - lb_hor) * m_to_p)
    maxy_subset, miny_subset = int(vertical_length - (miny - lb_ver) * m_to_p), int(
        vertical_length - (maxy - lb_ver) * m_to_p)
    subset = stitched_image[miny_subset:maxy_subset, minx_subset:maxx_subset]

    # Upscale image and increase resolution
    zoom_factor = 10
    subset_highres = zoom(subset, zoom_factor, order=1)

    _export_ahn_subset(output_name, subset_highres, minx, maxy, res/zoom_factor)

def _check_tile_continuity(boundaries):
    # First check if there are now row or column gaps
    x_coords = sorted(set([boundary[0] for boundary in boundaries]))
    x_interval = [b - a for a, b in zip(x_coords, x_coords[1:])]
    assert len(set(x_interval)) == 1, "A column of tiles is missing creating a gap"

    y_coords = sorted(set([boundary[1] for boundary in boundaries]))
    y_interval = [b - a for a, b in zip(y_coords, y_coords[1:])]
    assert len(set(y_interval)) == 1, "A column of tiles is missing creating a gap"

    # Create grid with tiles
    pairs = {(x, y): False for x, y in product(x_coords, y_coords)}
    for boundary in boundaries:
        pairs[(boundary[0], boundary[1])] = True

    assert all(pairs.values()), "The grid is incomplete: "+str(list(pairs.values()).count(False))+" tiles are missing"

def get_ahn_grid_boundaries(ahn_folder):
    tiles = _get_ahn_tiles(ahn_folder)
    boundaries = [[tile.minx, tile.miny, tile.maxx, tile.maxy] for tile in tiles]
    minx_values, miny_values, maxx_values, maxy_values = zip(*boundaries)
    _check_tile_continuity(boundaries)
    return min(minx_values), min(miny_values), max(maxx_values), max(maxy_values)


