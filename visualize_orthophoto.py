import glob
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from itertools import product


class Orthophoto_Tile():
    def __init__(self, image_location):
        self.image_location = image_location
        self.minx = 0
        self.miny = 0
        self.maxx = 0
        self.maxy = 0
        self.resolution = 0
        self.width_px = 0
        self.width_px = 0
        self._extract_photo_information()
        self.width_m = self.maxx - self.minx
        self.height_m = self.maxy - self.miny

    def _extract_photo_information(self):
        with rasterio.open(self.image_location) as src:
            self.resolution = src.res[0]
            self.minx = int(src.bounds[0])
            self.miny = int(src.bounds[1])
            self.maxx = int(src.bounds[2])
            self.maxy = int(src.bounds[3])
            self.width_px = src.width
            self.height_px = src.height

    def converted_image(self):
        with rasterio.open(self.image_location) as src:
            img_read = src.read([1, 2,
                                 3])  # RGB channels (1=R, 2=G, 3=B) # Export information bands and puts them into one an array shaped like: (RGB, height, width) its a height x width array of 3 item arrays.
            converted_image = img_read.transpose(1, 2, 0)  # Reorder np.array to height, width, RGB)
        return converted_image

    def show_image(self):
        extent = [self.minx, self.minx + self.width_px, self.miny, self.miny + self.height_px]
        plt.imshow(self.converted_image(), extent=extent, origin="upper")
        plt.title("Orthophoto image")
        plt.xlabel("X - coordinates")
        plt.ylabel("Y - coordinates")
        plt.show()


def _get_orthophoto_tiles(photo_folder):
    images = glob.glob(photo_folder + "/*.tif")
    orthophoto_tiles = []
    # Import images into orthophoto tiles from 251 - 259k horizontally 471 - 478k vertically (Rijksdriehoek coordinates)
    for image_name in images:
        orthophoto_tiles.append(Orthophoto_Tile(image_name))
    return orthophoto_tiles


# Get tiles and place them in a dictionary with key based on their position, every tile should be placed at their corresponding key value pair
def _get_orthophoto_grid(orthophoto_tiles, minx, maxx, miny, maxy):
    origin_x = (minx // 1000) * 1000
    origin_y = (miny // 1000) * 1000

    orthophoto_grid = {}
    for photo in orthophoto_tiles:
        if not (photo.maxx <= minx or photo.minx >= maxx or
                photo.maxy <= miny or photo.miny >= maxy):
            dict_x = int((photo.minx - origin_x) / 1000)
            dict_y = int((photo.miny - origin_y) / 1000)
            orthophoto_grid[dict_x, dict_y] = photo
    # print("Orthophoto grid size created with number of tiles: " + str(len(orthophoto_grid)))
    return orthophoto_grid


# Create stitched image
def _get_stitched_image(orthophoto_grid, width_px, height_px):
    # Max indices for each dimension
    max_i = max(i for i, j in orthophoto_grid.keys())
    max_j = max(j for i, j in orthophoto_grid.keys())

    # Determine colums and row numbers
    width = (max_i + 1) * width_px
    height = (max_j + 1) * height_px

    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)

    # place tiles: invert vertical index so j=0 (bottom) maps to bottom of array
    for (i, j), photo in orthophoto_grid.items():
        row_start = (max_j - j) * height_px
        col_start = i * width_px
        stitched_image[row_start:row_start + height_px,
            col_start:col_start + width_px] = photo.converted_image()

    return stitched_image


def _export_photo_subset(output_name, photo, minx, maxy, resolution):
    crs = "EPSG:28992"

    # Ensure array is in (bands, height, width)
    if photo.shape[2] == 3:
        photo = np.transpose(photo, (2, 0, 1))

    height, width = photo.shape[1], photo.shape[2]
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

    with rasterio.open(
            output_name + '.tif',
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=photo.dtype,
            crs=crs,
            transform=transform,
            compress="jpeg",
            jpeg_quality=90  # 90 is usually visually indistinguishable from original
    ) as dst:
        dst.write(photo)
        # print("Saved output image to: " + output_name + ".tif")


# Extracts tiles and forms the requested image.
def get_image(photo_folder, output_name, minx, miny, maxx, maxy):
    # Determine bounds (meters conversion to pixels count)
    #ToDo: Check if the hardcoded 1000 meter value could pose a problem
    lb_hor = int(minx // 1000) * 1000
    lb_ver = int(miny // 1000) * 1000

    # Create orthophoto tiles
    orthophoto_tiles = _get_orthophoto_tiles(photo_folder)
    width_px = orthophoto_tiles[0].width_px
    height_px = orthophoto_tiles[0].height_px

    # Create orthophoto grid
    orthophoto_grid = _get_orthophoto_grid(orthophoto_tiles, minx, maxx, miny, maxy)

    # Create stitched image
    stitched_image = _get_stitched_image(orthophoto_grid, width_px, height_px)
    vertical_length = stitched_image.shape[0]

    # Subset the requested area
    resolution = orthophoto_tiles[0].resolution
    m_to_p = int(1 / orthophoto_tiles[0].resolution)
    minx_subset, maxx_subset = int((minx - lb_hor) * m_to_p), int((maxx - lb_hor) * m_to_p)
    maxy_subset, miny_subset = int(vertical_length - (miny - lb_ver) * m_to_p), int(
        vertical_length - (maxy - lb_ver) * m_to_p)
    subset = stitched_image[miny_subset:maxy_subset, minx_subset:maxx_subset]
    _export_photo_subset(output_name, subset, minx, maxy, resolution)


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

    assert all(pairs.values()), "The grid is incomplete: " + str(
        list(pairs.values()).count(False)) + " tiles are missing"


def get_orthophoto_grid_boundaries(orthophoto_folder):
    tiles = _get_orthophoto_tiles(orthophoto_folder)
    boundaries = [[tile.minx, tile.miny, tile.maxx, tile.maxy] for tile in tiles]
    minx_values, miny_values, maxx_values, maxy_values = zip(*boundaries)
    _check_tile_continuity(boundaries)
    return min(minx_values), min(miny_values), max(maxx_values), max(maxy_values)
