import glob
import matplotlib.pyplot as plt
import rasterio
from math import floor
import numpy as np

class Orthophoto_Tile():
    def __init__(self, x_min, y_min, image_location):
        self.tile_size = 1000
        self.x_min = x_min
        self.y_min = y_min
        self.image_location = image_location
        #ToDo: Implement resolution

    def converted_image(self):
        with rasterio.open(self.image_location) as src:
            img_read = src.read([1, 2, 3])  # RGB channels (1=R, 2=G, 3=B) # Export information bands and puts them into one an array shaped like: (RGB, height, width) its a height x width array of 3 item arrays.
            converted_image = img_read.transpose(1, 2, 0)  # Reorder np.array to height, width, RGB)
        return converted_image

    def show_image(self):
        extent = [self.x_min, self.x_min + self.tile_size, self.y_min, self.y_min + self.tile_size]
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
        image_name_str = image_name.split('_')
        orthophoto_tiles.append(Orthophoto_Tile(int(image_name_str[1]), int(image_name_str[2]), image_name))

    return orthophoto_tiles

    
# Get tiles and place them in a dictionary with key based on their position, every tile should be placed at their corresponding key value pair
def _get_orthophoto_grid(orthophoto_tiles, lb_hor, ub_hor, lb_ver, ub_ver):
    orthophoto_grid = {}
    for photo in orthophoto_tiles:
        if lb_hor <= photo.x_min <= ub_hor and lb_ver <= photo.y_min <= ub_ver:
            orthophoto_grid[((photo.x_min-lb_hor)*20, (photo.y_min-lb_ver)*20)] = photo
    print("Orthophoto grid size created with number of tiles: " + str(len(orthophoto_grid)))
    return orthophoto_grid


# Create image in which they will be stitched
def _get_stitched_image(orthophoto_grid, lb_hor, ub_hor, lb_ver, ub_ver):
    stitched_image = np.zeros([(int(ub_ver/1000) - int(lb_ver/1000) + 1)*20000,
                               (int(ub_hor/1000)-int(lb_hor/1000) + 1)*20000,
                               3], dtype=np.uint8)

    # Stitch images
    for key in orthophoto_grid.keys():
        stitched_image[key[1]:(key[1]+20000), key[0]:(key[0]+20000)] = orthophoto_grid[key].converted_image()

    return stitched_image


# Extracts tiles and forms the requested image.
# Default values: x_min=251000, x_max=251999, y_min=471000, y_max=471999
def get_image(photo_folder, output_name, minx, miny, maxx, maxy):
    # Determine bounds
    lb_hor = int(floor(minx/1000))*1000
    ub_hor = int(floor(maxx/1000))*1000
    lb_ver = int(floor(miny/1000))*1000
    ub_ver = int(floor(maxy/1000))*1000

    # Create orthophoto tiles
    orthophoto_tiles = _get_orthophoto_tiles(photo_folder)

    # Create orthophoto grid
    orthophoto_grid = _get_orthophoto_grid(orthophoto_tiles, lb_hor, ub_hor, lb_ver, ub_ver)

    # Create stitched image
    stitched_image = _get_stitched_image(orthophoto_grid, lb_hor, ub_hor, lb_ver, ub_ver)

    # Subset the requested area
    minx, maxx = (minx-lb_hor)*20, (maxx-lb_hor)*20
    miny, maxy = (miny-lb_ver)*20, (maxy-lb_ver)*20
    flipped_stitch = np.flipud(stitched_image)
    subset = flipped_stitch[miny:maxy, minx:maxx]
    plt.imshow(subset)
    plt.savefig(output_name + ".jpg")
    print("Saved output image to: " + output_name + ".jpg")


