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

# Extracts tiles and forms the requested image.
def get_image(photo_list, x_min, x_max, y_min, y_max):
    # Determine bounds
    lower_bound_horizontal = int(floor(x_min/1000))*1000
    upper_bound_horizontal = int(floor(x_max/1000))*1000
    lower_bound_vertical = int(floor(y_min/1000))*1000
    upper_bound_vertical = int(floor(y_max/1000))*1000

    orthophoto_grid = {}

    # Get tiles and place them in a dictionary with key based on their position, every tile should be placed at their corresponding key value pair
    for photo in photo_list:
        if lower_bound_horizontal <= photo.x_min <= upper_bound_horizontal and lower_bound_vertical <= photo.y_min <= upper_bound_vertical:
            orthophoto_grid[((photo.x_min-lower_bound_horizontal)*20, (photo.y_min-lower_bound_vertical)*20)] = photo

    # Create image in which they will be stitched
    stitched_image = np.zeros([(int(upper_bound_vertical/1000) - int(lower_bound_vertical/1000) + 1)*20000,
                               (int(upper_bound_horizontal/1000)-int(lower_bound_horizontal/1000) + 1)*20000,
                               3], dtype=np.uint8)

    # Stitch images
    for key in orthophoto_grid.keys():
        stitched_image[key[1]:(key[1]+20000), key[0]:(key[0]+20000)] = orthophoto_grid[key].converted_image()

    # Subset the requested area
    x_min, x_max = (x_min-lower_bound_horizontal)*20, (x_max-lower_bound_horizontal)*20
    y_min, y_max = (y_min-lower_bound_vertical)*20, (y_max-lower_bound_vertical)*20
    flipped_stitch = np.flipud(stitched_image)
    subset = flipped_stitch[y_min:y_max, x_min:x_max]
    return np.flipud(subset), x_min, x_max, y_min, y_max


if __name__ == "__main__":
    FILE_FOLDER = "Data/AHN and Ortho/hwh-ortho/2025"
    images = glob.glob(FILE_FOLDER + "/*.tif")
    orthophoto_tiles = []

    # Import images into orthophoto tiles from 251 - 259k horizontally 471 - 478k vertically (Rijksdriehoek coordinates)
    for image_name in images:
        image_name_str = image_name.split('_')
        orthophoto_tiles.append(Orthophoto_Tile(int(image_name_str[1]), int(image_name_str[2]), image_name))

    # Test platform
    plt.imshow(get_image(orthophoto_tiles, x_min=251350, x_max=252500, y_min=471000, y_max=471500))
    plt.show()


