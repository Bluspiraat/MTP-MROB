import os

from tqdm import tqdm

from visualize_brt import create_BRT_export, get_brt_boundaries
from visualize_orthophoto import get_image, get_orthophoto_grid_boundaries
from visualize_ahn import get_ahn_data, get_ahn_grid_boundaries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.windows import Window
import json


def visualize_ahn(ahn_file):
    with rasterio.open(ahn_file) as src:
        elevation_data = src.read(1)
        resolution = src.res
    plt.imshow(elevation_data)
    plt.title('Digital surface model')
    plt.show()


def visualize_brt(brt_file, class_map_file):
    colors = [
        "white", "black", "dimgray", "darkgray", "darkmagenta",
        "darkred", "firebrick", "orange", "forestgreen", "lightgreen",
        "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"
    ]
    cmap = mcolors.ListedColormap(colors)

    with rasterio.open(brt_file) as src:
        class_data = src.read(1)

    plt.imshow(class_data, cmap, vmin=0, vmax=14)
    plt.title('Classes coloured')
    plt.show()


def visualize_orthophoto(orthophoto_file):
    with rasterio.open(orthophoto_file) as src:
        img_read = src.read([1, 2,
                             3])  # RGB channels (1=R, 2=G, 3=B) # Export information bands and puts them into one an array shaped like: (RGB, height, width) its a height x width array of 3 item arrays.
        converted_image = img_read.transpose(1, 2, 0)  # Reorder np.array to height, width, RGB)
    plt.imshow(converted_image)
    plt.title('Orthophoto image')
    plt.show()


def visualize(orthophoto_file, AHN_file, BRT_file, class_map):
    visualize_ahn(AHN_file)
    visualize_brt(BRT_file, class_map)
    visualize_orthophoto(orthophoto_file)

def _check_boundaries(minx, miny, maxx, maxy, ahn_folder, orthophoto_folder, gml_files_location):
    assert minx < maxx, 'Min x must be less than max x'
    assert miny < maxy, 'Min y must be less than max y'

    ahn_boundaries = get_ahn_grid_boundaries(ahn_folder)
    assert minx >= ahn_boundaries[0], 'Min x must be more than ahn boundaries min x'
    assert miny >= ahn_boundaries[1], 'Min y must be more than ahn boundaries min y'
    assert maxx <= ahn_boundaries[2], 'Max x must be less than ahn boundaries max x'
    assert maxy <= ahn_boundaries[3], 'Max y must be less than ahn boundaries max y'

    ortho_boundaries = get_orthophoto_grid_boundaries(orthophoto_folder)
    assert minx >= ortho_boundaries[0], 'Min x must be more than ortho boundaries min x'
    assert miny >= ortho_boundaries[1], 'Min y must be more than ortho boundaries min y'
    assert maxx <= ortho_boundaries[2], 'Max x must be less than ortho boundaries max x'
    assert maxy <= ortho_boundaries[3], 'Max y must be less than ortho boundaries max y'

    brt_boundaries = get_brt_boundaries(gml_files_location)
    assert minx >= brt_boundaries[0], 'Min x must be more than brt boundaries min x'
    assert miny >= brt_boundaries[1], 'Min y must be more than brt boundaries min y'
    assert maxx <= brt_boundaries[2], 'Max x must be less than brt boundaries max x'
    assert maxy <= brt_boundaries[3], 'Max y must be less than brt boundaries max y'

    # Display the smallest bounds of the input data
    bounds_minx = max(ahn_boundaries[0], ortho_boundaries[0], brt_boundaries[0])
    bounds_miny = max(ahn_boundaries[1], ortho_boundaries[1], brt_boundaries[1])
    bounds_maxx = min(ahn_boundaries[2], ortho_boundaries[2], brt_boundaries[2])
    bounds_maxy = min(ahn_boundaries[3], ortho_boundaries[3], brt_boundaries[3])

    print("The bounds of the dataset are: " + str([bounds_minx, bounds_miny, bounds_maxx, bounds_maxy]))


def export_tiles(minx, miny, maxx, maxy, output_name, ahn_folder, orthophoto_folder, resolution, gml_files_location):
    _check_boundaries(min(minx), min(miny), max(maxx), max(maxy), ahn_folder, orthophoto_folder, gml_files_location)
    index = list(range(len(minx)))
    os.makedirs(os.path.dirname(output_name + "/dsm/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_name + "/ortho/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_name + "/brt/"), exist_ok=True)

    for minx, miny, maxx, maxy, index in tqdm(
            zip(minx, miny, maxx, maxy, index),
            total=len(index),  # tells tqdm how many iterations to expect
            desc="Processing tiles"
    ):
        get_ahn_data(ahn_folder, output_name + "/dsm/" + str(index), minx, miny, maxx, maxy)
        get_image(orthophoto_folder, output_name + "/ortho/" + str(index), minx, miny, maxx, maxy)
        create_BRT_export(gml_files_location, resolution, output_name + "/brt/" + str(index), minx, miny, maxx, maxy)


def create_patches(input_folder, dimension, output_folder):
    # Load input folders
    folders = ["/ortho/", "/dsm/", "/brt/"]
    ortho_tiles = os.listdir(input_folder + "/ortho/")
    dsm_tiles = os.listdir(input_folder + "/dsm/")
    brt_tiles = os.listdir(input_folder + "/brt/")

    # Create output folders
    os.makedirs(os.path.dirname(output_folder + "/dsm/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_folder + "/ortho/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_folder + "/brt/"), exist_ok=True)

    # Go over all tiles
    for modality, folder in zip([ortho_tiles, dsm_tiles, brt_tiles], folders):
        index = 1
        for tile in modality:
            with rasterio.open(input_folder + folder + tile) as src:
                width = src.width
                height = src.height
                x_steps = width // dimension
                y_steps = height // dimension
                total_patches = x_steps * y_steps

                for i in tqdm(range(total_patches), desc=f"Creating patches of {tile} of modality {folder}", unit="patch", leave=False):
                    x = i % x_steps
                    y = i // x_steps

                    window = Window(x * dimension, y * dimension, dimension, dimension)
                    patch = src.read(window=window)
                    transform = src.window_transform(window)

                    profile = src.profile
                    profile.update({
                        "height": patch.shape[1],
                        "width": patch.shape[2],
                        "transform": transform
                    })

                    with rasterio.open(output_folder + folder + str(index) + ".tif", "w", **profile) as dst:
                        dst.write(patch)

                    index += 1


if __name__ == '__main__':

    file_location = "test_export"
    start_x = 251000  # Minimum = 251000
    start_y = 471000  # Minimum = 471000
    interval = 1000
    size = 1000
    horizontal = 7
    vertical = 8

    bounds = [[start_x + i*interval, start_y + i*interval, start_x + size + i*interval, start_y + size + i*interval] for i in range(horizontal) for j in range(vertical)]

    minx, miny, maxx, maxy = zip(*bounds)

    ahn_folder = "Data/AHN and Ortho/hwh-ahn/"
    orthophoto_folder = "Data/AHN and Ortho/hwh-ortho/2025/"
    resolution = 0.05
    gml_files_location = ["Data/BRT/top10nl_terrein.gml", "Data/BRT/top10nl_waterdeel.gml",
                          "Data/BRT/top10nl_spoorbaandeel.gml", "Data/BRT/top10nl_wegdeel.gml"]


    export_tiles(minx, miny, maxx, maxy, file_location, ahn_folder, orthophoto_folder, resolution, gml_files_location)

    output_folder = "dataset"
    create_patches(file_location, 512, output_folder)

    # index = 45
    # visualize(f"dataset/ortho/{index}.tif", f"dataset/dsm/{index}.tif", f"dataset/brt/{index}.tif", "Data/BRT/class_map.json")


