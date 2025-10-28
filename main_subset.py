import os

from tqdm import tqdm
import geopandas as gpd
from visualize_brt import create_BRT_export
from visualize_orthophoto import get_image, get_orthophoto_grid_boundaries
from visualize_ahn import get_ahn_data, get_ahn_grid_boundaries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.windows import Window
import json
import random
import numpy as np

def visualize_ahn(ahn_file, axes):
    with rasterio.open(ahn_file) as src:
        elevation_data = src.read(1)
    mean_value = np.nanmean(elevation_data, dtype='float32')
    std_value = np.nanstd(elevation_data, dtype='float32')
    axes[1].imshow(elevation_data)
    axes[1].set_title(f'DSM with mean {mean_value:.2f} '
                      f'and stdev {std_value:.2f}')


def visualize_brt(brt_file, class_to_color_json, axes):
    with open(class_to_color_json) as f:
        class_to_color = json.load(f)
    colors = [v for k, v in class_to_color.items()]
    cmap = mcolors.ListedColormap(colors)

    with rasterio.open(brt_file) as src:
        class_data = src.read(1)

    axes[2].imshow(class_data, cmap, vmin=0, vmax=len(colors))
    axes[2].set_title('Classes coloured')


def visualize_orthophoto(orthophoto_file, axes):
    with rasterio.open(orthophoto_file) as src:
        img_read = src.read([1, 2,
                             3])  # RGB channels (1=R, 2=G, 3=B) # Export information bands and puts them into one an array shaped like: (RGB, height, width) its a height x width array of 3 item arrays.
        converted_image = img_read.transpose(1, 2, 0)  # Reorder np.array to height, width, RGB)
    axes[0].imshow(converted_image)
    axes[0].set_title(f'{orthophoto_file.split('/')[-1]}')


def visualize(orthophoto_file, AHN_file, BRT_file, class_to_color):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    visualize_ahn(AHN_file, axes)
    visualize_brt(BRT_file, class_to_color, axes)
    visualize_orthophoto(orthophoto_file, axes)
    plt.show()


def _check_boundaries(minx, miny, maxx, maxy, ahn_folder, orthophoto_folder):
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


    # Display the smallest bounds of the input data
    bounds_minx = max(ahn_boundaries[0], ortho_boundaries[0])
    bounds_miny = max(ahn_boundaries[1], ortho_boundaries[1])
    bounds_maxx = min(ahn_boundaries[2], ortho_boundaries[2])
    bounds_maxy = min(ahn_boundaries[3], ortho_boundaries[3])

    print("The bounds of the dataset are: " + str([bounds_minx, bounds_miny, bounds_maxx, bounds_maxy]))


def export_tiles(minx, miny, maxx, maxy, output_name, ahn_folder, orthophoto_folder, resolution,
                 gml_files_location, class_grouping_file, class_map_file):
    _check_boundaries(min(minx), min(miny), max(maxx), max(maxy), ahn_folder, orthophoto_folder)
    index = list(range(len(minx)))
    os.makedirs(os.path.dirname(output_name + "/dsm/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_name + "/ortho/"), exist_ok=True)
    os.makedirs(os.path.dirname(output_name + "/brt/"), exist_ok=True)

    for minx_tile, miny_tile, maxx_tile, maxy_tile, index in tqdm(
            zip(minx, miny, maxx, maxy, index),
            total=len(index),  # tells tqdm how many iterations to expect
            desc="Processing tiles"
    ):
        create_BRT_export(gml_files_location, resolution, output_name + "/brt/" + str(index),
                          minx_tile, miny_tile, maxx_tile, maxy_tile, class_grouping_file, class_map_file)
        get_image(orthophoto_folder, output_name + "/ortho/" + str(index), minx_tile, miny_tile, maxx_tile, maxy_tile)
        get_ahn_data(ahn_folder, output_name + "/dsm/" + str(index), minx_tile, miny_tile, maxx_tile, maxy_tile)



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

    tiles_output_folder = "C:/MTP-Data/datasets/soesterberg_tiles"
    patches_output_folder = "C:/MTP-Data/datasets/soesterberg_patches"
    class_map_file = "C:/MTP-Data/raw_datasets_2022/class_map.json"
    class_grouping_file = "C:/MTP-Data/raw_datasets_2022/class_grouping.json"
    start_x = 149000
    start_y = 455000
    interval = 1000  # Meter size
    size = 1000  # Meter size
    horizontal = 7
    vertical = 7

    bounds = [[start_x + i*interval, start_y + j*interval, start_x + size + i*interval, start_y + size + j*interval] for i in range(horizontal) for j in range(vertical)]

    minx, miny, maxx, maxy = zip(*bounds)

    ahn_folder = "C:/MTP-Data/raw_datasets_2022/soesterberg/ahn/"
    orthophoto_folder = "C:/MTP-Data/raw_datasets_2022/soesterberg/ortho/"
    resolution = 0.1
    gml_files_location = ["C:/MTP-Data/raw_datasets_2022/soesterberg/brt/top10nl_terrein.gml",
                          "C:/MTP-Data/raw_datasets_2022/soesterberg/brt/top10nl_waterdeel.gml",
                          "C:/MTP-Data/raw_datasets_2022/soesterberg/brt/top10nl_spoorbaandeel.gml",
                          "C:/MTP-Data/raw_datasets_2022/soesterberg/brt/top10nl_wegdeel.gml"]

    # export_tiles(minx, miny, maxx, maxy, tiles_output_folder, ahn_folder, orthophoto_folder, resolution,
    #              gml_files_location, class_grouping_file=class_grouping_file, class_map_file=class_map_file)
    #
    # create_patches(tiles_output_folder, 512, patches_output_folder)

    samples = 10
    dataset_folder = "C:/MTP-Data/dataset_diverse_2022_512/soesterberg"
    indices = [random.randint(0, 14000) for _ in range(samples)]

    for index in indices:
        visualize(f"{dataset_folder}/ortho/{index}.tif",
                  f"{dataset_folder}/dsm/{index}.tif",
                  f"{dataset_folder}/brt/{index}.tif",
                  "C:/MTP-Data/raw_datasets_2022/class_to_color.json")


