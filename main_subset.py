from visualize_brt import create_BRT_export
from visualize_orthophoto import get_image
from visualize_ahn import get_ahn_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
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
    # labels = colors
    cmap = mcolors.ListedColormap(colors)
    #
    # # Open the class map and grouping dictionaries
    # with open(class_map_file, 'r') as f:
    #     class_map = json.load(f)
    # index_to_class = {v: k for k, v in class_map.items()}

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


def visualize(AHN_file, BRT_file, class_map, orthophoto_file):
    visualize_ahn(AHN_file)
    visualize_brt(BRT_file, class_map)
    visualize_orthophoto(orthophoto_file)


if __name__ == '__main__':
    # minx = 251000  # 251000
    # miny = 471000  # 471000
    # maxx = 251500  # 251999
    # maxy = 471500  # 471999

    # Home = 254700, 474150
    minx = 253500  # 251000
    miny = 473500  # 471000
    maxx = 255500  # 251999
    maxy = 474500  # 471999

    # get AHN data
    file_folder = "Data/AHN and Ortho/hwh-ahn/"
    output_name = "AHN_data_test"
    get_ahn_data(file_folder, output_name, minx, miny, maxx, maxy)

    # Get orthophoto
    output_name = "orthophoto_test"
    orthophoto_folder = "Data/AHN and Ortho/hwh-ortho/2025"
    get_image(orthophoto_folder, output_name, minx, miny, maxx, maxy)

    # Get BRT labelled image
    output_name = "BRT_image_test"
    gml_files_location = ["Data/BRT/top10nl_terrein.gml", "Data/BRT/top10nl_waterdeel.gml",
                          "Data/BRT/top10nl_spoorbaandeel.gml", "Data/BRT/top10nl_wegdeel.gml"]
    resolution = 0.05
    create_BRT_export(gml_files_location, resolution, output_name, minx, miny, maxx, maxy)

    visualize("Exports/AHN_data_test.tif", "Exports/BRT_image_test.tif", "Data/BRT/class_map.json",
              "Exports/orthophoto_test.tif")
