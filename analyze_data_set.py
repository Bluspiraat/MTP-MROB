import json
import rasterio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def make_pixel_count(brt_folder, class_map_file, pixel_count_file):
    # Open class map
    with open(class_map_file) as f:
        class_map = json.load(f)

    # Setup dictionary to count values
    classes_count = {k: 0 for k in range(len(class_map.keys()))}

    brt_files = os.listdir(brt_folder)

    for i in tqdm(range(len(brt_files)), desc=f"Grouping pixels", unit="patch", leave=False):
        brt_file = brt_files[i]
        with rasterio.open(brt_folder + brt_file) as src:
            class_data = src.read(1)
            unique, counts = np.unique(class_data, return_counts=True)
            for k, v in dict(zip(unique, counts)).items():
                classes_count[k] += int(v)

    with open(pixel_count_file, 'w') as dump_file:
        json.dump(classes_count, dump_file)

def calculate_fractions(pixel_count_file, class_map_file, title):
    with open(pixel_count_file, 'r') as f:
        pixel_count = json.load(f)

    with open(class_map_file, 'r') as f:
        class_map = json.load(f)

    total_pixels = sum(pixel_count.values())
    percentages = []
    for k, v in class_map.items():
        percentages.append(round(pixel_count[str(v)] / total_pixels*100, 2))

    classes = list(class_map.keys())
    colors = [
        "white", "black", "dimgray", "darkgray", "darkmagenta", "firebrick", "orange", "forestgreen", "lightgreen",
        "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"
    ]

    # Bar positions
    x = np.arange(len(classes))

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, percentages, color=colors)

    # Add percentage labels above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{percentages[i]}%",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(percentages) + 10)  # leave room for text

    plt.tight_layout()
    plt.show()


def classes_combined(files, output_file):
    jsons = []
    for file in files:
        with open(file, 'r') as f:
            jsons.append(json.load(f))
    json_combined = {k: 0 for k, v in jsons[0].items()}
    for json_file in jsons:
        print(f'Processing {json_file}')
        for k, v in json_file.items():
            print(f'Processing {k}: {v}')
            json_combined[k] += v

    print(f'Finished {json_combined}')

    with open(output_file, 'w') as dump_file:
        json.dump(json_combined, dump_file)


if __name__ == '__main__':

    folder = 'train'
    class_map_file = "C:/MTP-Data/dataset_diverse_2022_512/class_map.json"
    pixel_count_file = f"C:/MTP-Data/dataset_diverse_2022_512_sep/{folder}/pixel_count.json"
    brt_folder = f"C:/MTP-Data/dataset_diverse_2022_512_sep/{folder}/brt/"
    title = f"Class distribution of folder {folder}"

    make_pixel_count(brt_folder, class_map_file, pixel_count_file)
    calculate_fractions(pixel_count_file, class_map_file, title)

    # output_file = 'C:/MTP-Data/dataset_diverse_2022_512/total_pixel_count.json'
    # calculate_fractions(output_file, class_map_file, f'Class distribution entire dataset')
    # classes_combined(files, output_file)
