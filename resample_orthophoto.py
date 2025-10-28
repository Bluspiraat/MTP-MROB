import rasterio
import glob
import os
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform


def convert_tile(input_path, output_path, resolution):
    with rasterio.open(input_path) as src:
        width_m = src.bounds[2] - src.bounds[0]
        height_m = src.bounds[3] - src.bounds[1]
        target_width = int(width_m / resolution)
        target_height = int(height_m / resolution)

        transform, width, height = calculate_default_transform(
            src_crs=src.crs,
            dst_crs=src.crs,
            width=src.width,
            height=src.height,
            dst_width=target_width,
            dst_height=target_height,
            left=src.bounds[0],
            right=src.bounds[2],
            bottom=src.bounds[1],
            top=src.bounds[3]
        )

        if os.path.exists(output_path):
            os.remove(output_path)

        data = src.read(
            out_shape=(src.count, target_width, target_height),
            resampling=resampling_method
        )

    with rasterio.open(output_path,
                       "w",
                       driver="GTiff",
                       height=target_height,
                       width=target_width,
                       resolution=resolution,
                       count=3,
                       dtype=src.dtypes[0],
                       crs=src.crs,
                       transform=transform,
                       compress="jpeg",
                       jpeg_quality=90  # 90 is usually visually indistinguishable from original
                       ) as dst:
        dst.write(data)


if __name__ == "__main__":
    goal_resolution = 0.1
    input_folder = "C:/MTP-Data/raw_datasets/vierhouten/ortho"
    output_folder = "vierhouten_10k/"
    resampling_method = Resampling.nearest  # Options: bilinear, average, etc
    images = glob.glob(input_folder + "/*.tif")
    indices = range(len(images))
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    for image, index in zip(images, indices):
        print(index)
        convert_tile(image, f'{output_folder}/{index}.tif', goal_resolution)
