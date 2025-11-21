from models import RGBUNet, MidFusionUNet
from datasets.augmentations import get_image_net_normalization
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Dict
import matplotlib.colors as mcolors

class RGBPrediction(Dataset):
    def __init__(self, rgb_files, normalization=get_image_net_normalization()):
        self.rgb_files = rgb_files
        self.normalization = normalization
        self.ids = range(len(rgb_files))

    def __len__(self):
        return len(self.ids)

    def _convert_rgb(self, rgb_file):
        return rgb_file / 255.0


    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = self._convert_rgb(self.rgb_files[id_])

        # Albumentations expects HWC format
        rgb = rgb.transpose(1, 2, 0)

        rgb = np.clip(rgb, 0.0, 1.0)

        # Assumes values are in range of [0, 1]
        if self.normalization:
            rgb = self.normalization(image=rgb)["image"]

        # Convert to torch tensors
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]

        return rgb


# Creates patches of 512x512 starting from the top left, from left to right and top to bottom
def get_patches(ortho: str, dsm: str) -> Tuple[Dataset, Tuple[int, int], Dict]:
    patch_dim = 512
    rgb_tile, dsm_tile = None, None
    with rasterio.open(ortho) as src:
        rgb_tile = src.read().astype(np.uint8)
    with rasterio.open(dsm) as src:
        meta_data = src.meta.copy()
        dsm_tile = src.read(1)
    horizontal_indices = np.ceil(dsm_tile.shape[0] / patch_dim).astype(int)
    vertical_indices = np.ceil(dsm_tile.shape[1] / patch_dim).astype(int)
    rgb_patches = []
    dsm_patches = []
    # RGB dimensions [C,H,W]
    for i_vert in range(vertical_indices):
        for i_hor in range(horizontal_indices):
            # Lower corner exemption
            if i_vert == vertical_indices - 1 and i_hor == horizontal_indices - 1:
                rgb_patches.append(rgb_tile[:, -patch_dim:, -patch_dim:])
                dsm_patches.append(dsm_tile[-patch_dim:, -patch_dim:])
            # Lower row exemption
            elif i_vert == vertical_indices - 1:
                rgb_patches.append(rgb_tile[:, -patch_dim:, patch_dim * i_hor:patch_dim * (i_hor + 1)])
                dsm_patches.append(dsm_tile[-patch_dim:, patch_dim * i_hor:patch_dim * (i_hor + 1)])
            # Last column exemption
            elif i_hor == horizontal_indices - 1:
                rgb_patches.append(rgb_tile[:, patch_dim * i_vert:patch_dim * (i_vert + 1), -patch_dim:])
                dsm_patches.append(dsm_tile[patch_dim * i_vert:patch_dim * (i_vert + 1), -patch_dim:])
            else:
                rgb_patches.append(
                    rgb_tile[:, patch_dim * i_vert:patch_dim * (i_vert + 1), patch_dim * i_hor:patch_dim * (i_hor + 1)])
                dsm_patches.append(
                    dsm_tile[patch_dim * i_vert:patch_dim * (i_vert + 1), patch_dim * i_hor:patch_dim * (i_hor + 1)])
    return RGBPrediction(rgb_patches), dsm_tile.shape, meta_data


def get_inference(dataset: Dataset, model: torch.nn.Module, device: torch.device):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    inferences = np.zeros([0, 512, 512], dtype='uint8')
    model.eval()
    with torch.inference_mode():
        for input_data in tqdm(dataloader, desc="Performing inference", leave=False):
            input_data = input_data.to(device)
            output_data = model(input_data)
            predictions = torch.argmax(output_data, dim=1)
            predictions = predictions.cpu().numpy()
            inferences = np.concatenate([inferences, predictions], axis=0)

            del input_data, output_data, predictions
            torch.cuda.empty_cache()
    return inferences


def stitch_predictions(inferences: np.ndarray, size_hor, size_vert) -> np.ndarray:
    image = np.zeros([size_hor, size_vert], dtype='uint8')  # [ROWS x COLUMNS]
    patch_dim = 512
    horizontal_indices = np.ceil(size_hor / patch_dim).astype(int)
    vertical_indices = np.ceil(size_vert / patch_dim).astype(int)
    for i_vert in range(vertical_indices):
        for i_hor in range(horizontal_indices):
            i_infer = i_vert * vertical_indices + i_hor
            # Lower corner exemption
            if i_vert == vertical_indices - 1 and i_hor == horizontal_indices - 1:
                image[-patch_dim:, -patch_dim:] = inferences[i_infer]
            # Lower row exemption
            elif i_vert == vertical_indices - 1:
                image[-patch_dim:, i_hor * patch_dim:(i_hor + 1) * patch_dim] = inferences[i_infer]
            # Last column exemption
            elif i_hor == horizontal_indices - 1:
                image[i_vert * patch_dim:(i_vert + 1) * patch_dim, -patch_dim:] = inferences[i_infer]
            else:
                image[i_vert * patch_dim:(i_vert + 1) * patch_dim, i_hor * patch_dim:(i_hor + 1) * patch_dim] = inferences[i_infer]
    return image


def perform_prediction(input_folder: str, output_folder: str, model: torch.nn.Module, device: torch.device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    index = 0
    # --- For each orthophoto and dsm tile generate patches for the dataset --- #
    for ortho_file, dsm_file in zip(os.listdir(os.path.join(input_folder, 'ortho/')),
                                    os.listdir(os.path.join(input_folder, 'dsm/'))):
        patches_dataset, size, meta_data = get_patches(os.path.join(input_folder, 'ortho', ortho_file),
                                            os.path.join(input_folder, 'dsm', dsm_file))
        inferences = get_inference(patches_dataset, model, device)
        predicted_tile = stitch_predictions(inferences, size[0], size[1])
        output_path = os.path.join(output_folder, f'{index:03d}.tif')

        # --- Write output to a file and notify user of progress --- #
        meta_data.update(dtype="uint8", count=1)
        with rasterio.open(output_path, mode='w', compress='lzw', **meta_data) as f:
            f.write(predicted_tile, 1)
        index += 1
        print(f'File exported to {output_path}')


def plot_prediction(inference: np.ndarray):
    class_colors = np.array(["white", "black", "dimgray", "darkgray", "darkmagenta", "darkred", "orange", "forestgreen",
                    "lightgreen", "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"])
    class_colors = np.array([mcolors.to_rgb(c) for c in class_colors])
    pred_rgb = class_colors[inference]
    plt.imshow(pred_rgb)
    plt.show()


if __name__ == "__main__":
    # --- input data streams --- #
    datafolder = "C:/MTP-Data/gpp_dataset/soesterberg"
    output_folder = f'{datafolder}/predictions'

    # --- Model setup --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights = 'C:/MTP-Data/trained_models/rgb_u_net_pr34_b16/rgb_u_net_pr34_b16.pth'
    model = RGBUNet(encoder_name='resnet34')
    model.load_state_dict(torch.load(model_weights, weights_only=True, map_location=device))
    model.to(device)

    # --- Perform predictions --- #
    perform_prediction(datafolder, output_folder, model, device)
