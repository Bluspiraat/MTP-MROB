import torch
from torch.utils.data import Dataset
import os
import numpy as np
import rasterio

class RGBDSMDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, mask_dir, geo_transform=None, rgb_transform=None, normalization=None):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.mask_dir = mask_dir
        self.geo_transform = geo_transform
        self.rgb_transform = rgb_transform
        self.normalization = normalization
        self.vmin = -5
        self.vmax = 100
        self.ids = [f.split('.')[0] for f in os.listdir(rgb_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.ids)

    def _convert_rgb(self, file_location):
        with rasterio.open(file_location) as src:
            rgb = src.read()
            return rgb / 255.0

    def _convert_dsm(self, file_location):
        with rasterio.open(file_location) as src:
            dsm = src.read(1)
            np.nan_to_num(dsm, copy=False, nan=self.vmin)
            dsm.clip(self.vmin, self.vmax)
            dsm = (dsm-self.vmin)/(self.vmax-self.vmin)
            return dsm

    def _convert_mask(self, file_location):
        with rasterio.open(file_location) as src:
            mask = src.read(1)
            return mask

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = self._convert_rgb(f'{self.rgb_dir}/{id_}.tif')
        dsm = self._convert_dsm(f'{self.dsm_dir}/{id_}.tif')
        mask = self._convert_mask(f'{self.mask_dir}/{id_}.tif')

        # Albumentations expects HWC format
        rgb = rgb.transpose(1, 2, 0)

        # Geometric changes
        if self.geo_transform:
            augmented = self.geo_transform(image=rgb, dsm=dsm, mask=mask)
            rgb, dsm, mask = augmented["image"], augmented["dsm"], augmented["mask"]

        rgb = np.clip(rgb, 0.0, 1.0)

        # Color distortions
        if self.rgb_transform:
            rgb = self.rgb_transform(image=rgb)["image"]

        # Assumes values are in range of [0, 1]
        if self.normalization:
            rgb = self.normalization(image=rgb)["image"]

        # Convert to torch tensors
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        dsm = torch.tensor(dsm, dtype=torch.float32)  # [H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # [H, W]

        return torch.cat((rgb, dsm.unsqueeze(0)), 0), mask

class RGBDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, geo_transform=None, rgb_transform=None, normalization=None):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.geo_transform = geo_transform
        self.rgb_transform = rgb_transform
        self.normalization = normalization
        self.ids = [f.split('.')[0] for f in os.listdir(rgb_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.ids)

    def _convert_rgb(self, file_location):
        with rasterio.open(file_location) as src:
            rgb = src.read().astype(np.float32)/255.0
            return rgb

    def _convert_mask(self, file_location):
        with rasterio.open(file_location) as src:
            mask = src.read(1).astype(np.int64)
            return mask

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = self._convert_rgb(f'{self.rgb_dir}/{id_}.tif')  # [3,H,W]
        mask = self._convert_mask(f'{self.mask_dir}/{id_}.tif')  # [H,W]

        # Albumentations expects HWC format
        rgb = rgb.transpose(1, 2, 0)

        # Geometric changes
        if self.geo_transform:
            augmented = self.geo_transform(image=rgb, mask=mask)
            rgb, mask = augmented["image"], augmented["mask"]

        rgb = np.clip(rgb, 0.0, 1.0)

        # Color distortions
        if self.rgb_transform:
            rgb = self.rgb_transform(image=rgb)["image"]

        # Assumes values are in range of [0, 1]
        if self.normalization:
            rgb = self.normalization(image=rgb)["image"]

        # Convert to torch tensors
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # [H, W]

        return rgb, mask
