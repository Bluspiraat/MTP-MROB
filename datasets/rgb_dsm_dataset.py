import torch
from torch.utils.data import Dataset
import os
import numpy as np
import rasterio

class RGBDSMDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, mask_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.mask_dir = mask_dir
        self.transform = transform
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
            return dsm / 255.0

    def _convert_mask(self, file_location):
        with rasterio.open(file_location) as src:
            mask = src.read(1)
            return mask

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = self._convert_rgb(f'{self.rgb_dir}/{id_}.tif')
        dsm = self._convert_dsm(f'{self.rgb_dir}/{id_}.tif')
        mask = self._convert_mask(f'{self.mask_dir}/{id_}.tif')

        rgb = torch.from_numpy(rgb)
        dsm = torch.from_numpy(dsm)
        mask = torch.from_numpy(mask)

        if self.transform:
            rgb, dsm, mask = self.transform(rgb, dsm, mask)

        return rgb, dsm, mask

class RGBDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ids = [f.split('.')[0] for f in os.listdir(rgb_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.ids)

    #ToDo: Normalize for ResNet network
    def _convert_rgb(self, file_location):
        with rasterio.open(file_location) as src:
            rgb = src.read()
            return rgb.astype(np.float32) / 255.0

    def _convert_mask(self, file_location):
        with rasterio.open(file_location) as src:
            mask = src.read(1)
            return mask.astype(np.int64)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = self._convert_rgb(f'{self.rgb_dir}/{id_}.tif')
        mask = self._convert_mask(f'{self.mask_dir}/{id_}.tif')

        rgb = torch.from_numpy(rgb)
        mask = torch.from_numpy(mask)

        if self.transform:
            rgb, mask = self.transform(rgb, mask)

        return rgb, mask