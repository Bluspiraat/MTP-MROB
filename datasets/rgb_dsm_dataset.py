import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class RGBDSMDataset(Dataset):
    def __init__(self, rgb_dir, dsm_dir, mask_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ids = [f.split('.')[0] for f in os.listdir(rgb_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        rgb = np.array(Image.open(os.path.join(self.rgb_dir, id_ + '.png')).convert('RGB')).astype(np.float32)/255.0
        dsm = np.array(Image.open(os.path.join(self.dsm_dir, id_ + '.png'))).astype(np.float32)/255.0
        mask = np.array(Image.open(os.path.join(self.mask_dir, id_ + '.png'))).astype(np.int64)

        rgb = torch.from_numpy(rgb).permute(2,0,1)
        dsm = torch.from_numpy(dsm).unsqueeze(0)
        mask = torch.from_numpy(mask)

        if self.transform:
            rgb, dsm, mask = self.transform(rgb, dsm, mask)

        return rgb, dsm, mask
