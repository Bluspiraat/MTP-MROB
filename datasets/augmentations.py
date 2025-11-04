import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_geometric_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            fill=-1  # template fill value for all inputs
        )
    ],
    additional_targets={'dsm': 'image'})

def get_rgb_transform():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.3),
    ])


def get_dsm_transform():
    return A.Compose([
        A.Normalize(mean=0.5, std=0.5),
    ])


def get_image_net_normalization():
    return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0)  # Values of ImageNet
