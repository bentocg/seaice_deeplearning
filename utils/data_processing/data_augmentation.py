__all__ = ["get_transforms"]

import albumentations as A
from albumentations.augmentations.transforms import CenterCrop, Flip, Rotate
from albumentations.pytorch import ToTensorV2
from torch.nn.modules import module


def train_transform(size, mode):
    if mode == "simple":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=size, width=size),
                A.RandomBrightnessContrast(p=0.5),
                A.Flip(p=0.5),
                A.RandomRotate90(p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif mode == "complex":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=int(size * 1.15),
                    width=int(size * 1.15),
                    scale=(0.15, 1.0),
                    ratio=(0.75, 1.33),
                ),
                A.Rotate(limit=(-15, 15)),
                A.OneOf(
                    [
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.IAAPiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=8),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomContrast(),
                        A.RandomBrightness(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(
                    p=0.4, hue_shift_limit=5, val_shift_limit=5, sat_shift_limit=5
                ),
                A.Flip(p=0.66),
                A.CenterCrop(height=size, width=size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def val_transform(size):
    return A.Compose(
        [
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def test_transform():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_transforms(phase, size=256, mode="simple"):
    if phase == "training":
        return train_transform(size, mode)
    elif phase == "test":
        return test_transform()
    else:
        return val_transform(size)
