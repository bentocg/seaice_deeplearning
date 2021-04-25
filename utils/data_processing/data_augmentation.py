__all__ = ['get_transforms']

import albumentations as A
from albumentations.augmentations.transforms import CenterCrop, Flip, Rotate
from albumentations.pytorch import ToTensorV2


def train_transform(size):
    return A.Compose(
        [
        A.RandomResizedCrop(height=int(size * 1.15), width=int(size * 1.15), 
                            scale=(0.15, 1.0), ratio=(0.75, 1.33)),
        A.Rotate(limit=(-15, 15)),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),      
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=8),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomContrast(),
            A.RandomBrightness(),
        ], p=0.3),
        A.HueSaturationValue(p=0.4, hue_shift_limit=10, val_shift_limit=10, sat_shift_limit=15),
        A.Flip(p=0.66),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()

        ]
    )


def val_transform(size):
    return A.Compose(
        [
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )


def get_transforms(phase, size):
    if phase == 'training':
        return train_transform(size)
    else:
        return val_transform(size)
