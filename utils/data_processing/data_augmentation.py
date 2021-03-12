__all__ = ['get_transforms']

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transform(size):
    return A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=size, width=size),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Flip(p=0.5),
            # A.ChannelShuffle(p=0.2),
            A.RandomRotate90(p=1),
            # A.Equalize(p=0.2),
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
    if phase == 'train':
        return train_transform(size)
    else:
        return val_transform(size)
