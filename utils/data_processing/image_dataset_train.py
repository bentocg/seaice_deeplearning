from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from utils.data_processing import get_transforms


class SeaIceDataset(Dataset):
    def __init__(self, df, data_folder, size=256, phase='train', segmentation=False):

        # read dataframe with filenames and associated labels
        self.ds = df
        self.root = data_folder
        self.transforms = get_transforms(phase, size)
        self.segmentation = segmentation
        self.size = size

        # get labels and img names
        self.long_labels = self.ds.label.values
        self.bin_labels = self.ds['pack_ice'].values
        self.img_names = [f'{self.root}/x/{file}' for file in self.ds.img_name.values]

        if self.segmentation:
            self.ds = self.ds.loc[(self.ds.has_mask == 1) | (self.ds.pack_ice == 0)]
            self.mask_names = [f'{self.root}/y/{file}' for file in self.ds.img_name.values]
            self.mask_names = [self.mask_names[idx] if ele else False for idx, ele in enumerate(self.ds.has_mask)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read img and apply transforms
        img_path = self.img_names[idx]
        label = self.bin_labels[idx]
        img = np.array(Image.open(img_path))
        if self.segmentation:
            mask_path = self.mask_names[idx]
            if mask_path:
                mask = np.array(Image.open(mask_path))
            else:
                mask = np.zeros([self.size, self.size, 1])

        if self.transforms is not None:
            if self.segmentation:
                try:
                    augmented = self.transforms(image=img, mask=mask)

                    img = augmented['image']
                    mask = augmented['mask'].reshape([1, self.size, self.size])

                except RuntimeError:
                    img_path = self.img_names[idx - 5]
                    mask_path = self.mask_names[idx - 5]
                    label = self.bin_labels[idx]

                    img = np.array(Image.open(img_path))
                    if mask_path:
                        mask = np.array(Image.open(mask_path))
                    else:
                        mask = np.zeros([self.size, self.size, 1])
                    augmented = self.transforms(image=img, mask=mask)

                    img = augmented['image']
                    mask = augmented['mask'].reshape([1, self.size, self.size])
            else:
                try:
                    img = self.transforms(image=img)['image']
                    label = self.bin_labels[idx]
                except RuntimeError:
                    img_path = self.img_names[idx - 20]
                    img = np.array(Image.open(img_path))
                    img = self.transforms(image=img)['image']
                    label = self.bin_labels[idx - 20]
        if self.segmentation:
            return img, label, mask

        else:
            return img, label
