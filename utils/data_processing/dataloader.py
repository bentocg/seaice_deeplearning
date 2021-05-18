from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, sampler
import pandas as pd
from utils.data_processing import SeaIceDataset
from .balanced_batch_sampler import BalancedBatchSampler


def provider(df_path, data_folder, phase, size, tsets, batch_size=8, num_workers=8,
             segmentation=False):
    df = pd.read_csv(df_path)
    
    image_dataset = SeaIceDataset(df=df, data_folder=data_folder, phase=phase, tsets=tsets, 
                                  size=size, segmentation=segmentation)
    if phase == 'train':
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            sampler=BalancedBatchSampler(image_dataset, image_dataset.bin_labels),
            drop_last=True
        )

    else:
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False,
        )

    return dataloader
