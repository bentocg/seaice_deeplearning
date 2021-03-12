from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import pandas as pd
from utils.data_processing import SeaIceDataset
from .balanced_batch_sampler import BalancedBatchSampler


def provider(fold, total_folds, df_path, data_folder, phase, size, batch_size=8, num_workers=4,
             segmentation=False):
    df = pd.read_csv(df_path)
    df_ice = df[df["pack_ice"] == 1]
    df_not = df[df["pack_ice"] == 0]

    df = pd.concat([df_ice, df_not.sample(len(df_ice), random_state=4)])

    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=4)
    train_idx, val_idx = list(kfold.split(df["img_name"], df["pack_ice"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df

    # NOTE: total_folds=5 -> train/val : 80%/20%
    image_dataset = SeaIceDataset(df=df, data_folder=data_folder, phase=phase, size=size,
                                  segmentation=segmentation)
    if phase == 'train':
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            sampler=BalancedBatchSampler(image_dataset, image_dataset.bin_labels)
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
