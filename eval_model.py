from typing_extensions import final
from utils.data_processing.tile_image import tile_image
import torch
from argparse import ArgumentParser
import segmentation_models_pytorch as smp
from utils.training.utility import seed_all
from itertools import product
from utils.data_processing import TestDataset, write_output, merge_output
from torch.utils.data import DataLoader
import ttach as tta
import os
import numpy as np
import cv2
import pandas as pd
from shutil import rmtree
from PIL import Image

def parse_args():
    parser = ArgumentParser('Inputs for temple classification pipeline')
    parser.add_argument('--model_name', '-n', type=str, help='model name for checkpoing loading')
    parser.add_argument('--random_seed', '-r', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--device_id', '-d', type=int, default=-1, help='uses all devices when set to -1')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--input_folder', '-i', type=str, help='path to input test set')
    parser.add_argument('--masks_folder', '-m', type=str, help='path to folder with groundtruth masks')
    parser.add_argument('--stride', '-s', type=float, default=1, help='stride for prediction')
    parser.add_argument('--tta', '-t', type=bool, default=0, help='toggle for test-time augmentation')
    parser.add_argument('--output_folder','-o', type=str,  default='test_output', help='output folder for test predictions')
    parser.add_argument('--threshold', '-x', type=float, default=0.5, help='threshold for output binarization')
    parser.add_argument('--save_output', '-u', type=bool, default=1, help='whether to save the output')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)
    
    # load model
    model_name = args.model_name
    model = smp.Unet()
    

    # extract model configs
    patch_size = int(model_name.split('_')[1])
    batch_size = int(model_name.split('_')[3])
    

    # move to GPU if available
    if torch.cuda.is_available():
        if args.device_id == -1:
            model = torch.nn.DataParallel(model)
            device = 'cuda:0'
        else:
            device = f'cuda:{args.device_id}'
            model = model.to(device)
    else:
        device = 'cpu'
    state_dict = torch.load(f'checkpoints/{model_name}', map_location=device)
    model.load_state_dict(state_dict['state_dict'])



    # add test-time-augmentation
    if args.tta:
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='tsharpen')
    model.eval()
            
            

    # scan input and mask folders
    os.makedirs(args.output_folder, exist_ok=True)
    masks = []
    for path, _, filenames in os.walk(args.masks_folder):
        for file in filenames:
            if file.endswith('.tif'):
                masks.append(f'{path}/{file}')
    scenes = [f'{args.input_folder}/{patch_size}/{os.path.basename(ele)}' for ele in masks]

    # store model performance
    if os.path.exists(f'{args.output_folder}/model_stats.csv'):
        model_stats = pd.read_csv(f'{args.output_folder}/model_stats.csv')
    else:
        model_stats = pd.DataFrame()

    # loop through image-mask pairs
    for idx, fname in enumerate(scenes):
        scene = os.path.basename(fname)
        mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
        heigth, width = mask.shape
        # instantiate dataset and dataloder
        dataset = TestDataset(fname)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=False,
        )
        
        # write predictions
        out_dir = f"{args.output_folder}/{scene}/pred_tiles"
        os.makedirs(out_dir, exist_ok=True)
    
        with torch.no_grad(): 
            for tiles, img_names in dataloader:
                tiles = tiles.to(device)
                preds = torch.sigmoid(model(tiles))
                preds = (preds > args.threshold).detach().float() * 255 
                write_output(preds, img_names, out_dir)
        
        # merge predictions
        final_output = merge_output((heigth, width), out_dir)
        final_output = (final_output > args.threshold).astype(np.uint8)
        final_output = final_output * 255
        cv2.imwrite(f'{args.output_folder}/{scene.split(".")[0]}_predicted.png', final_output)

        # get IoU and DICE
        final_output = (final_output > 0).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        intersection = (final_output * mask).sum()
        cardinality = (final_output + mask).sum()
        union = np.logical_or(final_output, mask).sum() 
        iou = intersection / (union + 1E-6)
        dice = 2 * intersection / (cardinality + 1E-6)

        # erase temp tiles
        if not args.save_output:
            rmtree(f"{args.output_folder}/{scene}/pred_tiles")

        model_stats = model_stats.append({'model': model_name, 
                                          'iou': iou, 
                                          'dice': dice,
                                          'scene': scene}, ignore_index=True)

    model_stats.to_csv(f'{args.output_folder}/model_stats.csv', index=False)


if __name__ == '__main__':
    main()