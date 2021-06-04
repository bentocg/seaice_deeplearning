import torch
from argparse import ArgumentParser
import segmentation_models_pytorch as smp
from utils.training.utility import seed_all
from itertools import product
from utils.data_processing import TestDataset, write_output, merge_output
from torch.utils.data import DataLoader
import ttach as tta
import os
import re
import cv2
import pandas as pd


def parse_args():
    parser = ArgumentParser('Inputs for temple classification pipeline')
    parser.add_argument('--model_name', '-n', type=str, help='model name for checkpoing loading')
    parser.add_argument('--random_seed', '-r', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--device_id', '-d', type=int, default=-1, help='uses all devices when set to -1')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--input_folder', '-i', type=str, help='path to input rasters')
    parser.add_argument('--masks_folder', '-m', type=str, help='path to folder with groundtruth masks')
    parser.add_argument('--stride', '-s', type=float, default=1, help='stride for prediction')
    parser.add_argument('--tta', '-t', type=bool, default=0, help='toggle for test-time augmentation')
    parser.add_argument('--output_folder','-o', type=str,  default='test_output', help='output folder for test predictions')
    parser.add_argument('--threshold', '-x', type=float, default=0.5, help='threshold for output binarization')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)
    
    # load model
    model_name = args.model_name
    model = smp.Unet()
    state_dict = torch.load(f'checkpoints/{model_name}')
    model.load_state_dict(state_dict['state_dict'])

    # extract model configs
    patch_size = int(model_name.split('_')[1])
    batch_size = int(model_name.split('_')[3])
    
    # add test-time-augmentation
    if args.tta:
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='tsharpen')

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
    model.eval()
            
            

    # scan input and mask folders
    os.makedirs(args.output_folder, exist_ok=True)
    imgs = []
    masks = []
    for path, _, filenames in os.walk(args.masks_folder):
        for file in filenames:
            if file.endswith('.tif'):
                masks.append(f'{path}/{file}')
    imgs = [f'{args.input_folder}/{os.path.basename(ele)}' for ele in masks]

    # store model performance
    model_stats = pd.DataFrame()

    # loop through image-mask pairs
    for idx, ele in enumerate(imgs):
        img = cv2.imread(ele)
        mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
        out_dir = f"{args.output_folder}/{os.path.basename(ele)}/temp_tiles"
        os.makedirs(out_dir, exist_ok=True)
        width, heigth, channels = img.shape

        # write tiles
        for corner in product(range(0, heigth - patch_size, int(patch_size * args.stride)), 
                              range(0, width - patch_size, int(patch_size * args.stride))):
            left, down = corner
            right, top = left + patch_size, down + patch_size
            crop_img = img[left: right, down: top, :]
            if crop_img.shape == (patch_size, patch_size, channels):
                filename = f'{out_dir}/{os.path.basename(ele).replace(".tif", "")}_{left}_{down}_{right}_{top}.tif'
                cv2.imwrite(filename, crop_img)
        
        # instantiate dataset and dataloder
        dataset = TestDataset(out_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=False,
        )
        
        # write predictions
        out_dir = f"{args.output_folder}/{os.path.basename(ele)}/pred_tiles"
        os.makedirs(out_dir, exist_ok=True)
        
        with torch.no_grad(): 
            for tiles, img_names in dataloader:
                tiles = tiles.to(device)
                preds = torch.sigmoid(model(tiles))
                preds = (preds.detach() > args.threshold).float()
                write_output(preds, img_names, out_dir)

        # merge predictions

        # get IoU and DICE



if __name__ == '__main__':
    main()