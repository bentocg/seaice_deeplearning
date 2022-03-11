import os
import shutil
import time
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import segmentation_models_pytorch as smp
import torch
import ttach as tta
from rasterio.features import shapes
from torch.utils.data import DataLoader

from utils.data_processing import (
    TestDataset,
    Tiff,
    merge_output,
    tile_image,
    write_output,
)
from utils.data_processing.polygonize_raster import polygonize_raster
from utils.training.utility import seed_all


def parse_args():
    parser = ArgumentParser("Inputs for temple classification pipeline")
    parser.add_argument(
        "--model_name", "-n", type=str, help="model name for checkpoing loading",
        default="UnetResnet34_512_8.197217315846395e-05_60_scratch_tsets_hand_aug_simple_ratio_0.75_loss_Mixed_dice-0.8704461455345154_iou-0.6997870008453754_epoch-20.pth"
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=-1,
        help="batch size for prediction, can be inferred by model name or set manually",
    )
    parser.add_argument(
        "--device_id",
        "-d",
        type=int,
        default=-1,
        help="uses all devices when set to -1 and forces cpu run if set to -9999",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=1,
        help="number of workers for dataloader",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        help="path to checkpoints folder",
    )
    parser.add_argument("--input_raster", "-i", type=str, help="path to input raster")
    parser.add_argument(
        "--stride", "-s", type=float, default=0.33, help="stride for prediction"
    )
    parser.add_argument(
        "--tta", "-t", type=int, default=1, help="toggle for test-time augmentation"
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="processed_imagery",
        help="output folder for predicted shapefiles",
    )
    parser.add_argument(
        "--threshold",
        "-x",
        type=float,
        default=0.5,
        help="threshold for output binarization",
    )
    parser.add_argument(
        "--polygons",
        "-z",
        type=int,
        default=0,
        help="write polygons to shapefile?"
    )

    parser.add_argument(
        "--thumbnail_outputs",
        "-u",
        type=int,
        default=1,
        help="create thumbnail outputs?"
    )

    return parser.parse_args()


def main():
    tic = time.time()
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)

    # load model
    model_name = args.model_name
    model = smp.Unet(in_channels=1)

    # extract model configs
    patch_size = int(model_name.split("_")[1])
    batch_size = args.batch_size or int(model_name.split("_")[3]) // 4

    # move to GPU if available
    if torch.cuda.is_available():
        if args.device_id == -1:

            device = "cuda:0"
            model = model.to(device)

        elif args.device_id == -9999:
            device = "cpu"
            model.to(device)
        else:
            device = f"cuda:{args.device_id}"
            model = model.to(device)
    else:
        device = "cpu"
    cp_path = f"checkpoints/{model_name}"
    if args.checkpoint_path:
        cp_path = f"{args.checkpoint_path}/{cp_path}"
    state_dict = torch.load(cp_path, map_location=device)
    model.load_state_dict(state_dict["state_dict"])

    # scan input and mask folder
    scene = os.path.basename(args.input_raster)
    out_dir = f"{args.output_folder}/{scene}/preds"
    os.makedirs(out_dir.replace("preds", "tiles"), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # extract RGB tiles from raster
    tic = time.time()
    img, width, height, meta = Tiff().process_raster(args.input_raster)
    tile_image(img, patch_size, args.stride, out_dir.replace("preds", "tiles"), scene)
    print(f"Finished tiling raster of size ({height}, {width}) in {time.time() - tic}.")

    # instantiate dataset and dataloder
    tic = time.time()
    dataset = TestDataset(out_dir.replace("preds", "tiles"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )

    # define transforms for tta
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

    # write predictions
    with torch.no_grad():
        for tiles, img_names in dataloader:
            tiles = tiles.to(device)
            if args.tta:
                mask = torch.zeros(tiles.size())
                mask.to(device)
                for transformer in transforms:
                    preds = model(transformer.augment_image(tiles))
                    preds = transformer.deaugment_mask(preds)
                    mask += preds
                preds = mask / len(transforms)

            else:
                preds = model(tiles)
            preds = torch.sigmoid(preds)
            preds = (preds > args.threshold).detach().float() * 255
            write_output(preds, img_names, out_dir)

    print(f"Finished writing CNN predictions in {time.time() - tic}")
    shutil.rmtree(f"{args.output_folder}/{scene}")
    return None

    # free up memory
    del dataloader
    del dataset
    del model

    # merge predictions
    tic = time.time()
    final_output = merge_output((height, width), out_dir)
    final_output = (final_output > args.threshold).astype(np.uint8)
    final_output = final_output * 255
    shutil.rmtree(f"{args.output_folder}/{scene}")

    # write alpha layers
    if args.thumbnail_outputs:
        alpha_layer = np.zeros(img.shape, dtype=np.uint8)
        alpha_layer[final_output > 0, :] = (45, 45, 255)
        blend = cv2.addWeighted(img, 0.65, alpha_layer, 0.3, 0)
        img[final_output > 0, :] = blend[final_output > 0, :]
        img = img[::8, ::8]
        cv2.imwrite(f"{args.output_folder}/{scene}", img)
        print(f"Finished mosaicing output in {time.time() - tic}")

    # create shapefiles
    tic = time.time()

    if args.polygons:
        gdf = polygonize_raster(args.input_raster, final_output)

        os.makedirs(f"{args.output_folder}/shapefiles", exist_ok=True)
        if len(gdf) > 0:
            gdf["scene"] = scene
            gdf.to_file(f"{args.output_folder}/shapefiles/{scene.split('.')[0]}.shp")
            print(time.time() - tic)
        else:
            print(f"No sea ice polygons for {args.input_scene}")


if __name__ == "__main__":
    main()
