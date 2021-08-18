import os
import shutil
import time
from argparse import ArgumentParser

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
from utils.training.utility import seed_all


def parse_args():
    parser = ArgumentParser("Inputs for temple classification pipeline")
    parser.add_argument(
        "--model_name", "-n", type=str, help="model name for checkpoing loading"
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=42,
        help="random seed for reproducibility",
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
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument("--input_raster", "-i", type=str, help="path to input raster")
    parser.add_argument(
        "--stride", "-s", type=float, default=1, help="stride for prediction"
    )
    parser.add_argument(
        "--tta", "-t", type=int, default=1, help="toggle for test-time augmentation"
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="test_output",
        help="output folder for predicted shapefiles",
    )
    parser.add_argument(
        "--threshold",
        "-x",
        type=float,
        default=0.5,
        help="threshold for output binarization",
    )

    return parser.parse_args()


def main():
    tic = time.time()
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)

    # load model
    model_name = args.model_name
    model = smp.Unet()

    # extract model configs
    patch_size = int(model_name.split("_")[1])
    batch_size = 16

    # move to GPU if available
    if torch.cuda.is_available():
        if args.device_id == -1:

            device = "cuda:0"
            model = model.to(device)

        elif args.device_id == -9999:
            device = 'cpu'
            model.to(device)
        else:
            device = f"cuda:{args.device_id}"
            model = model.to(device)
    else:
        device = "cpu"
    state_dict = torch.load(f"checkpoints/{model_name}", map_location=device)
    model.load_state_dict(state_dict["state_dict"])

    # add test-time-augmentation
    if args.tta == 1:
        model = tta.SegmentationTTAWrapper(
            model, tta.aliases.d4_transform(), merge_mode="tsharpen"
        )
    model.eval()

    # scan input and mask folder
    scene = os.path.basename(args.input_raster)
    out_dir = f"{args.output_folder}/{scene}/preds"
    os.makedirs(out_dir.replace('preds', 'tiles'), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # extract RGB tiles from raster
    img, width, height, meta = Tiff().process_raster(args.input_raster)
    tile_image(img, patch_size, args.stride, out_dir, scene)

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
    with torch.no_grad():
        for tiles, img_names in dataloader:
            tiles = tiles.to(device)
            preds = torch.sigmoid(model(tiles))
            preds = (preds > args.threshold).detach().float() * 255
            write_output(preds, img_names, out_dir)

    # merge predictions
    final_output = merge_output((height, width), out_dir)
    final_output = (final_output > args.threshold).astype(np.uint8)

    # create shapefiles
    with rasterio.open(args.input_raster) as src:
        image = src.read(1)  # first band
        crs = src.crs
        results = (
            {"properties": {"raster_val": v}, "geometry": s}
            for i, (s, v) in enumerate(
                shapes(image, mask=final_output, transform=src.transform)
            )
        )

    os.makedirs(f"{args.output_folder}/shapefiles", exist_ok=True)
    gdf = gpd.GeoDataFrame(crs=crs, geometry=list(results))
    gdf["scene"] = scene
    gdf.to_file(f"{args.output_folder}/shapefiles/{scene.split('.')[0]}.shp")
    print(time.time() - tic)


if __name__ == "__main__":
    main()
