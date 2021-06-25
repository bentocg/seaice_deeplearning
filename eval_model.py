import shutil
import torch
from argparse import ArgumentParser
import segmentation_models_pytorch as smp
from utils.training.utility import seed_all
from utils.data_processing import TestDataset, write_output, merge_output
from torch.utils.data import DataLoader
import ttach as tta
import os
import numpy as np
import cv2
import pandas as pd


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
        help="uses all devices when set to -1",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument("--input_folder", "-i", type=str, help="path to input test set")
    parser.add_argument(
        "--masks_folder", "-m", type=str, help="path to folder with groundtruth masks"
    )
    parser.add_argument("--raw_image_folder", "-f", type=str, help="path to raw images")
    parser.add_argument(
        "--stride", "-s", type=float, default=1, help="stride for prediction"
    )
    parser.add_argument(
        "--tta", "-t", type=int, default=0, help="toggle for test-time augmentation"
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="test_output",
        help="output folder for test predictions",
    )
    parser.add_argument(
        "--threshold",
        "-x",
        type=float,
        default=0.5,
        help="threshold for output binarization",
    )
    parser.add_argument(
        "--save_output", "-u", type=bool, default=0, help="whether to save the output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)

    # load model
    model_name = args.model_name
    model = smp.Unet()

    # extract model configs
    patch_size = int(model_name.split("_")[1])
    batch_size = int(model_name.split("_")[3])

    # move to GPU if available
    if torch.cuda.is_available():
        if args.device_id == -1:

            device = "cuda:0"
            model = model.to(device)
        else:
            device = f"cuda:{args.device_id}"
            model = model.to(device)
    else:
        device = "cpu"
    state_dict = torch.load(f"checkpoints/{model_name}", map_location=device)
    model.load_state_dict(state_dict["state_dict"])

    if args.device_id == -1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        batch_size *= 24

    # add test-time-augmentation
    if args.tta == 1:
        model = tta.SegmentationTTAWrapper(
            model, tta.aliases.d4_transform(), merge_mode="tsharpen"
        )
    model.eval()

    # scan input and mask folders
    os.makedirs(f"{args.output_folder}/scene_masks/{model_name[:-4]}", exist_ok=True)
    masks = []
    for path, _, filenames in os.walk(args.masks_folder):
        for file in filenames:
            if file.endswith(".tif"):
                masks.append(f"{path}/{file}")
    scenes = [
        f"{args.input_folder}/{patch_size}/{os.path.basename(ele)}" for ele in masks
    ]
    raw_images = [f"{args.raw_image_folder}/{os.path.basename(ele)}" for ele in masks]

    # store model performance
    if not os.path.exists(f"{args.output_folder}/global_stats.csv"):
        with open(f'{args.output_folder}/global_stats.csv', 'w') as file:
            file.write("avg_acc,global_f1,global_precision,global_recall,mean_dice,mean_iou,model_name,tta\n")
    model_stats = pd.DataFrame()
    global_fp = 0
    global_tp = 0
    global_tn = 0
    global_fn = 0

    # loop through image-mask pairs
    for idx, fname in enumerate(scenes):
        scene = os.path.basename(fname)
        mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(raw_images[idx]).astype(np.uint8)
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
        out_dir = f"{args.output_folder}/{scene}/{model_name[:-4]}/pred_tiles"
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

        # get alpha masks
        tp = final_output * (mask // 255)
        fp = final_output * (mask == 0).astype(np.uint8) 
        fn = mask - tp

        # write alpha layers
        alpha_layer_tp = np.zeros(raw.shape, dtype=np.uint8)
        alpha_layer_tp[tp > 0, :] = (45, 255, 45)
        blend_fp = cv2.addWeighted(raw, 0.65, alpha_layer_tp, 0.3, 0)
        raw[tp > 0, :] = blend_fp[tp > 0, :]
        
        alpha_layer_fp = np.zeros(raw.shape, dtype=np.uint8)
        alpha_layer_fp[fp > 0, :] = (255, 45, 45)
        blend_fp = cv2.addWeighted(raw, 0.65, alpha_layer_fp, 0.3, 0)
        raw[fp > 0, :] = blend_fp[fp > 0, :]
        
        alpha_layer_fn = np.zeros(raw.shape, dtype=np.uint8)
        alpha_layer_fn[fn > 0, :] = (45, 45, 255)
        blend_fn = cv2.addWeighted(raw, 0.65, alpha_layer_fn, 0.3, 0)
        raw[fn > 0, :] = blend_fn[fn > 0, :]

        # get IoU and DICE
        final_output = (final_output > 0).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        intersection = (final_output * mask).sum()
        cardinality = (final_output + mask).sum()
        union = np.logical_or(final_output, mask).sum()
        iou = (intersection + 1) / (union + 1)
        dice = (2 * intersection + 1) / (cardinality + 1)

        # store global metrics
        global_tp += intersection
        global_fp += final_output.sum() - intersection
        global_fn += mask.sum() - intersection
        global_tn += (
            (final_output == 0).astype(np.uint8) * (mask == 0).astype(np.uint8)
        ).sum()

        model_stats = model_stats.append(
            {
                "model": model_name,
                "iou": iou,
                "dice": dice,
                "scene": scene,
                "tta": args.tta == 1,
            },
            ignore_index=True,
        )
        shutil.rmtree("/".join(out_dir.split("/")[:-1]))

    # log results
    model_stats.to_csv(
        f"{args.output_folder}/scene_masks/{model_name[:-4]}/scene_stats.csv"
    )

    global_precision = global_tp / max(1, global_tp + global_fp)
    global_recall = global_tp / max(1, global_tp + global_fn)
    avg_acc = (global_precision + global_tn / max(1, global_tn + global_fn)) / 2
    global_f1 = 2 * (
        global_precision * global_recall / (global_precision + global_recall)
    )
    with open(f'{args.output_folder}/global_stats.csv', 'a') as file:
        file.write(f"{avg_acc},{global_f1},{global_precision},{global_recall},{model_stats.dice.mean()},{model_stats.iou.mean()},{model_name},{int(args.tta == 1)}\n")


if __name__ == "__main__":
    main()
