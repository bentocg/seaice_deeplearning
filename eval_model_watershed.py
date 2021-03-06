from utils.watershed.watershed import extract_watershed_mask
from argparse import ArgumentParser
from utils.training.utility import seed_all
from utils.data_processing import write_output, merge_output
import os
import numpy as np
import cv2
import pandas as pd
import shutil


def parse_args():
    parser = ArgumentParser("Inputs for temple classification pipeline")
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
        "--tta", "-t", type=bool, default=0, help="toggle for test-time augmentation"
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
    parser.add_argument(
        "--patch_size", "-p", type=int, default=512, help="patch size for tiles."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # set random seet
    seed_all(args.random_seed)

    # load model
    model_name = "Watershed"

    # extract model configs
    patch_size = args.patch_size
    assert patch_size in [256, 384, 512], 'unsupported patch size!'

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
    if os.path.exists(f"{args.output_folder}/global_stats.csv"):
        global_stats = pd.read_csv(f"{args.output_folder}/global_stats.csv")
    else:
        global_stats = pd.DataFrame()

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
        dataloader = (
            [cv2.imread(img_name, 0), os.path.basename(img_name)]
            for img_name in [f"{fname}/{ele}" for ele in os.listdir(fname)]
        )

        # write predictions
        out_dir = f"{args.output_folder}/{scene}/{model_name[:-4]}/pred_tiles"
        os.makedirs(out_dir, exist_ok=True)

        for ele in dataloader:
            tile, img_name = ele
            pred = extract_watershed_mask(tile)
            write_output([pred], [img_name], out_dir)

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

        cv2.imwrite(
            f'{args.output_folder}/scene_masks/{model_name[:-4]}/{scene.split(".")[0]}_tta{args.tta}_predicted.png',
            raw,
        )

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
    #
    model_stats.to_csv(
        f"{args.output_folder}/scene_masks/{model_name[:-4]}/scene_stats.csv"
    )

    global_precision = global_tp / max(1, global_tp + global_fp)
    global_recall = global_tp / max(1, global_tp + global_fn)
    avg_acc = (global_precision + global_tn / max(1, global_tn + global_fn)) / 2
    global_f1 = 2 * (
        global_precision * global_recall / (global_precision + global_recall)
    )
    global_stats = global_stats.append(
        {
            "model_name": model_name,
            "global_precision": global_precision,
            "global_recall": global_recall,
            "avg_acc": avg_acc,
            "global_f1": global_f1,
            "tta": args.tta == 1,
            "mean_iou": model_stats.iou.mean(),
            "mean_dice": model_stats.dice.mean(),
        },
        ignore_index=True,
    )
    global_stats.to_csv(f"{args.output_folder}/global_stats.csv", index=False)


if __name__ == "__main__":
    main()
