from argparse import ArgumentParser
from utils.data_processing import tile_image
from PIL import Image
import numpy as np
import os


def parse_args():
    parser = ArgumentParser('reads and tiles input images for prediction/testing')
    parser.add_argument('--input_folder', '-i', type=str, help='input folder with images to tile')
    parser.add_argument('--output_folder', '-o', type=str, default='test_set', help='destination folder for crops')
    parser.add_argument('--patch_size', '-p', type=int, default=256, help='patch size to crop, must match model input requirements')
    parser.add_argument('--stride', '-s', type=float, default=1.0, help='stride for sliding window, multiplies patch_size')
    return parser.parse_args()


def main():
    args = parse_args()
    for root, _, images in os.walk(args.input_folder):
        fnames = [ele for ele in images if ele.endswith('.tif')]
        for fname in fnames:
            scene = os.path.basename(fname)
            img = np.array(Image.open(f'{root}/{fname}'))
            out_path = f'{args.output_folder}/{args.patch_size}/{scene}'
            os.makedirs(out_path, exist_ok=True)
            tile_image(img, args.patch_size, args.stride, out_path, scene)



if __name__ == '__main__':
    main()