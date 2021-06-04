__all__ = ['write_output']

import cv2
import numpy as np
import os


def write_output(out, img_names, out_dir):
    out = out.cpu().numpy().astype(np.uint8) * 255
    for idx, img in enumerate(out):
        cv2.imwrite(f'{out_dir}/{os.path.basename(img_names[idx])}', img.transpose(1, 2, 0))

