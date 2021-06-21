# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['extract_watershed_mask']

import cv2
import numpy as np
import os
from PIL import Image


def watershed(img, kernel_size):
    if type(img) == str:
        img = cv2.imread(img, 0)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # blur image and apply Otsu`s binarization
    img_blur = cv2.medianBlur(img, 5)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # denoise output with openings
    kernel = np.ones([kernel_size, kernel_size])
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # find centroids using distance transforms
    dist_transform = cv2.distanceTransform(img_thresh, cv2.DIST_L2, 5)
    _, centroid_masks = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    centroid_masks = centroid_masks.astype(np.uint8)

    # apply watershed segmentation
    background = cv2.dilate(img_thresh, kernel, iterations=1)
    unknown = cv2.subtract(background, centroid_masks)
    _, markers, stats, centroids = cv2.connectedComponentsWithStats(centroid_masks, connectivity=8)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_fill = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_out = img_fill.copy()
    markers = cv2.watershed(img_fill, markers)

    # extract filled shapes
    img_fill[markers != 1] = [255, 255, 255]
    img_fill = cv2.cvtColor(img_fill, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    img_fill[0, :] = 0
    img_fill[img_fill.shape[0] - 1, :] = 0
    img_fill[:, 0] = 0
    img_fill[:, img_fill.shape[1] - 1] = 0
    img_fill[img_fill < 255] = 0

    # extract outlines
    img_out[markers == -1] = [255, 255, 255]
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    img_out[0, :] = 0
    img_out[img_out.shape[0] - 1, :] = 0
    img_out[:, 0] = 0
    img_out[:, img_out.shape[1] - 1] = 0
    img_out = cv2.dilate(img_out, np.ones([3, 3]))
    img_out[img_out < 255] = 0

    return img_fill, img_out


def extract_sea_ice(img, kernel_size=9, n_iter=3):
    """
    Helper function to extract sea ice masks from panchromatic imagery
    Parameters
    ----------
    img: np.array or string with the image
    kernel_size: kernel size for morphological transforms
    add_centroids: boolean for whether to add centroids to outline

    Returns
    -------

    """
    fill = np.zeros(img.shape)
    outline = np.zeros(img.shape)
    background_img = np.random.choice(os.listdir("/home/bento/GIS/sea-ice-deeplearning/background_imgs"))
    background_img = np.array(
        Image.open(f'/home/bento/GIS/sea-ice-deeplearning/background_imgs/{background_img}').convert('L'))
    for _ in range(n_iter):
        curr_fill, curr_outline = watershed(img, kernel_size)
        fill = curr_fill + fill
        outline = curr_outline + outline
        img[fill == 255] = 0
        #img = img * np.array(fill == 0) + background_img * np.array(fill == 255)

    return fill


def extract_watershed_mask(img, kernel_size=9, get_outline=False, add_centroids=False):
    """
    Helper function to extract sea ice masks from panchromatic imagery
    Parameters
    ----------
    img: np.array or string with the image
    kernel_size: kernel size for morphological transforms
    add_centroids: boolean for whether to add centroids to outline
    Returns
    -------
    """
    if type(img) == str:
        img = cv2.imread(img, 0)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # blur image and apply Otsu`s binarization
    img_blur = cv2.medianBlur(img, 5)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # denoise output with openings
    kernel = np.ones([kernel_size, kernel_size])
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # find centroids using distance transforms
    dist_transform = cv2.distanceTransform(img_thresh, cv2.DIST_L2, 5)
    _, centroid_masks = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    centroid_masks = centroid_masks.astype(np.uint8)

    # apply watershed segmentation
    background = cv2.dilate(img_thresh, kernel, iterations=1)
    unknown = cv2.subtract(background, centroid_masks)
    _, markers, stats, centroids = cv2.connectedComponentsWithStats(centroid_masks, connectivity=8)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_fill = img_col.copy()
    markers = cv2.watershed(img_col, markers)

    if get_outline:
        # draw centroids
        if add_centroids:
            for cent in centroids[1:]:
                cent = [int(ele) for ele in cent]
                img_col[cent[1], cent[0]] = [255, 255, 255]
                img_col = cv2.dilate(img_col, np.ones([5, 5]))

        # draw edge and apply gaussian kernel
        img_col[markers == -1] = [255, 255, 255]
        img_col[0, :] = 0
        img_col[img_col.shape[0] - 1, :] = 0
        img_col[:, 0] = 0
        img_col[:, img_col.shape[1] - 1] = 0
        img_col = cv2.dilate(img_col, np.ones([3, 3]))
        outline = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        outline = cv2.GaussianBlur(outline, (5, 5), 0)

    # extract filled shapes
    img_fill[markers != 1] = [255, 255, 255]
    img_fill = cv2.cvtColor(img_fill, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    img_fill[0, :] = 0
    img_fill[img_fill.shape[0] - 1, :] = 0
    img_fill[:, 0] = 0
    img_fill[:, img_fill.shape[1] - 1] = 0
    img_fill[img_fill < 255] = 0

    if get_outline:
        return {'mask': img_fill, 'outline': outline}
    else:
        return img_fill
