"""
    Provides various image transformation functions
"""

import numpy as np
import cv2 as cv
import scipy.misc
import math

def check_coords(img, kp_2D):
    """Checks if the supplied 2D keypoints are contained fully in the image

    Paramters
    ---------
    img: np.array
        The image containing the hand. The dimensions of img should be HxWxC.
    kp_2D: np.array
        The 2D keypoints of the hand
    """
    x_min, x_max = math.ceil(kp_2D[:,0].min()), math.floor(kp_2D[:,0].max())
    y_min, y_max = math.ceil(kp_2D[:,1].min()), math.floor(kp_2D[:,1].max())
    if (x_min >= 0 and x_max <= img.shape[1] and y_min >=0 and
        y_max <= img.shape[0]):
        return True
    else:
        return False

def crop_hand(img, kp_2D):
    """Crops the hand based on the 2D keypoints.

    The crop is created by placing a square bounding box around the hand based
    on the 2D keypoints provided. The bounding box is not tight, instead
    theres a buffer of 8 pixels to the outer most keypoint and its guaranteed
    to be contained in the image.

    Parameters
    ----------
    img : np.array
        An numpy array containing the image from which the hand will be
        cropped. The dimensions of img should be HxWxC.
    kp_2D : np.array
        The 2D coordinates of the hand joints wrt. the image in img

    Returns
    -------
    img_crop : np.array
        The cropped image of a hand
    kp_2D_crop: np.array
        The shifted 2D joint coordinates wrt. img_crop.
    """

    # Find the outer most keypoints to define the tight bounding box
    x_min, x_max = math.ceil(kp_2D[:,0].min()), math.floor(kp_2D[:,0].max())
    y_min, y_max = math.ceil(kp_2D[:,1].min()), math.floor(kp_2D[:,1].max())
    # Assert that the joints are all located within the image
    assert(check_coords(img, kp_2D))

    if x_max == img.shape[1]:
        x_max = img.shape[1]-1
    if y_max == img.shape[0]:
        y_max = img.shape[0]-1
    # Joint coordinate maxes
    left_max, right_max = x_min, x_max
    up_max, bottom_max = y_min, y_max
    # Add a buffer around the extracted bounding box
    buff = 8
    if x_min - buff >= 0:
        x_min -= buff
    if x_max + buff < img.shape[1]:
        x_max += buff
    if y_min - buff >= 0:
        y_min -= buff
    if y_max + buff < img.shape[0]:
        y_max += buff
    # Expand rectangle to square by elongating the shorter side of the rectangle
    y_diff = y_max - y_min
    x_diff = x_max - x_min
    if y_diff > x_diff:
        len_to_board_x = img.shape[1] - x_max
        # Check if square would extend beyond image. Extend on other side instead
        if x_min + y_diff < img.shape[1]:
            x_max = x_min + y_diff
        else:
            # Can't extend only on one side, extend on both sides
            x_min = x_max - (y_diff - (len_to_board_x - 1))
            x_max += (len_to_board_x - 1)
    else:
        len_to_board_y = img.shape[0] - y_max
        if y_min + x_diff < img.shape[0]:
            y_max = y_min + x_diff
        else:
            # Can't extend only on one side, extend on both sides
            y_min = y_max - (x_diff - (len_to_board_y - 1))
            y_max += (len_to_board_y - 1)


    # Extract the crop
    img_crop = img[y_min:y_max, x_min:x_max, :]

    assert(x_min >= 0 and x_max < img.shape[1] and y_min >=0 and
        y_max < img.shape[0])
    assert(y_min<y_max and x_min<x_max)

    # Translate the coordinates accordingly
    kp_2D_crop = kp_2D - np.array([[x_min, y_min]])
    # Some coordinates are slighty outside the image border. Fix
    if kp_2D_crop[:,0].max() > img_crop.shape[1]:
        idx = kp_2D_crop[:,0].argmax()
        kp_2D_crop[idx, 0] = img_crop.shape[1]
    if kp_2D_crop[:,1].max() > img_crop.shape[0]:
        idx = kp_2D_crop[:,1].argmax()
        kp_2D_crop[idx, 1] = img_crop.shape[0]

    return img_crop, kp_2D_crop

def resize(img, kp_2D, res_size):
    """Resizes the image to res_size and adjusts the 2D keypoints accordingly

    Parameters
    ----------
    img: np.array
        The image to be resized
    kp_2D: np.array
        The 2D keypoints of the hand
    res_size: tuple
        The new image size

    Returns
    -------
    img_res: np.array
        The resized image
    kp_2D_res: np.array
        The adjusted 2D keypoints
    """

    # img_res = scipy.misc.imresize(img, res_size)
    img_res = cv.resize(img, res_size)
    # Y x X
    assert(img.shape[0] == img.shape[1] and res_size[0] == res_size[1])
    skew = [res_size[0] / img.shape[0], res_size[1] / img.shape[1]]

    # Adjust the 2D keypoints
    kp_2D_res = kp_2D * np.array(skew).reshape((1,2))

    return img_res, kp_2D_res


