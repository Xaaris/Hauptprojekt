"""Miscellaneous utility functions."""
import os

import cv2.cv2 as cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from timer import timing


def letterbox_image(image, desired_size):
    """resize image with unchanged aspect ratio using padding"""

    ih, iw = image.shape[:2]
    w, h = desired_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized_image = cv2.resize(image, (nw, nh))

    delta_w = w - nw
    delta_h = h - nh
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [128, 128, 128]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image


def take_center_square(image):
    """take the center square of the original image"""

    height, width = image.shape[:2]
    min_dimension = min(height, width)
    center = (width / 2, height / 2)

    square_center_image = cv2.getRectSubPix(image, (min_dimension, min_dimension), center)

    return square_center_image


def resize_image(image, desired_size):
    """resize image to desired size"""

    resized_image = cv2.resize(image, desired_size)

    return resized_image


def get_image_patch_from_rect(image, rect):
    top, left, bottom, right = rect
    size = (right - left, bottom - top)
    center = (left + size[0] / 2, top + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


def get_image_patch_from_contour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    size = (int(w * 1.2), int(h * 1.5))
    center = (x + w / 2, y + h / 2)
    return cv2.getRectSubPix(image, size, center)


def show(image, label="image"):
    cv2.imshow(label, image)
    cv2.waitKey()


@timing
def save_debug_image(image, filename, folder=None):
    if folder:
        path = "debugImages/" + folder + "/" + filename + ".png"
    else:
        path = "debugImages/" + filename + ".png"
    cv2.imwrite(path, image)


def get_frames(path_to_video, from_sec=0, to_sec=None):
    fullpath = os.path.abspath(path_to_video)
    video = VideoFileClip(fullpath, audio=False).subclip(from_sec, to_sec)
    for frame in video.iter_frames():
        color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield color_corrected_frame
