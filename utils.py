"""Miscellaneous utility functions."""

import cv2


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
    """take the center square of the original image"""

    resized_image = cv2.resize(image, desired_size)

    return resized_image


def get_image_patch(image, box):
    top, left, bottom, right = box
    size = (right - left, bottom - top)
    center = (left + size[0] / 2, top + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


def show(image, label="image"):
    cv2.imshow(label, image)
    cv2.waitKey()


def save_debug_image(image, filename, folder=None):
    if folder:
        path = "debugImages/" + folder + "/" + filename + ".png"
    else:
        path = "debugImages/" + filename + ".png"
    cv2.imwrite(path, image)
