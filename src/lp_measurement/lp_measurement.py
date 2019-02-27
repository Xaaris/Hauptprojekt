import math

import cv2.cv2 as cv2
import numpy as np

from src.utils.image_utils import show


def load_image(path):
    return cv2.imread(path)


def line_is_vertical(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y1 - y2, x1 - x2) * 180 / math.pi
    return True if 80 < abs(angle) < 100 else False


def line_is_horizontal(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y1 - y2, x1 - x2) * 180 / math.pi
    return True if abs(angle) < 10 or abs(angle) > 170 else False


def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return image


def correct_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(image)


def detect_vertical_lines(gray_image):

    v_edges = cv2.Sobel(gray_image, -1, 1, 0)
    v_edges = cv2.Canny(v_edges, 50, 150, apertureSize=3)
    show(v_edges, "v_edges")

    minLineLength = gray_image.shape[0] / 4
    maxLineGap = minLineLength / 2
    lines = cv2.HoughLinesP(v_edges, 1, np.pi / 180, threshold=10, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is not None:
        vertical_lines = []
        for line in lines:
            if line_is_vertical(line[0]):
                vertical_lines.append(line[0])
        return vertical_lines


def detect_horizontal_lines(gray_image):

    h_edges = cv2.Sobel(gray_image, -1, 0, 1)
    h_edges = cv2.Canny(h_edges, 50, 150, apertureSize=3)
    show(h_edges, "h_edges")

    minLineLength = gray_image.shape[1] / 2
    maxLineGap = minLineLength / 2
    lines = cv2.HoughLinesP(h_edges, 1, np.pi / 180, threshold=80, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is not None:
        horizontal_lines = []
        for line in lines:
            if line_is_horizontal(line[0]):
                horizontal_lines.append(line[0])
        return horizontal_lines


def find_lp_contour(image):
    # convert to HSV color scheme
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # split HSV to three chanels
    hue, saturation, value = cv2.split(hsv_image)
    # show(hue, "hue")
    # show(saturation, "saturation")
    # show(value, "value")
    # threshold to find the contour
    _, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # show(thresholded, "thresholded")
    # morphological operations
    kernel = (7, 7)
    thresholded_close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    # show(thresholded_close, "thresholded_close")
    thresholded_open = cv2.morphologyEx(thresholded_close, cv2.MORPH_OPEN, kernel)
    # show(thresholded_open, "thresholded_open")
    # The cv2.findContours method is destructive (meaning it manipulates the image you pass in)
    # so if you plan on using that image again later, be sure to clone it.
    contours = cv2.findContours(thresholded_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # keep only 10 the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # the contour that we seek for
    # loop over our 10 largest contours in the query image
    # for contour in contours:
    #     image_copy = np.copy(image)
    #     cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 1)
    #     show(image_copy, "all contours")
    for contour in contours:
        # approximate the contour
        # These methods are used to approximate the polygonal curves of a contour.
        # In order to approximate a contour, you need to supply your level of approximation precision.
        # In this case, we use 2% of the perimeter of the contour. The precision is an important value to consider.
        # If you intend on applying this code to your own projects, you’ll likely have to play around with the precision value.
        perimeter = cv2.arcLength(contour, True)
        approx_poly = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        # image_copy2 = np.copy(image)
        # cv2.drawContours(image_copy2, [approx_poly], -1, (0, 255, 0), 1)
        # show(image_copy2, "approx")
        if len(approx_poly) == 4:
            return approx_poly





test_image_path = "../lp_validation/data/train/positives/3227759838959127893.png"
image = load_image(test_image_path)
show(image, "input")
balanced_image = correct_white_balance(image)
# show(balanced_image, "balanced")

lp_contour = find_lp_contour(balanced_image)
cv2.drawContours(balanced_image, [lp_contour], -1, (0, 255, 0), 1)
show(balanced_image, "final")


# gray_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2GRAY)
# show(gray_image, "gray")
#
# vertical_lines = detect_vertical_lines(gray_image)
# horizontal_lines = detect_horizontal_lines(gray_image)
#
# image_with_lines = draw_lines(image, horizontal_lines + vertical_lines)
# show(image_with_lines, "final")
