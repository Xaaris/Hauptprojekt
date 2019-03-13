import math

import cv2.cv2 as cv2
import numpy as np
import numpy.polynomial.polynomial as poly

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
    image_copy = np.copy(image)
    if lines is not None:
        for line in lines:
            color = list(np.random.random(size=3) * 256)
            cv2.line(image_copy, line[0], line[1], color, 1)
    return image_copy


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


def get_lines_from_contour(contour):
    lines = []
    for i in range(len(contour)):
        j = (i + 1) % len(contour)
        lines.append(((contour[i][0][0], contour[i][0][1]), (contour[j][0][0], contour[j][0][1])))
    return lines


def get_2_longest_lines_from_contour(contour):
    lines = []
    for i in range(len(contour)):
        j = (i + 1) % len(contour)
        lines.append(((contour[i][0][0], contour[i][0][1]), (contour[j][0][0], contour[j][0][1])))
    two_longest_lines = sorted(lines, key=get_line_length, reverse=True)[:2]
    return two_longest_lines


def find_lp_contour(image):
    # convert to HSV color scheme
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # split HSV to three chanels
    hue, saturation, value = cv2.split(hsv_image)
    # show(hue, "hue")
    # show(saturation, "saturation")
    show(value, "value")
    # threshold to find the contour
    _, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show(thresholded, "thresholded")
    # morphological operations
    kernel = (7, 7)
    thresholded_close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    show(thresholded_close, "thresholded_close")
    thresholded_open = cv2.morphologyEx(thresholded_close, cv2.MORPH_OPEN, kernel)
    show(thresholded_open, "thresholded_open")
    # The cv2.findContours method is destructive (meaning it manipulates the image you pass in)
    # so if you plan on using that image again later, be sure to clone it.
    contours = cv2.findContours(thresholded_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # keep only 10 the largest ones
    biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # the contour that we seek for
    # loop over our 10 largest contours in the query image
    image_copy = np.copy(image)
    cv2.drawContours(image_copy, [biggest_contour], -1, (0, 255, 0), 1)
    show(image_copy, "all contours")

    perimeter = cv2.arcLength(biggest_contour, True)
    approx_poly = cv2.approxPolyDP(biggest_contour, 0.01 * perimeter, True)
    image_copy2 = np.copy(image)
    cv2.drawContours(image_copy2, [approx_poly], -1, (0, 255, 0), 1)
    show(image_copy2, "approx")
    return approx_poly


def perfect_line(image, line):
    ten_percent_of_image_height = image.shape[0] * 0.1
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    angle_rad = math.atan2(dy, dx)
    points_on_line = getEquidistantPoints(pt1, pt2, 10)
    for point in points_on_line:
        first_point = get_point_at(point, ten_percent_of_image_height, angle_rad + np.pi / 2)
        second_point = get_point_at(point, - ten_percent_of_image_height, angle_rad + np.pi / 2)
        cv2.line(image, first_point, second_point, (0, 0, 255), 1)
        number_of_measuring_points = 10
        measuring_points = getEquidistantPoints(first_point, second_point, number_of_measuring_points)
        gray_values = []
        for measuring_point in measuring_points:
            gray_values.append(float(gray_image[measuring_point[1]][measuring_point[0]]))

        line_point_in_array = find_actual_line_point(gray_values, ten_percent_of_image_height * 2)
        if line_point_in_array is not None:
            line_point = get_point_at(first_point, line_point_in_array, -(angle_rad + np.pi / 2))
            cv2.line(image, line_point, get_point_at(line_point, 30, angle_rad), (0, 255, 0), 1)

    show(image)


def find_actual_line_point(gray_values, length_of_measuring_line):
    coeffs = poly.polyfit(np.linspace(0, length_of_measuring_line, len(gray_values)), gray_values, 3)
    polynom = np.poly1d(coeffs[::-1])
    first_derivative = polynom.deriv()
    crit = first_derivative.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = first_derivative.deriv(2)(r_crit)
    x_min = r_crit[test > 0]
    if 0 <= x_min <= length_of_measuring_line:
        return x_min


def getEquidistantPoints(pt1, pt2, num_of_points):
    return zip(np.linspace(pt1[0], pt2[0], num_of_points, dtype=int),
               np.linspace(pt1[1], pt2[1], num_of_points, dtype=int))


def get_point_at(origin, dist, theta):
    return int(origin[0] + dist * math.cos(theta)), int(origin[1] + dist * math.sin(theta))


def get_line_length(line):
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    length = math.hypot(dx, dy)
    return length


test_image_path = "3227759838959127893.png"
image = load_image(test_image_path)
balanced_image = correct_white_balance(image)
show(balanced_image, "balanced")

lp_contour = find_lp_contour(balanced_image)

lines = get_2_longest_lines_from_contour(lp_contour)
image_with_lines = draw_lines(balanced_image, lines)
show(image_with_lines, "image_with_lines")

for line in lines:
    perfect_line(image, line)
