import math

import cv2
import numpy as np
import numpy.polynomial.polynomial as poly

from src.utils.image_utils import show


def load_image(path):
    return cv2.imread(path)


def draw_lines(image, lines, color=None):
    image_copy = np.copy(image)
    if lines is not None:
        for line in lines:
            if color is None:
                color = list(np.random.random(size=3) * 256)
            cv2.line(image_copy, line[0], line[1], color, 1)
    return image_copy


def correct_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    return wb.balanceWhite(image)


def get_2_longest_lines_from_contour(contour):
    lines = []
    for i in range(len(contour)):
        j = (i + 1) % len(contour)
        lines.append(((contour[i][0][0], contour[i][0][1]), (contour[j][0][0], contour[j][0][1])))
    two_longest_lines = sorted(lines, key=get_line_length, reverse=True)[:2]
    return two_longest_lines


def find_lp_contour(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    _, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show(thresholded, "thresholded")
    kernel = (7, 7)
    thresholded_close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded_open = cv2.morphologyEx(thresholded_close, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(thresholded_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    image_copy = np.copy(image)
    cv2.drawContours(image_copy, [biggest_contour], -1, (0, 255, 0), 1)

    perimeter = cv2.arcLength(biggest_contour, True)
    approx_poly = cv2.approxPolyDP(biggest_contour, 0.01 * perimeter, True)
    image_copy2 = np.copy(image)
    cv2.drawContours(image_copy2, [approx_poly], -1, (0, 255, 0), 1)
    return approx_poly


def perfect_line(image, line):
    ten_percent_of_image_height = image.shape[0] * 0.1
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    angle_rad = math.atan2(dy, dx)
    points_on_line = get_equidistant_points(pt1, pt2, 10)
    actual_line_points = []
    for point in points_on_line:
        first_point = get_point_at(point, ten_percent_of_image_height, angle_rad + np.pi / 2)
        second_point = get_point_at(point, - ten_percent_of_image_height, angle_rad + np.pi / 2)
        cv2.line(image, first_point, second_point, (0, 0, 255), 1)
        number_of_measuring_points = 10
        measuring_points = get_equidistant_points(first_point, second_point, number_of_measuring_points)
        gray_values = []
        for measuring_point in measuring_points:
            gray_values.append(float(gray_image[measuring_point[1]][measuring_point[0]]))

        line_point_in_array = find_actual_line_point(gray_values, ten_percent_of_image_height * 2)
        if line_point_in_array is not None:
            actual_line_point = get_point_at(first_point, line_point_in_array, angle_rad - np.pi / 2)
            actual_line_points.append(actual_line_point)
            cv2.line(image, actual_line_point, get_point_at(actual_line_point, ten_percent_of_image_height, angle_rad), (0, 255, 0), 1)

    show(image, "measuring")
    line_start, line_end = best_fit_line_from_points(actual_line_points)
    return line_start, line_end


def best_fit_line_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line_coeffs = poly.polyfit(xs, ys, 1)
    line_polynom = np.poly1d(line_coeffs[::-1])
    line_start = (xs[0], int(line_polynom(xs[0])))
    line_end = (xs[-1], int(line_polynom(xs[-1])))
    return line_start, line_end


def find_actual_line_point(gray_values, length_of_measuring_line):
    coeffs = poly.polyfit(np.linspace(0, length_of_measuring_line, len(gray_values)), gray_values, 3)
    polynom = np.poly1d(coeffs[::-1])
    first_derivative = polynom.deriv()
    second_derivative = first_derivative.deriv()
    roots = second_derivative.roots
    if 0 <= roots[0] <= length_of_measuring_line:
        return roots[0]


def get_equidistant_points(pt1, pt2, num_of_points):
    return zip(np.linspace(pt1[0], pt2[0], num_of_points, dtype=int),
               np.linspace(pt1[1], pt2[1], num_of_points, dtype=int))


def get_point_at(origin, dist, theta):
    return int(origin[0] + dist * math.cos(theta)), int(origin[1] + dist * math.sin(theta))


def get_line_length(line):
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    length = math.hypot(dx, dy)
    return length


def get_nearest_point_on_line(line, point):
    pt1, pt2 = line
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = point

    k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    x4 = x3 - k * (y2 - y1)
    y4 = y3 + k * (x2 - x1)
    if min(x1, x2) <= x4 <= max(x1, x2):
        return x4, y4


def get_average_distance_of_lines(line1, line2):
    inner_edge_lines = []
    for pt in line1:
        lot_fuss = get_nearest_point_on_line(line2, pt)
        if lot_fuss is not None:
            inner_edge_lines.append([lot_fuss, pt])
    for pt in line2:
        lot_fuss = get_nearest_point_on_line(line1, pt)
        if lot_fuss is not None:
            inner_edge_lines.append([lot_fuss, pt])
    assert len(inner_edge_lines) == 2
    average_distance_between_lines = 0
    for line in inner_edge_lines:
        average_distance_between_lines += get_line_length(line)
    return average_distance_between_lines / 2


test_image_path = "2505119030049077848.png"
image = load_image(test_image_path)
balanced_image = correct_white_balance(image)
show(balanced_image, "balanced")

lp_contour = find_lp_contour(balanced_image)

lines = get_2_longest_lines_from_contour(lp_contour)
image_with_lines = draw_lines(balanced_image, lines, (0, 0, 255))
show(image_with_lines, "image_with_lines")

final_lines = [perfect_line(image, line) for line in lines]
image_with_lines = draw_lines(image_with_lines, final_lines, (0, 255, 0))
show(image_with_lines, "image_with_lines")
print(get_average_distance_of_lines(final_lines[0], final_lines[1]))
