"""
Various functions that are used to get an accurate height measurement of a license plate.
"""

import math

import cv2
import numpy as np
import numpy.polynomial.polynomial as poly

from src.utils.image_utils import show

_NUM_OF_HORIZONTAL_MEASURING_POINTS = 10
_NUM_OF_VERTICAL_MEASURING_POINTS = 10


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
    """This can be used return the top and bottom lines of a license plate contour"""
    lines = []
    for i in range(len(contour)):
        j = (i + 1) % len(contour)
        lines.append(((contour[i][0][0], contour[i][0][1]), (contour[j][0][0], contour[j][0][1])))
    two_longest_lines = sorted(lines, key=get_line_length, reverse=True)[:2]
    return two_longest_lines


def find_lp_contour(image):
    """Returns a poly that roughly encapsulates the license plate at its outer edges"""

    # Thresholding the image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    _, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cleaning up the thresholded image
    kernel = (7, 7)
    thresholded_close = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded_open = cv2.morphologyEx(thresholded_close, cv2.MORPH_OPEN, kernel)

    # Retrieving the biggest contour
    contours = cv2.findContours(thresholded_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Laying a poly around it
    perimeter = cv2.arcLength(biggest_contour, True)
    approx_poly = cv2.approxPolyDP(biggest_contour, 0.01 * perimeter, True)
    return approx_poly


def perfect_line(image, line):
    """
    This method takes a line and tries to align it as closely as possible with a high contrast boundary found on the
    image in its vicinity. It does so by measuring at multiple points along the line how far of it is from the highest
    contrast boundary and uses curve fitting to adopt the line accordingly.
    """
    ten_percent_of_image_height = image.shape[0] * 0.1
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    angle_rad = math.atan2(dy, dx)
    points_on_line = get_equidistant_points(pt1, pt2, _NUM_OF_HORIZONTAL_MEASURING_POINTS)
    actual_line_points = []
    for point in points_on_line:
        first_point = get_pixel_at(point, ten_percent_of_image_height, angle_rad + np.pi / 2)
        second_point = get_pixel_at(point, - ten_percent_of_image_height, angle_rad + np.pi / 2)
        cv2.line(image, first_point, second_point, (0, 0, 255), 1)
        measuring_points = get_equidistant_points(first_point, second_point, _NUM_OF_VERTICAL_MEASURING_POINTS)
        gray_values = []
        for measuring_point in measuring_points:
            y = min(measuring_point[1], gray_image.shape[0] - 1)
            x = min(measuring_point[0], gray_image.shape[1] - 1)
            gray_values.append(float(gray_image[y][x]))

        line_point_in_array = find_actual_line_point(gray_values, ten_percent_of_image_height * 2)
        if line_point_in_array is not None:
            actual_line_point = get_point_at(first_point, line_point_in_array, angle_rad - np.pi / 2)
            actual_line_points.append(actual_line_point)
    #         cv2.line(image, (int(actual_line_point[0]), int(actual_line_point[1])), get_pixel_at(actual_line_point, ten_percent_of_image_height, angle_rad), (0, 255, 0), 1)
    #
    # show(image, "measuring")
    line_start, line_end = best_fit_line_from_points(actual_line_points)
    return line_start, line_end


def best_fit_line_from_points(points):
    """Returns the best fit line going through the list of points"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line_coeffs = poly.polyfit(xs, ys, 1)
    line_polynom = np.poly1d(line_coeffs[::-1])
    line_start = (int(xs[0]), int(line_polynom(xs[0])))
    line_end = (int(xs[-1]), int(line_polynom(xs[-1])))
    return line_start, line_end


def find_actual_line_point(gray_values, length_of_measuring_line):
    """Returns point of highest contrast along the line of gray_values"""
    coeffs = poly.polyfit(np.linspace(0, length_of_measuring_line, len(gray_values)), gray_values, 3)
    polynom = np.poly1d(coeffs[::-1])
    first_derivative = polynom.deriv()
    second_derivative = first_derivative.deriv()
    roots = second_derivative.roots
    if 0 <= roots[0] <= length_of_measuring_line:
        return roots[0]


def get_equidistant_points(pt1, pt2, num_of_points):
    """Returns num_of_points points equally spaced out along the line of pt1 and pt2 including those two points"""
    return zip(np.linspace(pt1[0], pt2[0], num_of_points, dtype=int),
               np.linspace(pt1[1], pt2[1], num_of_points, dtype=int))


def get_pixel_at(origin, dist, theta):
    """Returns the closest pixel (rounded int value) that lays 'dist' distance away from 'origin' in direction 'theta'"""
    return int(origin[0] + dist * math.cos(theta)), int(origin[1] + dist * math.sin(theta))


def get_point_at(origin, dist, theta):
    """Returns the point that lays 'dist' distance away from 'origin' in direction 'theta'"""
    return origin[0] + dist * math.cos(theta), origin[1] + dist * math.sin(theta)


def get_line_length(line):
    pt1, pt2 = line
    dx, dy = pt1[0] - pt2[0], pt1[1] - pt2[1]
    length = math.hypot(dx, dy)
    return length


def get_nearest_point_on_line(line, point):
    """Returns the perpendicular point on the line. In case it falls outside the line, it returns None """
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
    """Returns the average distance between tow lines"""
    if get_line_length(line1) > 0 and get_line_length(line2) > 0:
        inner_edge_lines = []
        for pt in line1:
            lot_fuss = get_nearest_point_on_line(line2, pt)
            if lot_fuss is not None:
                inner_edge_lines.append([lot_fuss, pt])
        for pt in line2:
            lot_fuss = get_nearest_point_on_line(line1, pt)
            if lot_fuss is not None:
                inner_edge_lines.append([lot_fuss, pt])
        average_distance_between_lines = 0
        for line in inner_edge_lines:
            average_distance_between_lines += get_line_length(line)
        if len(inner_edge_lines) > 0:
            return average_distance_between_lines / len(inner_edge_lines)


def get_height_of_license_plate(lp_image):
    balanced_image = correct_white_balance(lp_image)
    lp_contour = find_lp_contour(balanced_image)
    lines = get_2_longest_lines_from_contour(lp_contour)
    final_lines = [perfect_line(balanced_image, line) for line in lines]
    return get_average_distance_of_lines(final_lines[0], final_lines[1])


if __name__ == "__main__":
    lp_image = load_image("4386399993575093578.png")
    balanced_image = correct_white_balance(lp_image)
    lp_contour = find_lp_contour(balanced_image)
    lines = get_2_longest_lines_from_contour(lp_contour)
    final_lines = [perfect_line(balanced_image, line) for line in lines]
    augmented_image = draw_lines(lp_image, final_lines)
    show(augmented_image)
