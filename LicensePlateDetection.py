import math

import cv2.cv2 as cv2
import numpy as np

from timer import timing
from utils import show


@timing
class LicensePlateDetection:

    def __init__(self, image, aspect_ratio_range=(1.8, 8.5), morph_closing_shape=(36, 216), morph_opening_shape=(3, 3),
                 min_extend=0.4, max_angle=10):
        self.original_img_height = image.shape[0]
        self.original_img_width = image.shape[1]

        self.input_image = image
        expected_plate_size = (self.original_img_width / 4.9, self.original_img_height / 12.8)
        self.plate_aspect_ratio_range = aspect_ratio_range
        self.plate_min_width = expected_plate_size[0] / 2
        self.plate_max_width = expected_plate_size[0] * 2
        self.plate_min_height = expected_plate_size[1] / 2
        self.plate_max_height = expected_plate_size[1] * 2
        self.morph_closing_shape = (math.ceil(self.original_img_width / morph_closing_shape[0]),
                                    math.ceil(self.original_img_width / morph_closing_shape[1]))
        self.morph_opening_shape = morph_opening_shape
        self.min_plate_extend = min_extend
        self.plate_max_angle = max_angle

    @timing
    def detect_license_plate(self, debug_mode=False):
        potential_plates = self.process_image(debug_mode)
        return self.find_best_plate(potential_plates)

    @timing
    def detect_license_plate_candidates(self, debug_mode=False):
        return self.process_image(debug_mode)

    def process_image(self, debug_mode):
        result_plates = []
        working_image = np.copy(self.input_image)

        if debug_mode: show(working_image, "input_image")
        working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        if debug_mode: show(working_image, "after cvtColor")

        working_image = cv2.Sobel(working_image, -1, 1, 0)
        if debug_mode: show(working_image, "after sobel")

        _, working_image = cv2.threshold(working_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug_mode: show(working_image, "after threshold")

        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_closing_shape)
        working_image = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, closing_kernel)
        if debug_mode: show(working_image, "after morph closing")

        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_opening_shape)
        working_image = cv2.morphologyEx(working_image, cv2.MORPH_OPEN, opening_kernel)
        if debug_mode: show(working_image, "after morph opening")

        _, contours, _ = cv2.findContours(working_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tmp_img_for_contours = np.copy(self.input_image)
        cv2.drawContours(tmp_img_for_contours, contours, -1, (255, 0, 255), 2)
        if debug_mode: show(tmp_img_for_contours, "all contours")

        tmp_img_for_valid_contours = np.copy(self.input_image)
        for contour in contours:

            if self.is_valid_contour(contour):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  # cast float to long

                result_plates.append(box)

                cv2.drawContours(tmp_img_for_valid_contours, [box], -1, (255, 0, 0), 1)
                cv2.drawContours(tmp_img_for_valid_contours, contour, -1, (255, 0, 255), 2)

        if debug_mode: show(tmp_img_for_valid_contours, "valid contours")
        return result_plates

    def find_best_plate(self, plates):
        if len(plates) < 1:
            return None
        sorted_plates = sorted(plates, key=lambda plate: abs(self.original_img_height * 0.66 - plate[0][0]))
        return sorted_plates[0]

    def is_valid_contour(self, contour):
        rect = cv2.minAreaRect(contour)

        if 45 < abs(rect[2]) < 135:  # rect is rotated ~90 degrees
            width = rect[1][1]
            height = rect[1][0]
        else:
            width = rect[1][0]
            height = rect[1][1]

        if self.plate_min_width < width < self.plate_max_width and self.plate_min_height < height < self.plate_max_height:
            aspect_ratio = float(width) / height
            if self.plate_aspect_ratio_range[0] < aspect_ratio < self.plate_aspect_ratio_range[1]:

                extend = cv2.contourArea(contour) / (rect[1][0] * rect[1][1])
                if extend > self.min_plate_extend:

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box_copy = list(box)
                    point = box_copy[0]
                    del (box_copy[0])
                    distances_to_first_point = [((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) for p in box_copy]
                    sorted_dists = sorted(distances_to_first_point)
                    opposite_point = box_copy[distances_to_first_point.index(sorted_dists[1])]

                    if abs(point[0] - opposite_point[0]) > 0:
                        contour_angle = abs(float(point[1] - opposite_point[1])) / abs(point[0] - opposite_point[0])
                        contour_angle = rad_to_deg(math.atan(contour_angle))

                        if contour_angle <= self.plate_max_angle:
                            return True
        return False


def rad_to_deg(angle):
    return angle * 180 / np.pi


def rotate_and_resize(image, rotation_matrix, old_size, new_size):
    rotated_image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=old_size)
    resized_image = cv2.getRectSubPix(rotated_image, new_size, (int(old_size[0] / 2), int(old_size[1] / 2)))
    return resized_image
