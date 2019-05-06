import os

import cv2.cv2 as cv2

from src.Video import Plate
from src.lp_validation.LPValidation import LPValidation
from src.lp_measurement.lp_measurement import get_height_of_license_plate
from src.utils.image_utils import get_image_patch_from_rect, save_debug_image
from src.utils.timer import timing


class LicensePlateDetection:

    def __init__(self):
        path_to_xml_classifier_file = os.path.abspath("lp_localization/lp_cascade.xml")
        self.classifier = cv2.CascadeClassifier(path_to_xml_classifier_file)
        self.validator = LPValidation()

    @timing
    def detect_license_plate_candidates(self, image, debug_mode=False):
        plate_candidates = self.process_image(image, debug_mode)
        self.validator.validate_plates(image, plate_candidates)
        for plate in plate_candidates:
            if plate.valid:
                self.measure_plate_height(image, plate)
        return plate_candidates

    def process_image(self, image, debug_mode):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lps = self.classifier.detectMultiScale(gray_image)

        if debug_mode:
            for box in lps:
                (left, top, w, h) = box
                lp_image_patch = cv2.getRectSubPix(image, (w, h), (left + w / 2, top + h / 2))
                save_debug_image(lp_image_patch, str(abs(hash(image.tostring()))), "plate_candidates")

        plates: [Plate] = []
        for box in lps:
            plate = Plate()
            left, top, width, height = box
            plate.box = [top, left, top + height, left + width]
            plates.append(plate)
        return plates

    def measure_plate_height(self, image, plate):
        image_patch = get_image_patch_from_rect(image, plate.box)
        plate_height = get_height_of_license_plate(image_patch)
        if plate_height is not None:
            plate.height = plate_height
        else:
            plate.valid = False
