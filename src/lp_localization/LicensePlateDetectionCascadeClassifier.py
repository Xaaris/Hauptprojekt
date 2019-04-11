import os

import cv2.cv2 as cv2

from src.Frame import Plate
from src.lp_validation.LPValidationNet import create_model, load_weights, predict
from src.lp_measurement.lp_measurement import get_height_of_license_plate
from src.utils.image_utils import show, get_image_patch_from_rect
from src.utils.timer import timing


class LicensePlateDetection:

    def __init__(self):
        path_to_xml_classifier_file = os.path.abspath("lp_localization/lp_cascade.xml")
        self.classifier = cv2.CascadeClassifier(path_to_xml_classifier_file)
        self.lp_validation_model = create_model()
        load_weights(self.lp_validation_model)

    @timing
    def detect_license_plate_candidates(self, image, debug_mode=False):
        plate_candidates = self.process_image(image, debug_mode)
        self.validate_plates(image, plate_candidates)
        for plate in plate_candidates:
            if plate.valid:
                self.measure_plate_height(image, plate)
        return plate_candidates

    def process_image(self, image, debug_mode):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lps = self.classifier.detectMultiScale(gray_image)
        if debug_mode:
            for (x, y, w, h) in lps:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            show(image)
        plates: [Plate] = []
        for box in lps:
            plate = Plate()
            left, top, width, height = box
            plate.box = [top, left, top + height, left + width]
            plates.append(plate)
        return plates

    @timing
    def validate_plates(self, image, plate_candidates):
        for plate in plate_candidates:
            image_patch = get_image_patch_from_rect(image, plate.box)
            plate.valid = predict(self.lp_validation_model, image_patch)


    def measure_plate_height(self, image, plate):
        image_patch = get_image_patch_from_rect(image, plate.box)
        plate_height = get_height_of_license_plate(image_patch)
        if (plate_height is not None):
            plate.height = plate_height
