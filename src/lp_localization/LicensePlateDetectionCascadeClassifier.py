import os

import cv2.cv2 as cv2

from src.Frame import Plate
from src.utils.image_utils import show
from src.utils.timer import timing


class LicensePlateDetection:

    def __init__(self):
        path_to_xml_classifier_file = os.path.abspath("src/lp_localization/lp_cascade.xml")
        self.classifier = cv2.CascadeClassifier(path_to_xml_classifier_file)

    @timing
    def detect_license_plate_candidates(self, image, debug_mode=False):
        return self.process_image(image, debug_mode)

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
