import os
import numpy as np

from src.utils.image_utils import resize_image, get_image_patch_from_rect
from src.utils.timer import timing
from src.lp_validation.TrainLPValidationNet import create_model

img_rows, img_cols = 50, 150


class LPValidation:

    def __init__(self):
        self.lp_validation_model = create_model()
        self.load_weights(self.lp_validation_model)

    def load_weights(self, model):
        model.load_weights(os.path.abspath("lp_validation/model_data/lp_validation.h5"))

    @timing
    def validate_plates(self, image, plate_candidates):
        if len(plate_candidates) > 0:
            for plate in plate_candidates:
                image_patch = get_image_patch_from_rect(image, plate.box)
                plate.confidence = self.predict(image_patch)
            plate_with_heighest_confidence = sorted(plate_candidates, key=lambda p: (p.confidence, ), reverse=True)[0]
            if plate_with_heighest_confidence.confidence >= 0.9:
                plate_with_heighest_confidence.valid = True

    def predict(self, license_plate_candidate):
        resized_patch = resize_image(license_plate_candidate, (img_cols, img_rows))
        expanded_dims_for_batch = np.expand_dims(resized_patch, axis=0)
        prediction = self.lp_validation_model.predict(expanded_dims_for_batch)
        return prediction[0][0]
