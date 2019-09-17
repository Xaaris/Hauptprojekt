import os
import numpy as np

from src.utils.image_utils import resize_image, get_image_patch_from_rect
from src.utils.timer import timing
from src.lp_validation.TrainLPValidationNet import create_model

img_rows, img_cols = 50, 150


class LPValidation:

    def __init__(self):
        self.lp_validation_model = create_model()
        self.lp_validation_model.load_weights(os.path.abspath("lp_validation/model_data/lp_validation.h5"))

    @timing
    def validate_plates(self, image, plate_candidates):
        """
        Takes a set of 'plate_candidates' and uses a cnn to validate if those coordinates contain a license
        plate on the original 'image'. The result is saved within the Plate model.
        """
        if len(plate_candidates) > 0:
            for plate in plate_candidates:
                image_patch = get_image_patch_from_rect(image, plate.box)
                plate.confidence = self._predict(image_patch)
            plate_with_heighest_confidence = sorted(plate_candidates, key=lambda p: (p.confidence, ), reverse=True)[0]
            if plate_with_heighest_confidence.confidence >= 0.9:
                plate_with_heighest_confidence.valid = True

    def _predict(self, license_plate_candidate):
        resized_patch = resize_image(license_plate_candidate, (img_cols, img_rows))
        expanded_dims_for_batch = np.expand_dims(resized_patch, axis=0)
        prediction = self.lp_validation_model.predict(expanded_dims_for_batch)
        return prediction[0][0]
