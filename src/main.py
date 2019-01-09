import time

import cv2.cv2 as cv2
import numpy as np

from src.LicensePlateDetection import LicensePlateDetection
from src.utils.timer import print_timing_results
from src.utils.image_utils import save_debug_image, get_image_patch_from_rect, get_frames, get_image_patch_from_contour
from src.car_detection.yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    start = time.time()
    for frame_counter, frame in enumerate(get_frames("testFiles/IMG_2993.m4v", 0, 3)):
        frame_copy = np.copy(frame)
        vehicle_boxes = yolo.detect_vehicle(frame)
        for vehicle_counter, vehicle_box in enumerate(vehicle_boxes):
            top, left, bottom, right = vehicle_box
            image_copy = cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 0, 255), 2)
            car_image = get_image_patch_from_rect(frame, vehicle_box)
            save_debug_image(car_image, "frame_" + str(frame_counter) + "_car_" + str(vehicle_counter), "found_vehicles")
            license_plate_detection = LicensePlateDetection(car_image)
            plate_candidates = license_plate_detection.detect_license_plate_candidates()
            for plate_counter, candidate in enumerate(plate_candidates):
                candidate_image_patch = get_image_patch_from_contour(car_image, candidate)
                save_debug_image(candidate_image_patch, "frame_" + str(frame_counter) + "_car_" + str(vehicle_counter) + "_plate_" + str(plate_counter), "plate_candidates")
            cv2.drawContours(frame_copy, plate_candidates, -1, (127, 0, 255), 2, offset=(left, top))
        save_debug_image(frame_copy, "frame_" + str(frame_counter), "processed_frames")

    total_duration = time.time() - start
    fps = frame_counter / total_duration
    print("\nTotal duration: " + str(total_duration) + "s, FPS: " + str(fps))

print_timing_results()
