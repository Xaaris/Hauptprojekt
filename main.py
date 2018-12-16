import time

import cv2.cv2 as cv2
import numpy as np

from LicensePlateDetection import LicensePlateDetection
from timer import print_timing_results
from utils import take_center_square, save_debug_image, get_image_patch, get_frames
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    start = time.time()
    for frame_counter, frame in enumerate(get_frames("testFiles/IMG_2993.m4v", 0, 1)):
        frame = take_center_square(frame)
        frame_copy = np.copy(frame)
        vehicle_boxes = yolo.detect_vehicle(frame)
        for vehicle_box in vehicle_boxes:
            top, left, bottom, right = vehicle_box
            image_copy = cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 0, 255), 2)
            car_image = get_image_patch(frame, vehicle_box)
            license_plate_detection = LicensePlateDetection(car_image)
            plate = license_plate_detection.detect_license_plate()
            if plate is not None:
                cv2.drawContours(frame_copy, [plate], -1, (127, 0, 255), 2, offset=(left, top))
        save_debug_image(frame_copy, "frame_" + str(frame_counter), "processed_frames")

    total_duration = time.time() - start
    fps = frame_counter / total_duration
    print("Total duration: " + str(total_duration) + "s, FPS: " + str(fps))

print_timing_results()
