import os
import time

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from LicensePlateDetection import LicensePlateDetection
from utils import take_center_square, save_debug_image, get_image_patch
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    start = time.time()
    fullpath = os.path.abspath("testFiles/IMG_2993.m4v")
    clip = VideoFileClip(fullpath, audio=False).subclip(0, 3)
    frame_counter = 0
    for frame in clip.iter_frames():
        frame_counter += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
