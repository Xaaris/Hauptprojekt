"""
This is the main entrypoint to the application

Given a video file and the appropriate camera calibration data, it runs the conv net and estimates the velocity
of the vehicle in the video.
"""

import time

from src.Video import Frame, Video
from src.camera_calibration.CameraCalibration import CameraCalibration
from src.lp_localization.LicensePlateDetectionCascadeClassifier import LicensePlateDetection
from src.car_detection.yolo import YOLO
from src.speed_estimation.SpeedEstimator import SpeedEstimator
from src.utils import timer
from src.utils.image_utils import save_debug_image, get_image_patch_from_rect, get_frames, draw_processed_image

VIDEO_FILE = "../testFiles/25,74kmh.mov"

CAMERA_MODEL = "camera_calibration/camera_calibration_iPhoneXR_4k_60.npz"

if __name__ == "__main__":
    yolo = YOLO()
    license_plate_detection = LicensePlateDetection()
    camera_calibration = CameraCalibration(CAMERA_MODEL)
    start = time.time()
    video = Video(VIDEO_FILE)
    for frame_number, image in enumerate(get_frames(video.path_to_file, from_sec=7, to_sec=9)):
        image = camera_calibration.undistort(image)
        frame = Frame(frame_number, image)
        video.frames.append(frame)
        frame.vehicles = yolo.detect_vehicle(frame.image)
        for vehicle in frame.vehicles:
            car_image = get_image_patch_from_rect(frame.image, vehicle.box)
            vehicle.plates = license_plate_detection.detect_license_plate_candidates(car_image)

        # processed_frame = draw_processed_image(frame)
        # save_debug_image(processed_frame, "frame_" + str(frame.frame_number), "processed_frames", resize_to=(1920, 1080))
        print(frame)

    total_duration = time.time() - start
    fps = frame.frame_number / total_duration
    print("\nTotal duration: {0:.2f}, FPS: {1:.2f}\n".format(total_duration, fps))

    timer.print_timing_results()

    estimator = SpeedEstimator()
    estimator.estimate_speed_of_vehicle(video)
