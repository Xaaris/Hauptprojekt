import time

from src.Frame import Frame
from src.lp_localization.LicensePlateDetectionCascadeClassifier import LicensePlateDetection
from src.car_detection.yolo import YOLO
from src.utils import timer
from src.utils.image_utils import save_debug_image, get_image_patch_from_rect, get_frames, draw_processed_image

if __name__ == "__main__":
    yolo = YOLO()
    license_plate_detection = LicensePlateDetection()
    start = time.time()
    path_to_video = "testFiles/IMG_2993.m4v"
    frame = Frame()
    for frame.frame_number, frame.image in enumerate(get_frames(path_to_video, 0, 3)):
        frame.vehicles = yolo.detect_vehicle(frame.image)
        for vehicle in frame.vehicles:
            car_image = get_image_patch_from_rect(frame.image, vehicle.box)
            vehicle.plates = license_plate_detection.detect_license_plate_candidates(car_image)

        processed_frame = draw_processed_image(frame)
        save_debug_image(processed_frame, "frame_" + str(frame.frame_number), "processed_frames", resize_to=(1920, 1080))

    total_duration = time.time() - start
    fps = frame.frame_number / total_duration
    print("\nTotal duration: " + str(total_duration) + "s, FPS: " + str(fps))

    timer.print_timing_results()