from src.Video import Video, Frame


class SpeedEstimator:

    # LENS_FACTOR = 361  # iPhone 8+ 4k 60
    LENS_FACTOR = 343  # iPhone XR 4k 60
    FRAME_RATE = 60

    def estimate_speed_of_vehicle(self, video: Video):
        """
        Takes a fully processed 'Video' file and calculates the speed of the first car it finds by calculating the
        distance travelled between frames based on the height of the license plate in each frame.
        """
        first_frame_with_valid_plate = self._first_frame_with_valid_plate(video)
        last_frame_with_valid_plate = self._last_frame_with_valid_plate(video)
        trimmed_video = video.frames[first_frame_with_valid_plate:last_frame_with_valid_plate + 1]

        speed_estimations = [s for s in self._yield_speed_estimations(trimmed_video)]

        average_speed = sum(speed_estimations) / len(speed_estimations)
        return average_speed

    def _yield_speed_estimations(self, trimmed_video):
        last_plate_height = self._get_plate_height_of_first_valid_plate(trimmed_video[0])
        elapsed_frames = 1
        for frame in trimmed_video[1:]:
            current_plate_height = self._get_plate_height_of_first_valid_plate(frame)
            if current_plate_height is not None:
                current_distance_to_plate = self.LENS_FACTOR / current_plate_height
                last_distance_to_plate = self.LENS_FACTOR / last_plate_height
                distance_delta_in_m = last_distance_to_plate - current_distance_to_plate
                avg_distance_per_frame = distance_delta_in_m / elapsed_frames
                estimated_speed = avg_distance_per_frame * self.FRAME_RATE * 3.6

                for _ in range(elapsed_frames):
                    yield estimated_speed

                # reset
                last_plate_height = current_plate_height
                elapsed_frames = 1

            else:
                elapsed_frames += 1


    def _get_plate_height_of_first_valid_plate(self, frame: Frame):
        if not frame.vehicles or not frame.vehicles[0].plates:
            return None
        for plate in filter(lambda it: it.valid, frame.vehicles[0].plates):
            return plate.height

    def _first_frame_with_valid_plate(self, video: Video):
        for frame in video.frames:
            for vehicle in frame.vehicles:
                for plate in vehicle.plates:
                    if plate.valid:
                        return frame.frame_number

    def _last_frame_with_valid_plate(self, video: Video):
        last_frame_with_valid_plate = None
        for frame in video.frames:
            for vehicle in frame.vehicles:
                for plate in vehicle.plates:
                    if plate.valid:
                        last_frame_with_valid_plate = frame.frame_number
        return last_frame_with_valid_plate if last_frame_with_valid_plate is not None else frame.frame_number
