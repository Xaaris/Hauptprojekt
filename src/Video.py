from dataclasses import dataclass, field

from numpy.core.multiarray import ndarray


@dataclass
class Plate:
    """
    Contains information about a found license plate such as:
    valid: If the image contains a license plate
    confidence: how confident the validation network is, that this is a license plate
    height: height of the license plate in pixel
    box: upper left and lower right corner of the license plate
    """
    valid = False
    confidence: float = 0
    height: float = 0
    box: [int, int, int, int] = field(default_factory=list)


@dataclass
class Vehicle:
    """Class that contains information about one vehicle, where it is located and potential license plates"""
    box: [int, int, int, int] = field(default_factory=list)
    plates: [Plate] = field(default_factory=list)


@dataclass
class Frame:
    """Class for keeping track of data found within a frame"""
    frame_number: int
    image: ndarray
    vehicles: [Vehicle] = field(default_factory=list)

    def __str__(self):
        ret_string = "Frame: " + str(self.frame_number)
        for index, vehicle in enumerate(self.vehicles):
            ret_string += "  Vehicle: " + str(index)
            for plate in vehicle.plates:
                if plate.valid and hasattr(plate, "height"):
                    ret_string += " plate height: " + "{:.4f}".format(plate.height)
        return ret_string


@dataclass
class Video:
    """Contains all information about a processed video file"""
    path_to_file: str
    frames: [Frame] = field(default_factory=list)
