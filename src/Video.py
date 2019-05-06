from numpy.core.multiarray import ndarray


class Plate:
    box: [int, int, int, int]
    valid = False
    confidence: float
    height: float


class Vehicle:
    box: [int, int, int, int]
    plates: [Plate]


class Frame:
    """Class for keeping track of data found within a frame"""
    frame_number: int
    image: ndarray
    vehicles: [Vehicle] = []

    def __init__(self, frame_number, image):
        self.frame_number = frame_number
        self.image = image

    def __str__(self):
        retString = "Frame: " + str(self.frame_number)
        for index, vehicle in enumerate(self.vehicles):
            retString += "  Vehicle: " + str(index)
            for plate in vehicle.plates:
                if (plate.valid and hasattr(plate, "height")):
                    retString += " plate height: " + "{:.4f}".format(plate.height)
        return retString


class Video:
    frames: [Frame] = []

    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
