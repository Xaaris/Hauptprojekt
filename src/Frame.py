from numpy.core.multiarray import ndarray


class Plate:
    box: [int, int, int, int]
    valid: bool
    height: float


class Vehicle:
    box: [int, int, int, int]
    plates: [Plate]


class Frame:
    """Class for keeping track of data found within a frame"""
    frame_number: int
    image: ndarray
    vehicles: [Vehicle]

    def __str__(self):
        retString = "Frame: " + str(self.frame_number)
        for index, vehicle in enumerate(self.vehicles):
            retString += "  Vehicle: " + str(index)
            for plate in vehicle.plates:
                if (plate.valid and hasattr(plate, "height")):
                    retString += " plate height: " + "{:.4f}".format(plate.height)
        return retString
