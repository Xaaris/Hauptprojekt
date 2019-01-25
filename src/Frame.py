from numpy.core.multiarray import ndarray


class Plate:
    box: [int, int, int, int]
    valid: bool


class Vehicle:
    box: [int, int, int, int]
    plates: [Plate]


class Frame:
    """Class for keeping track of data found within a frame"""
    frame_number: int
    image: ndarray
    vehicles: [Vehicle]
