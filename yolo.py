"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model

from model import yolo_eval
from utils import resize_image


class YOLO(object):
    _defaults = {
        # "model_path": 'model_data/yolo.h5',
        # "anchors_path": 'model_data/yolo_anchors.txt',
        "model_path": 'model_data/yolo_tiny.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, parameter_name):
        if parameter_name in cls._defaults:
            return cls._defaults[parameter_name]
        else:
            return "Unrecognized attribute name '" + parameter_name + "'"

    def __init__(self):
        self.__dict__.update(self._defaults)  # set up default values
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        self.yolo_model = load_model(model_path, compile=False)
        assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(self.yolo_model.output) * (
                num_classes + 5), 'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, self.score, self.iou)
        return boxes, scores, classes

    def detect_vehicle(self, image):
        resized_image = resize_image(image, self.model_image_size)
        image_data = np.array(resized_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # Where the magic happens
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[1], image.shape[0]],
                K.learning_phase(): 0
            })

        out_class_names = [self.class_names[class_index] for class_index in out_classes]
        print("Found the following objects: " + str(out_class_names))

        vehicle_boxes = []
        for i, class_name in enumerate(out_class_names):
            if class_name == "car" or class_name == "bus" or class_name == "truck":
                vehicle_boxes.append(out_boxes[i].round())

        return vehicle_boxes
