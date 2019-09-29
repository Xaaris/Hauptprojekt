"""YOLO_v3 Model Defined in Keras"""

import tensorflow as tf
from tensorflow.python.keras import backend as K


def yolo_head(feats, anchors, num_classes, input_shape):
    """Convert final layer features to bounding box parameters"""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width (13 x 13)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs


def scale_boxes_to_original_image_size(box_xy, box_wh, image_shape):
    """Scale boxes from internal representation back to original image shape"""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    top = box_mins[..., 1:2]
    left = box_mins[..., 0:1]
    bottom = box_maxes[..., 1:2]
    right = box_maxes[..., 0:1]

    boxes = K.concatenate([top, left, bottom, right])

    # Scale boxes back to original image shape.
    scaling_tensor = K.concatenate([image_shape, image_shape])
    boxes = boxes * scaling_tensor
    return boxes


def process_yolo_layer_output(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)

    box_scores = box_confidence * box_class_probs
    highest_score_indexes = K.argmax(box_scores, axis=-1)
    box_classes = K.reshape(highest_score_indexes, [-1])
    highest_scores = K.max(highest_score_indexes, axis=-1)
    highest_box_scores = K.reshape(highest_scores, [-1])

    boxes = scale_boxes_to_original_image_size(box_xy, box_wh, image_shape)
    boxes = K.reshape(boxes, [-1, 4])

    return boxes, highest_box_scores, box_classes


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              score_threshold,
              iou_threshold,
              max_boxes=20):
    """Evaluate YOLO model on given input and return nms filtered boxes."""
    num_layers = len(yolo_outputs)  # This refers to the different scales at which yolo detects objects
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default or tiny yolo
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    box_classes = []
    for l in range(num_layers):
        _boxes, _box_scores, _box_classes = process_yolo_layer_output(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
        box_classes.append(_box_classes)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    box_classes = K.concatenate(box_classes, axis=0)

    scores_, boxes_, classes_ = non_max_suppression(box_scores, boxes, box_classes, max_boxes, iou_threshold, score_threshold)

    return boxes_, scores_, classes_


def non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5, score_threshold=0.3):
    """
    Applies Non-max suppression (NMS) to a set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    score_threshold -- real value, minimum score used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold, score_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
