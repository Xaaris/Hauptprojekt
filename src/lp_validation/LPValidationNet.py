"""
Trains a simple convnet on images to determine if they show a full license plate or not.
Based on https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""

import glob
import os

import cv2.cv2 as cv2
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from src.utils.image_utils import resize_image

batch_size = 32
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 50, 150
number_of_channels = 3


def read_data_set(path_to_images):
    filenames = glob.glob(os.path.join(path_to_images, '*.png'))
    images = []
    for filename in filenames:
        orig_image = cv2.imread(filename)
        resized_image = cv2.resize(orig_image, (img_cols, img_rows))
        images.append(resized_image)
    return images


def get_train_and_test_data():
    path_positives = "dataLabeling/license_plates/positives/"
    path_negatives = "dataLabeling/license_plates/negatives/"
    positives = read_data_set(path_positives)
    negatives = read_data_set(path_negatives)
    number_of_positives = len(positives)
    number_of_negatives = len(negatives)
    split_ratio = 0.8
    split_index_positives = int(number_of_positives * split_ratio)
    split_index_negatives = int(number_of_negatives * split_ratio)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(number_of_positives):
        if i < split_index_positives:
            x_train.append(positives[i])
            y_train.append(True)
        else:
            x_test.append(positives[i])
            y_test.append(True)

    for i in range(number_of_negatives):
        if i < split_index_negatives:
            x_train.append(negatives[i])
            y_train.append(False)
        else:
            x_test.append(negatives[i])
            y_test.append(False)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows, img_cols, number_of_channels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes - 1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print(model.summary())

    return model


def train_model(model):
    (x_train, y_train), (x_test, y_test) = get_train_and_test_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, number_of_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, number_of_channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights("model_data/lp_validation.h5")


def load_weights(model):
    model.load_weights(os.path.abspath("lp_validation/model_data/lp_validation.h5"))


def predict(model, license_plate_candidate):
    resized_patch = resize_image(license_plate_candidate, (img_cols, img_rows))
    expanded_dims_for_batch = np.expand_dims(resized_patch, axis=0)
    prediction = model.predict(expanded_dims_for_batch)
    return True if prediction[0][0] >= 0.9 else False

# model = create_model()
# train_model(model)
