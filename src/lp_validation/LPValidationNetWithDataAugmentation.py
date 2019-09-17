"""
Trains a simple convnet on images to determine if they show a full license plate or not.
To enhance the training effect on the relatively small data set, data augmentation is used
Based on https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import os

import keras
import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from src.utils.image_utils import resize_image

img_rows, img_cols = 50, 150
batch_size = 32


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts the 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])
    print(model.summary())

    return model


def train_model(model):
    train_datagen = ImageDataGenerator(
        rotation_range=3,
        shear_range=3,
        brightness_range=(1, 1.2))
    test_datagen = ImageDataGenerator(brightness_range=(1, 1.2))
    train_generator = train_datagen.flow_from_directory(
        "data/train",
        target_size=(img_rows, img_cols),  # all images will be resized to 50x150
        batch_size=batch_size,
        class_mode="binary")  # since we use binary_crossentropy loss, we need binary labels
    validation_generator = test_datagen.flow_from_directory(
        "data/test",
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="binary")
    model.fit_generator(
        train_generator,
        steps_per_epoch=307 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=88 // batch_size)
    model.save_weights("model_data/lp_validation.h5")


def load_weights(model):
    model.load_weights(os.path.abspath("src/lp_validation/model_data/lp_validation.h5"))


def predict(model, license_plate_candidate):
    """returns a boolean whether the model predicts the 'license_plate_candidate' to be a valid license plate"""
    resized_patch = resize_image(license_plate_candidate, (img_cols, img_rows))
    expanded_dims_for_batch = np.expand_dims(resized_patch, axis=0)
    prediction = model.predict(expanded_dims_for_batch)
    if prediction[0][0] < 0.5 and prediction[0][1] > 0.5:
        return True
    else:
        return False
