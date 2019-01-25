"""
Trains a simple convnet on images to determine if they show a full license plate or not.
Based on https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

img_rows, img_cols = 50, 150
batch_size = 16

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(
    rotation_range=3,
    shear_range=3,
    brightness_range=(1, 1.2))

test_datagen = ImageDataGenerator(brightness_range=(1, 1.2))

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_rows, img_cols),  # all images will be resized to 50x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=307 // batch_size,
    epochs=12,
    validation_data=validation_generator,
    validation_steps=88 // batch_size)
# model.save_weights('first_try.h5')
