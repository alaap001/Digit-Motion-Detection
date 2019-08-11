# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:58:31 2019

@author: Alaap
"""

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense,Conv2D,BatchNormalization,Dropout,Flatten,MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
#model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
epochs = 45

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size = 0.1)
X_train2 = X_train2.reshape(-1, 28, 28, 1)
X_val2 = X_val2.reshape(-1, 28, 28, 1)

from keras.utils import np_utils

Y_train2 = np_utils.to_categorical(Y_train2)
Y_val2 = np_utils.to_categorical(Y_val2)
Y_val2.shape, Y_train2.shape

history = model.fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  validation_data = (X_val2,Y_val2), callbacks=[annealer])
print("Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        epochs,max(history['acc']),max(history['val_acc'])))
model.evaluate(x_test, y_test)