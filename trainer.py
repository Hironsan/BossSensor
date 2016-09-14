# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from boss_input import extract_data



class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def read(self, img_rows=32, img_cols=32, img_channels=3, nb_classes=2):
        with tf.Session() as sess:
            images, labels = extract_data('./data/', sess)
        labels = np.reshape(labels, [-1])
        # numpy.reshape
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=random.randint(0, 100))
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


class Model(object):

    FILE_PATH = './store/model.h5'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=40, data_augmentation=True):
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.X_test, dataset.Y_test),
                           shuffle=True)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=0,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)                  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataset.X_train)

            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.X_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.X_test, dataset.Y_test))

    def save(self, file_path=FILE_PATH):
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        self.model = load_model(file_path)

    def predict(self, image):
        result = self.model.predict_proba(self, image, batch_size=32, verbose=1)
        print(result)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save()
    # model.load()