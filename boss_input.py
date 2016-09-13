# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base


images = []
labels = []
def traverse_dir(path, sess):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path, sess)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image_(abs_path, sess)
                images.append(image)
                labels.append(path)
    return images, labels


IMAGE_SIZE = 32
def read_image_(file_path, sess):
    jpeg_r = tf.read_file(file_path)
    image = tf.image.decode_jpeg(jpeg_r, channels=3)
    image.set_shape(sess.run(image).shape)
    h, w, _ = image.get_shape()
    longest_edge = int(max(h, w))
    image = tf.image.resize_image_with_crop_or_pad(image, longest_edge, longest_edge)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = sess.run(image)

    return image


def conv_image(image, sess):
    h, w, _ = image.shape
    longest_edge = int(max(h, w))
    image = tf.image.resize_image_with_crop_or_pad(image, longest_edge, longest_edge)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = sess.run(image)

    return image


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)
    labels_one_hot = np.zeros((num_labels, num_classes))
    label_list = list(set(labels_dense))
    for i, label in enumerate(labels_dense):
        labels_one_hot[i][label_list.index(label)] = 1
    return labels_one_hot


def extract_data(path, sess):
    images, labels = traverse_dir(path, sess)
    #images = np.array([np.reshape(image, -1) for image in images])
    images = np.array(images)
    dic = dict([(label, i) for i, label in enumerate(set(labels))])
    labels = np.array([dic[label] for label in labels])
    return images, labels
"""

def extract_data(path, sess):
    images, labels = traverse_dir(path, sess)
    images = np.array([np.reshape(image, -1) for image in images])
    labels_one_hot = dense_to_one_hot(labels, len(set(labels)))
    return images, labels_one_hot
"""

class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 reshape=True):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(path, sess, dtype=dtypes.float32, reshape=False):
    images, labels = extract_data(path, sess)
    for i in range(images.shape[0]):
        j = random.randint(i, images.shape[0]-1)
        images[i], images[j] = images[j], images[i]
        labels[i], labels[j] = labels[j], labels[i]

    num_images = images.shape[0]

    TRAIN_SIZE = int(num_images * 0.8)
    VALIDATION_SIZE = int(num_images * 0.1)

    train_images = images[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]
    validation_images = images[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
    validation_labels = labels[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
    test_images = images[TRAIN_SIZE+VALIDATION_SIZE:]
    test_labels = labels[TRAIN_SIZE+VALIDATION_SIZE:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)