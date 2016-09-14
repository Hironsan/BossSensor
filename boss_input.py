# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
import cv2


def resize_with_pad(image, height, width):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0,0,0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


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


def extract_data(path, sess):
    images, labels = traverse_dir(path, sess)
    #images = np.array([np.reshape(image, -1) for image in images])
    images = np.array(images)
    dic = dict([(label, i) for i, label in enumerate(set(labels))])
    labels = np.array([dic[label] for label in labels])
    return images, labels
