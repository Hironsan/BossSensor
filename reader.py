# -*- coding: utf-8 -*-
import tensorflow as tf

import os
import cv2


def read_image(file_path):
    image = cv2.imread(file_path, 0)  # grayscale
    print(image)
    print(image.shape)
    return image

images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image_(abs_path)
                images.append(image)
                labels.append(path)
    return images, labels



#csv_name = 'path/to/filelist.csv'
#fname_queue = tf.train.string_input_producer([csv_name])
#reader = tf.TextLineReader()
#key, val = reader.read(fname_queue)
#fname, label = tf.decode_csv(val, [["aa"], [1]])
def read_image_(file_path):
    jpeg_r = tf.read_file(file_path)
    image = tf.image.decode_jpeg(jpeg_r, channels=3)
    IMAGE_SIZE = 32
    image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)
    return image

traverse_dir('./data')
"""
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess)
x = sess.run(image)
print(x)
print(x.shape)
"""