# -*- coding: utf-8 -*-
import tensorflow as tf
import os


images = []
labels = []
def traverse_dir(path, sess):
    for file_or_dir in os.listdir(path):
        if len(images) > 100:
            return images, labels
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


IMAGE_SIZE = 96
def read_image_(file_path, sess):
    jpeg_r = tf.read_file(file_path)
    image = tf.image.decode_jpeg(jpeg_r, channels=3)
    image.set_shape(sess.run(image).shape)
    h, w, _ = image.get_shape()
    longest_edge = int(max(h, w))
    image = tf.image.resize_image_with_crop_or_pad(image, longest_edge, longest_edge)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    return image


if __name__ == '__main__':
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess)
