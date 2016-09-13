# -*- coding: utf-8 -*-
import random

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.cross_validation import train_test_split

from boss_input import extract_data



def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and
    # height final dimension being the number of color channels.
    X = tf.reshape(X, [-1, 32, 32, 3])

    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    # second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = learn.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    # densely connected layer with 1024 neurons.
    h_fc1 = learn.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)

    return learn.models.logistic_regression(h_fc1, y)


if __name__ == '__main__':
    path = './data'
    """
    with tf.Session() as sess:
        images, labels = extract_data(path, sess)
        labels = tf.reshape(labels, [-1])
        labels = sess.run(labels)
        # numpy.reshape
        train_x, test_x, train_t, test_t = train_test_split(images, labels, test_size=0.2,
                                                            random_state=random.randint(0, 100))

    # Training and predicting.
    classifier = learn.TensorFlowEstimator(
        model_fn=conv_model, n_classes=2, batch_size=40, steps=1000,
        learning_rate=0.001)

    classifier.fit(train_x, train_t)
    classifier.save('./store/')

    score = metrics.accuracy_score(test_t, classifier.predict(test_x))
    print('Accuracy: {0:f}'.format(score))
    """
    with tf.Session() as sess:
        import boss_input
        image = boss_input.read_image_('./data/boss/1.jpg', sess)
        import numpy as np
        image = np.reshape(image, [-1])

        classifier = learn.Estimator(model_fn=conv_model, model_dir='./store/')
        #images = np.array([image])
        classifier.predict(image)
