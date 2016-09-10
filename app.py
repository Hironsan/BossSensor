import tensorflow as tf
import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(input_placeholder, keep_prob):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(input_placeholder, [-1, 32, 32, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


def loss(output, supervisor_labels_placeholder):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(supervisor_labels_placeholder * tf.log(output), reduction_indices=[1]))
    return cross_entropy


def training(loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step



x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = inference(x, keep_prob)
    loss = loss(output, y_)
    training_op = training(loss)

    init = tf.initialize_all_variables()
    sess.run(init)
    dataset = input_data.read_data_sets('data', sess)

    for step in range(1000):
        batch = dataset.train.next_batch(40)
        sess.run(training_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if step % 100 == 0:
            print('step: {0}, loss: {1}'.format(step, sess.run(loss, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})))

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy %g' % accuracy.eval(feed_dict={x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0}))