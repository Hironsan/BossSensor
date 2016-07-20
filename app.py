from pgm_reader import read_data_sets
import tensorflow as tf


mnist = read_data_sets()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 644])
y_ = tf.placeholder(tf.float32, shape=[None, 20])
W = tf.Variable(tf.zeros([644, 20]))
b = tf.Variable(tf.zeros([20]))
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(10)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))