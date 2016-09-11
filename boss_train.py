# -*- coding: utf-8 -*-
import os.path

import tensorflow as tf

import boss_input
import boss_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './store',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    """
    Train Boss Face for a number of steps.
    """

    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(tf.float32, shape=[None, 3072])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)

        saver = tf.train.Saver(tf.all_variables())

        output = boss_model.inference(x, keep_prob)
        loss = boss_model.loss(output, y_)
        training_op = boss_model.training(loss)

        init = tf.initialize_all_variables()
        sess.run(init)
        dataset = boss_input.read_data_sets('data', sess)

        for step in range(1000):
            batch = dataset.train.next_batch(40)
            sess.run(training_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if step % 100 == 0:
                print('step: {0}, loss: {1}'.format(step, sess.run(loss, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('test accuracy %g' % accuracy.eval(feed_dict={x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0}))


def predict(saver, top_k_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        predictions = sess.run([top_k_op])

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
