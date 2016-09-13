# -*- coding: utf-8 -*-
import os.path

import tensorflow as tf
import numpy as np

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
tf.app.flags.DEFINE_string('checkpoint_dir', './store',
                           """Directory where to read model checkpoints.""")

def train():
    """
    Train Boss Face for a number of steps.
    """

    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(tf.float32, shape=[None, 3072])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)

        output = boss_model.inference(x, keep_prob)
        loss = boss_model.loss(output, y_)
        training_op = boss_model.training(loss)

        saver = tf.train.Saver(tf.all_variables())

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


def predict(image=None):
    with tf.Session() as sess:
        #image = boss_input.conv_image(image, sess)
        image = boss_input.read_image_('./data/boss/1.jpg', sess)
        #image = boss_input.read_image_('./data/other/Abdel_Nasser_Assidi_0002.jpg', sess)
        global_step = tf.Variable(0, trainable=False)
        image = np.reshape(image, [-1])
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        keep_prob = tf.placeholder(tf.float32)
        #logits = boss_model.inference(image, keep_prob)

        x = tf.placeholder(tf.float32, shape=[None, 3072])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)

        output = boss_model.inference(x, keep_prob)
        loss = boss_model.loss(output, y_)
        training_op = boss_model.training(loss)

        saver = tf.train.Saver()

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

        feed_dict = {x: np.array([image], dtype=np.float32), keep_prob: 1.0}
        classification = sess.run(output, feed_dict=feed_dict)
        print(classification)
        label = np.argmax(classification[0])
        if label == 1:
            print('Boss')
        else:
            print('Other')



class FacePredictor(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 3072])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32)

        self.output = boss_model.inference(self.x, self.keep_prob)
        self.loss = boss_model.loss(self.output, self.y_)
        self.training_op = boss_model.training(self.loss)

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        with tf.Session() as sess:
            self.sess = sess
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

    def predict(self, image):
        feed_dict = {self.x: np.array([image], dtype=np.float32), self.keep_prob: 1.0}
        classification = self.sess.run(self.output, feed_dict=feed_dict)
        print(classification)


def main(argv=None):
    #if tf.gfile.Exists(FLAGS.train_dir):
        #tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    #train()
    predict()


if __name__ == '__main__':
    tf.app.run()
