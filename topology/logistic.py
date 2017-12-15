# encoding=utf-8
import tensorflow as tf
import numpy as np

# master = data_input.data_master()


time_steps = 12
channel_size = 3
embedding_size = 64
embedding_fn_size = 256
filter_num = 64

# fixed size 3
filter_sizes = [1, 3, 5]
threshold = 0.6


class LogisticModel(object):
    def __init__(self, init_learning_rate, decay_steps, decay_rate):
        global_step = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, 0.96, staircase=True)

        self.e = tf.placeholder(tf.float32, [None, embedding_size])
        self.y = tf.placeholder(tf.int32, [None, 1])

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("FN_Part"):
            output_fn = tf.layers.dense(self.e, embedding_fn_size, activation=tf.nn.relu,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        with tf.name_scope("Output_Part"):
            weight_last = tf.Variable(
                tf.truncated_normal([embedding_fn_size, 1]) * np.sqrt(2. / (3 * filter_num)))
            bias_last = tf.Variable(tf.truncated_normal([1], stddev=0.1))
            output_fn = tf.nn.dropout(output_fn, self.dropout_keep_prob)
            logist = tf.matmul(output_fn, weight_last) + bias_last

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=logist))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                          global_step=global_step)

            self.logits_pred = tf.nn.sigmoid(logist)

            ones = tf.ones_like(self.y)
            zeros = tf.zeros_like(self.y)
            self.prediction = tf.cast(tf.where(tf.greater(self.logits_pred, threshold), ones, zeros), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y), tf.float32))
