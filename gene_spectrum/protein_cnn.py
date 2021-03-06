# encoding=utf-8
import tensorflow as tf
import numpy as np

# master = data_input.data_master()


time_steps = 12
embedding_size = 3

filter_num = 64

# fixed size 3
filter_sizes = [1, 3, 5]
threshold = 0.6


class CnnModel(object):
    def __init__(self, init_learning_rate, decay_steps, decay_rate):
        weights = {
            'wc1': tf.Variable(tf.truncated_normal([filter_sizes[0], embedding_size, filter_num], stddev=0.1)),
            'wc2': tf.Variable(
                tf.truncated_normal([filter_sizes[1], embedding_size, filter_num], stddev=0.1)),
            'wc3': tf.Variable(
                tf.truncated_normal([filter_sizes[2], embedding_size, filter_num], stddev=0.1))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc3': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1))
        }
        global_step = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)

        # define placehold
        self.x = tf.placeholder(tf.float32, [None, embedding_size, time_steps])
        x_emb = tf.transpose(self.x, [0, 2, 1])  # [None,time_steps,embedding_size]

        self.y = tf.placeholder(tf.int32, [None, 1])

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        x_convs = self.multi_conv(x_emb, weights, biases)
        print('after multiply convolutions: ', x_convs)
        x_convs = tf.reshape(x_convs, [-1, 3 * filter_num])

        ones = tf.ones_like(self.y)
        zeros = tf.zeros_like(self.y)

        x_convs = tf.nn.dropout(x_convs, self.dropout_keep_prob)

        with tf.name_scope("CNN_Part"):
            weight_cnn = tf.Variable(
                tf.truncated_normal([3 * filter_num, 1]) * np.sqrt(2. / (3 * filter_num)))
            biase_cnn = tf.Variable(tf.truncated_normal([1], stddev=0.1))
            logits_cnn = tf.matmul(x_convs, weight_cnn) + biase_cnn
            self.loss_cnn = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=logits_cnn))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn,
                                                                                              global_step=global_step)
            self.logits_pred = tf.nn.sigmoid(logits_cnn)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.logits_pred, threshold), ones, zeros), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction_cnn, self.y), tf.float32))

    def conv1d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, time_steps, embedding_size])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        # shape=(n,time_steps,filter_num)
        h = tf.nn.relu(x)

        print('conv size:', h.get_shape().as_list())

        pooled = tf.reduce_max(h, axis=1)
        print('pooled size:', pooled.get_shape().as_list())
        return pooled

    def multi_conv(self, x, weights, biases):
        # Convolution Layer
        conv1 = self.conv1d(x, weights['wc1'], biases['bc1'])
        conv2 = self.conv1d(x, weights['wc2'], biases['bc2'])
        conv3 = self.conv1d(x, weights['wc3'], biases['bc3'])
        #  n*time_steps*(3*filter_num)
        convs = tf.concat([conv1, conv2, conv3], 1)
        return convs
