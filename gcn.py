import tensorflow as tf
import numpy as np

from utils import glorot


class GCN(object):

    def __init__(self, n, m, c, lr, epoch, support, dropout, hidden_dim, weight_decay):
        self.init_lr = lr
        self.epoch = epoch
        self.n = n
        self.m = m
        self.c = c
        self.support = support
        self.dropout = dropout
        self.h = hidden_dim
        self.weight_decay = weight_decay

    def build(self, chebyshev=False):
        self.input = tf.placeholder(dtype=tf.float32, shape=[self.n, self.m])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[self.n])
        self.output = tf.placeholder(dtype=tf.float32, shape=[self.n, self.c])
        self.support = [tf.constant(support_, dtype=tf.float32) for support_ in self.support] if chebyshev else \
            tf.constant(self.support, dtype=tf.float32)
        self.lr = tf.placeholder(dtype=tf.float32)
        self.keep_prop = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('gcn1'):
            self.dropout1 = tf.nn.dropout(self.input, self.keep_prop)
            self.gcn1, weights1 = layer(self.support, self.dropout1, self.m, self.h, chebyshev)
            self.gcn1 = tf.nn.relu(self.gcn1)

        with tf.name_scope('gcn2'):
            self.dropout2 = tf.nn.dropout(self.gcn1, self.keep_prop)
            self.logits, _ = layer(self.support, self.dropout2, self.h, self.c, chebyshev)
            self.pred = tf.nn.softmax(self.logits, axis=1)

        with tf.name_scope('optimize'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.output, self.logits)
            self.loss = tf.reduce_mean(self.loss * self.mask / tf.reduce_mean(self.mask))
            self.loss += tf.nn.l2_loss(weights1) * self.weight_decay
            self.op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.pred, axis=1), tf.argmax(self.output, axis=1)), tf.float32) \
                * self.mask / tf.reduce_mean(self.mask)
            )

    def train(self, x, y, train_mask, val_mask, sess):
        sess.run(tf.global_variables_initializer())
        val_losses = []
        for e_ in range(1, 1+self.epoch):
            feed_dict = {
                self.input: x,
                self.output: y,
                self.mask: train_mask,
                self.lr: self.init_lr,
                self.keep_prop: self.dropout
            }
            _, train_loss, train_acc = sess.run([self.op, self.loss, self.acc], feed_dict=feed_dict)
            print('epoch %s: loss %.4f | accuracy %.4f' % (e_, train_loss, train_acc))
            val_loss, val_acc = self.eval(x, y, val_mask, sess)
            val_losses.append(train_loss)

            if e_ % 200 == 0:
                self.init_lr *= 0.1

            if e_ % 10 == 0:
                print('evaluate(epoch %s): loss %.4f | accuracy %.4f' % (e_, val_loss, val_acc))

            if e_ > 10 and val_losses[-1] > np.mean(val_losses[-11:-1]):
                print('Early stopping...')
                break

    def eval(self, x, y, val_mask, sess):
        feed_dict = {
            self.input: x,
            self.output: y,
            self.mask: val_mask,
            self.keep_prop: 1
        }
        val_loss, val_acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)
        return val_loss, val_acc

    def predict(self, x, sess):
        feed_dict = {
            self.input: x,
            self.keep_prop: 1
        }
        pred = sess.run(self.pred, feed_dict=feed_dict)
        return pred


def layer(support, x, input_dim, output_dim, chebyshev=False):
    weights = tf.Variable(initial_value=glorot(input_dim, output_dim), dtype=tf.float32)
    bias = tf.Variable(initial_value=tf.zeros([1]), dtype=tf.float32)
    if chebyshev:
        theta = [tf.Variable(tf.random.normal([1]), dtype=tf.float32) for _ in range(len(support))]
        support = tf.add_n([support_ * theta_ for support_, theta_ in zip(support, theta)])
    return tf.matmul(support, tf.matmul(x, weights)) + bias, weights
