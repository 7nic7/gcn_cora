import tensorflow as tf
import numpy as np

class GCN(object):

    def __init__(self, n, m, c, lr, epoch, support):
        self.init_lr = lr
        self.epoch = epoch
        self.n = n
        self.m = m
        self.c = c
        self.support = support

    def build(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[self.n, self.m])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[self.n])
        self.output = tf.placeholder(dtype=tf.float32, shape=[self.n, self.c])
        self.support = tf.constant(self.support, dtype=tf.float32)
        self.lr = tf.placeholder(dtype=tf.float32)
        self.keep_prop = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('gcn1'):
            self.dropout1 = tf.nn.dropout(self.input, self.keep_prop)
            weights1 = tf.Variable(initial_value=tf.random_uniform([self.m, 16],
                                                                   minval=-np.sqrt(6.0/(self.m+16)),
                                                                   maxval=np.sqrt(6.0/(self.m+16))),
                                   dtype=tf.float32)
            bias1 = tf.Variable(initial_value=tf.zeros([1]), dtype=tf.float32)
            self.gcn1 = tf.nn.relu(tf.matmul(self.support, tf.matmul(self.dropout1, weights1)) + bias1)

        with tf.name_scope('gcn2'):
            self.dropout2 = tf.nn.dropout(self.gcn1, self.keep_prop)
            weights2 = tf.Variable(initial_value=tf.random_uniform([16, self.c],
                                                                   minval=-np.sqrt(6.0/(self.c+16)),
                                                                   maxval=np.sqrt(6.0/(self.c+16))),
                                   dtype=tf.float32)
            bias2 = tf.Variable(initial_value=tf.zeros([1]), dtype=tf.float32)
            self.logits = tf.matmul(self.support, tf.matmul(self.dropout2, weights2)) + bias2
            self.pred = tf.nn.softmax(self.logits)

        with tf.name_scope('optimize'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.output, self.logits)
            self.loss = tf.reduce_mean(self.loss * self.mask / tf.reduce_mean(self.mask))
            self.loss += tf.nn.l2_loss(weights1)*5e-4
            self.op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.pred, axis=1), tf.argmax(self.output, axis=1)), tf.float32) \
                * self.mask / tf.reduce_mean(self.mask)
            )

    def train(self, x, y, train_mask, val_mask, test_mask, sess):
        feed_dict = {
            self.input: x,
            self.output: y,
            self.mask: train_mask,
            self.lr: self.init_lr,
            self.keep_prop: 0.5
        }
        sess.run(tf.global_variables_initializer())
        val_losses = []
        for e_ in range(1, 1+self.epoch):
            _, train_loss, train_acc = sess.run([self.op, self.loss, self.acc], feed_dict=feed_dict)
            print('epoch %s: loss %.4f | accuracy %.4f' % (e_, train_loss, train_acc))
            val_loss, val_acc = self.eval(x, y, val_mask, sess)
            val_losses.append(train_loss)

            # if e_ % 250 == 0:
            #     self.init_lr *= 0.1

            if e_ % 10 == 0:
                print('evaluate(epoch %s): loss %.4f | accuracy %.4f' % (e_, val_loss, val_acc))

            if e_ > 10 and val_losses[-1] > np.mean(val_losses[-11:-1]):
                print('Early stopping...')
                test_loss, test_acc = self.eval(x, y, test_mask, sess)
                print('Testing data: loss %.4f | accuracy %.4f' % (test_loss, test_acc))
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

    # def predict(self, x, sess):
    #     feed_dict = {
    #         self.input: x,
    #         self.keep_prop: 1
    #     }
    #     pred = sess.run(self.logits, feed_dict=feed_dict)
    #     return pred

    def hidden(self, x, sess):
        feed_dict = {
            self.input: x,
            self.keep_prop: 1
        }
        a = sess.run(self.gcn1, feed_dict=feed_dict)
        return a
