import numpy as np
import tensorflow as tf

from utils import glorot


class Aggregate(object):

    def __init__(self, output_dim=None):
        self.output_dim = output_dim

    def sample(self, adj_matrix, sample_num):
        sample_matrix = np.zeros_like(adj_matrix)
        for i in range(adj_matrix.shape[0]):
            index = np.arange(adj_matrix.shape[1])
            neigh_ids = index[adj_matrix[i, :] == 1]
            if len(neigh_ids) < sample_num:
                sample_neighs = np.random.choice(neigh_ids, sample_num, replace=True)
            else:
                sample_neighs = np.random.choice(neigh_ids, sample_num, replace=False)
            for j in sample_neighs:
                sample_matrix[i, j] += 1
        return sample_matrix

    def mean_agg(self, feats, adj_matrix, sample_num):
        matmul_matrix = self.sample(adj_matrix, sample_num)
        matmul_matrix /= sample_num
        matmul_matrix = tf.constant(matmul_matrix, dtype=tf.float32)
        output = tf.matmul(matmul_matrix, feats)
        return output  # shape: n*feat_dim

    def maxpool_agg(self, feats, adj_matrix, sample_num):
        sample_matrix = self.sample(adj_matrix, sample_num)
        input_dim = feats.get_shape()[1].value if isinstance(feats, tf.Tensor) else feats.shape[1]
        weights = tf.Variable(glorot(input_dim, self.output_dim), dtype=tf.float32)
        bias = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        matmul_matrix = []
        for i in range(sample_matrix.shape[0]):
            neighs = sample_matrix[i, :].astype(np.int)
            sample_matmul_matrix = np.zeros([sample_num, sample_matrix.shape[1]])
            sample_matmul_matrix[np.arange(sample_num), np.repeat(np.where(neighs!=0)[0], neighs[neighs!=0])] = 1
            matmul_matrix.append(sample_matmul_matrix)
        matmul_matrix = tf.concat(matmul_matrix, axis=0)   # (n*sample_size)*n
        neighs_embed = tf.matmul(tf.cast(matmul_matrix, dtype=tf.float32), feats)    # (n*sample_size)*m
        fc = tf.nn.relu(tf.matmul(neighs_embed, weights) + bias)   # shape: (n*sample_num)*output_dim
        output = tf.reduce_max(tf.reshape(fc, [-1, sample_num, self.output_dim]), axis=1)
        return output  # shape: n*output_dim


class Concat(object):

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def concat(self, neighs_embed, node_embed=None, activation=None):
        if node_embed is not None:
            input_ = tf.concat([neighs_embed, node_embed], axis=1)  # shape:n*(neigh_dim+feat_dim)
        else:
            input_ = neighs_embed
        self.weights = tf.Variable(glorot(input_.get_shape()[1].value, self.output_dim), dtype=tf.float32)
        self.b = tf.Variable(tf.random_uniform([1], minval=0, maxval=1), dtype=tf.float32)
        if activation is not None:
            output_ = activation(tf.matmul(tf.cast(input_, dtype=tf.float32), self.weights) + self.b)
        else:
            output_ = tf.matmul(tf.cast(input_, dtype=tf.float32), self.weights) + self.b
        return output_


class GraphSage(object):
    """
    GraphSAGE updates gradients by using all of the data instead of using batch samples in paper.
    """
    def __init__(self, adj_matrix, feats, sample_nums, m, c, lr, output_dims, epoch, aggregator):
        self.feats = feats
        self.adj_matrix = adj_matrix
        self.epoch = epoch
        self.sample_nums = sample_nums
        self.output_dims = output_dims
        self.lr = lr
        self.m = m
        self.c = c
        self.aggregator = aggregator

    def embed_lookup(self, nodes):
        feats_tf = tf.constant(self.feats, dtype=tf.float32)
        embeds = tf.nn.embedding_lookup(feats_tf, tf.reshape(nodes, [1, -1]))
        return embeds

    def build(self):
        self.input_ = tf.placeholder(dtype=tf.int32, shape=[None])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.c])
        agg1 = self.aggregator(self.feats, self.adj_matrix, self.sample_nums[0])
        self.concat1 = Concat(self.output_dims[0]).concat(agg1, self.feats, tf.nn.relu)
        agg2 = self.aggregator(self.concat1, self.adj_matrix, self.sample_nums[1])
        concat2 = Concat(self.c).concat(agg2, self.concat1)
        logits = tf.nn.embedding_lookup(concat2, self.input_)

        self.prediction = tf.nn.softmax(logits, axis=1)

        with tf.name_scope('optimizer'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.labels, logits))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prediction, axis=1),
                                                            tf.argmax(self.labels, axis=1)), dtype=tf.float32))
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, train_x, train_y, val_x, val_y, sess):
        sess.run(tf.global_variables_initializer())
        for e_ in range(self.epoch):
            feed_dict = {
                self.input_: train_x,
                self.labels: train_y
            }
            _, train_loss, train_acc = sess.run([self.optim, self.loss, self.accuracy], feed_dict=feed_dict)
            print('training loss %.4f | training accuracy %.4f' % (train_loss, train_acc))
            if e_ % 10 == 0:
                val_loss, val_acc = self.eval(val_x, val_y, sess)
                print('validation loss %.4f | validation accuracy %.4f' % (val_loss, val_acc))

    def eval(self, x, y, sess):
        feed_dict = {
            self.input_: x,
            self.labels: y
        }
        loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, acc
