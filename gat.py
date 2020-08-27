import tensorflow as tf
from utils import glorot


class GAT(object):

    def __init__(self, c, m, epoch, adj_matrix, head_num, hidden_dim, lr):
        self.m = m
        self.adj_matrix = adj_matrix
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.c = c
        self.epoch = epoch
        self.init_lr = lr

    def att_head(self, inputs, output_dim):
        """
          e_ij
        = a'[W*h_i || W*h_j]
        = [a_1', a_2'] [W*h_i || W*h_j]
        = a_1'*W*h_i + a_2'*W*h_j

        next hidden layer = \sum_{j \in N(i)} e_ij*W*h_j

        e = a1*W*h1+a2*W*h1, a1*W*h1+a2*W*h2, ..., a1*W*h1+a2*W*hN
            a1*W*h2+a2*W*h1, a1*W*h2+a2*W*h2, ..., a1*W*h2+a2*W*hN
            ...
        """
        input_dim = inputs.shape[1] if not isinstance(inputs, tf.Tensor) else inputs.get_shape()[1].value
        weights = tf.Variable(glorot(input_dim, output_dim), dtype=tf.float32)  # F*F'
        inputs_prime = tf.matmul(inputs, weights)  # N*F'  [W*h_1, W*h_2, ..., W*h_N]
        a = [tf.Variable(glorot(output_dim, 1), dtype=tf.float32), tf.Variable(glorot(output_dim, 1), dtype=tf.float32)]
        f_1 = tf.matmul(inputs_prime, a[0])  # N*1  [a_1*W*h_1, a_1*W*h_2, ..., a_1*W*h_N]
        f_2 = tf.matmul(inputs_prime, a[1])  # N*1  [a_2*W*h_1, a_2*W*h_2, ..., a_2*W*h_N]
        e = tf.transpose(f_2, [1, 0]) + f_1  # N*N broadcast
        e_hat = tf.nn.leaky_relu(e)
        e_hat += -10e8*(1-self.adj_matrix)  # ignore the nodes which are not neighbors of node i
        alpha = tf.nn.softmax(e_hat, axis=1)  # N*N
        output = tf.nn.leaky_relu(tf.matmul(alpha, inputs_prime))  # N*F'
        return output

    def att_multi_head(self, inputs, output_dim, head_num):
        outputs = []
        for k in range(head_num):
            output = self.att_head(inputs, output_dim)
            output = tf.expand_dims(output, axis=0)
            outputs.append(output)
        h = tf.reduce_mean(tf.concat(outputs, axis=0), axis=0)  # N*F'
        return h

    def build(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.m])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.c])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None])
        self.lr = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('hidden_layer'):
            self.h = tf.nn.leaky_relu(self.att_multi_head(self.inputs, self.hidden_dim, self.head_num))

        with tf.name_scope('output'):
            logits = self.att_multi_head(self.h, self.c, self.head_num)
            self.prediction = tf.nn.softmax(logits, axis=1)

        with tf.name_scope('optimizer'):
            self.loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(self.labels, logits),
                                                    self.mask) / tf.reduce_mean(self.mask))
            self.accuracy = tf.reduce_mean(tf.multiply(
                tf.cast(tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.labels, axis=1)), dtype=tf.float32),
                self.mask
            ) / tf.reduce_mean(self.mask))
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, x, y, train_mask, val_mask, sess):
        sess.run(tf.global_variables_initializer())
        val_loss_list = []
        for e in range(self.epoch):
            feed_dict = {
                self.inputs: x,
                self.labels: y,
                self.mask: train_mask,
                self.lr: self.init_lr
            }
            _, train_loss, train_acc = sess.run([self.optim, self.loss, self.accuracy], feed_dict=feed_dict)
            print('epoch %s: training loss %.4f | training accuracy %.4f' % (e, train_loss, train_acc))
            val_loss, val_acc = self.eval(x, y, val_mask, sess)
            val_loss_list.append(val_loss)

            if (e+1) % 10 == 0:
                print('Evaluation: validation loss %.4f | validation accuracy %.4f (epoch %s)' % (val_loss, val_acc, e))

            if (e+1) % 10 == 50:
                self.init_lr *= 0.1

            if e > 10 and sum(val_loss_list[(-10-1):(-1)]) / 10 < val_loss_list[-1]:
                val_loss, val_acc = self.eval(x, y, val_mask, sess)
                print('Early Stopping: validation loss %.4f | validation accuracy %.4f (epoch %s)' \
                      % (val_loss, val_acc, e))
                break

    def eval(self, x, y, mask, sess):
        feed_dict = {
            self.inputs: x,
            self.labels: y,
            self.mask: mask,
        }
        loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, acc

