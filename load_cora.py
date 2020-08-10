import os
import numpy as np
import pickle
from scipy import sparse


class Cora(object):

    def __init__(self):
        pass

    def load_data(self, dir):
        test_index = np.genfromtxt(os.path.join(dir, 'ind.cora.test.index'), dtype='int64')
        x = pickle.load(open(os.path.join(dir, 'ind.cora.x'), 'rb'), encoding='latin1')
        y = pickle.load(open(os.path.join(dir, 'ind.cora.y'), 'rb'), encoding='latin1')
        tx = pickle.load(open(os.path.join(dir, 'ind.cora.tx'), 'rb'), encoding='latin1')
        ty = pickle.load(open(os.path.join(dir, 'ind.cora.ty'), 'rb'), encoding='latin1')
        allx = pickle.load(open(os.path.join(dir, 'ind.cora.allx'), 'rb'), encoding='latin1')
        ally = pickle.load(open(os.path.join(dir, 'ind.cora.ally'), 'rb'), encoding='latin1')
        graph = pickle.load(open(os.path.join(dir, 'ind.cora.graph'), 'rb'), encoding='latin1')
        return x, y, tx, ty, allx, ally, test_index, graph

    def split_data(self, allx, ally, tx, ty, y, test_index):
        self.x = sparse.vstack([allx, tx]).tolil().toarray()
        self.y = sparse.vstack([ally, ty]).toarray()
        sort_index = sorted(test_index)
        self.x[test_index] = self.x[sort_index]
        self.y[test_index] = self.y[sort_index]

        n = self.x.shape[0]  # the number of nodes of graph
        train_mask = np.zeros(n); train_mask[:y.shape[0]] = 1
        val_mask = np.zeros(n); val_mask[y.shape[0]:(y.shape[0]+500)] = 1
        test_mask = np.zeros(n); test_mask[test_index] = 1
        return train_mask, val_mask, test_mask


