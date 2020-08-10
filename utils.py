import itertools
import numpy as np
from scipy import sparse


def adj_matrix(graph):
    nodes = []
    for src, v in graph.items():
        nodes.extend([[src, v_] for v_ in v])
        nodes.extend([[v_, src] for v_ in v])
    nodes = [k for k, _ in itertools.groupby(sorted(nodes))]
    nodes = np.array(nodes)
    return sparse.coo_matrix((np.ones(nodes.shape[0]), (nodes[:, 0], nodes[:, 1])),
                             (len(graph), len(graph)))


def norm_adj_matrix(matrix):
    """(D+I)^(-0.5)*(A+I)*(D+I)^(-0.5)"""
    matrix += sparse.eye(matrix.shape[0])
    degree = np.array(matrix.sum(axis=1))
    d_sqrt = sparse.diags(np.power(degree, -0.5).flatten())
    return d_sqrt.dot(matrix).dot(d_sqrt).toarray()


def norm_x(x):
    return np.diag(np.power(x.sum(axis=1), -1).flatten()).dot(x)
