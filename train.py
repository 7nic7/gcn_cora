import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf

from gcn import GCN
from load_cora import Cora
from utils import adj_matrix, norm_adj_matrix, norm_x


if __name__ == '__main__':
    cora = Cora()
    x, y, tx, ty, allx, ally, test_index, graph = cora.load_data('cora/')
    train_mask, val_mask, test_mask = cora.split_data(allx, ally, tx, ty, y, test_index)
    adj_mat = adj_matrix(graph)
    support = norm_adj_matrix(adj_mat)
    cora.x = norm_x(cora.x)

    model = GCN(
        n=cora.x.shape[0],
        m=cora.x.shape[1],
        c=cora.y.shape[1],
        lr=0.01,
        epoch=200,
        support=support
    )

    model.build()
    with tf.Session() as sess:
        model.train(cora.x, cora.y, train_mask, val_mask, test_mask, sess)

        a = model.hidden(cora.x, sess)
        print('visualize the outputs of last layer')

    tsne = TSNE()
    embedding = tsne.fit_transform(a)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=np.argmax(cora.y, axis=1))
    plt.show()
