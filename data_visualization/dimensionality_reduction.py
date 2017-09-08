from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold

def plot_embedding_2d(X, Y, title=None):
    # normalize to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], label=np.str(Y[i]),
                   color=plt.cm.Set1(Y[i] / 10.))

    if title is not None:
        plt.title(title)

def plot_embedding_3d(X, Y, title=None):
    # normalize to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=40,
                   color=plt.cm.Set1(Y[i]))

    if title is not None:
        plt.title(title)

def draw_tsne2d(X, Y, fig_path, title='t-SNE'):
    # t-SNE
    tsne = manifold.TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding_2d(X_tsne, Y, title=title)
    plt.savefig(fig_path, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.1, bbox_inches='tight')

def draw_tsne3d(X, Y, fig_path, title='t-SNE'):
    # t-SNE
    tsne = manifold.TSNE(n_components=3, random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding_3d(X_tsne, Y, title=title)
    plt.savefig(fig_path, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.1, bbox_inches='tight')