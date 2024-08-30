import random
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# import tensorflow._api.v2.compat.v1 as tf
from utils import *
from time import time
import argparse

import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans

# Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import scG
from evaluation import eva2
# from graph_function import *
from scipy import sparse as sp
import h5py
import numpy as np
import tensorflow as tf


def get_args(dataset_path, model_pth, seed=0, pretrain_epochs=800 , pretrain_alpha=0.1,
             maxiter=300, train_alpha=0.1, n_pairs=0.1):

    parser = argparse.ArgumentParser(description='Parser for scG')
    parser.add_argument("--seed", default=seed, type=int)

    parser.add_argument('--dataset_path', default=dataset_path, type=str,
                        help='path to dataset (adata)')
    # Pretrain
    parser.add_argument("--pretrain_epochs", default=pretrain_epochs, type=int)  # Pre-trained epochs
    parser.add_argument("--pretrain_alpha", default=pretrain_alpha, type=float)
    parser.add_argument("--model_pth", default=model_pth, type=str)
    # Train
    parser.add_argument("--maxiter", default=maxiter, type=int)  # Trained epochs
    parser.add_argument("--train_alpha", default=train_alpha, type=float)
    parser.add_argument("--n_pairs", default=n_pairs, type=float)  # Number of triplets

    args = parser.parse_args()
    return args


def computeCentroids(X, labels):
    num_clusters = np.max(labels) + 1
    centroids = np.zeros((num_clusters, X.shape[1]))
    for k in range(num_clusters):
        centroids[k, :] = np.mean(X[labels == k, :], axis=0)
    return centroids

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def degree_power(A, k):   # A is the adjacency matrix of the input
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D



if __name__ == "__main__":

    dataset_path = "data/LPS/lps_int2_2000.h5"
    model_pth = "model/LPS/"
    args = get_args(dataset_path, model_pth, seed = 0)
    print(args)

    # seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)


    with h5py.File(dataset_path, 'r') as file:
        X = file['X'][:]  # normalized data
        y = file['Y'][:]  # labels
        hvg = file['highly_variable'][:]  # highly variable gene
        size_factor = file['size_factors'][:]
        raw = file['raw'][:]



    cluster_number = int(max(y) - min(y) + 1)
    print("cluster_number: ", cluster_number)

    count = X
    print(count.shape)
    print(count.shape)
    raw_count = raw
    print(raw_count.shape)


    # adjacency graph
    file_path = 'graph/LPS/lps_int2_2global_ncos.txt'
    edges = np.loadtxt(file_path, dtype=int)
    num_nodes = np.max(edges) + 1

    # Initialize the adjacency matrix
    G_adj = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge in edges:
        G_adj[edge[0], edge[1]] = 1
        G_adj[edge[1], edge[0]] = 1

    np.fill_diagonal(G_adj, 1)
    G_adj_n = norm_adj(G_adj)

    # adjacency graph
    file_path = 'graph/LPS/lps_int2_2local_ncos.txt'
    edges = np.loadtxt(file_path, dtype=int)
    num_nodes = np.max(edges) + 1

    L_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        L_adj[edge[0], edge[1]] = 1
        L_adj[edge[1], edge[0]] = 1

    np.fill_diagonal(L_adj, 1)
    L_adj_n = norm_adj(L_adj)


    # scG model instance
    model = scG(raw_count, count, size_factor, model_pth, G_adj, G_adj_n, L_adj, L_adj_n)

    # Pre-training
    t0 = time()
    model.pre_train(epochs=args.pretrain_epochs)
    print("Pretrain end!")
    t1 = time()
    print('Pretrain: run time is {:.2f} '.format(t1 - t0), 'seconds')


    # The initial center of mass is obtained from the pre-training result
    model.load_model("pretrain")
    X_pretrain = model.embedding(count)
    pca = PCA(n_components=15)
    countp = pca.fit_transform(count)
    labels = KMeans(n_clusters=cluster_number).fit(countp).labels_
    centers = computeCentroids(X_pretrain, labels)


    # clustering training
    t2 = time()
    model.train(y, epochs=args.maxiter, centers=centers)
    print("Train end!")
    t3 = time()
    print('Train: run time is {:.2f} '.format(t3 - t2), 'seconds')
    print('Pre+Train: run time is {:.2f} '.format(t3 - t0), 'seconds')


    # Evaluating clustering results
    if y is not None:
        model.load_model("train")
        print("load_model success")
        X_train, y_pred = model.get_cluster()

        # 计算 ACC、NMI、ARI
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)

        print('ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

        # F1 score, Silhouette
        f1, sc = eva2(X_train, y, y_pred)
        print('F1= %.4f, SC= %.4f' % (f1, sc))




