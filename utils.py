import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import  TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
from random import randint
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def min_max_normalizer(m):
    print("normalizing")
    _max = m.max()
    _min = m.min()

    m = m-_min
    return m/(_max - _min)

# the diagonal matrix st f_ii = sum_j S_ij
def getF(S):
    values = []
    for row in S.numpy():
        values.append(row.sum())
    return np.diag(values)

# (1-lambda)normalized(S att)
def getS(features):
    print("got S")
    s_att = cosine_similarity(features)
    S = min_max_normalizer(s_att)
    return S

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

def dotsne(X, dim=2, ran=23):
    tsne = TSNE(n_components=dim, random_state=ran)
    Y_tsne = tsne.fit_transform(X)
    return Y_tsne

def scatter_single(X, class_idxs, legend=False, ran=True):
    X = np.array(X)

    fig, axes = plt.subplots()

    # plt.figure(figsize=(8, 3))
    classes = list(np.unique(class_idxs))
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(classes)))
    if ran:
        np.random.shuffle(colors)

    for i, cls in enumerate(classes):
        axes.plot(X[class_idxs == cls, 0], X[class_idxs == cls, 1], marker="o",
                        linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                        markeredgecolor='black', markeredgewidth=0.15)
    if legend:
        axes.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0, fontsize=10, markerscale=2, frameon=False,
                    ncol=1, handletextpad=0.1, columnspacing=0.5)

    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')

    plt.show()

    return axes

def myscatter(filename, X_orig, X_pretrain, X_train, class_idxs, legend=False, ran=True):
    X_orig = np.array(X_orig)
    X_pretrain = np.array(X_pretrain)
    X_train = np.array(X_train)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey='row')

    # plt.figure(figsize=(8, 3))
    classes = list(np.unique(class_idxs))
    markers = 'osD' * len(classes)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(classes)))
    if ran:
        np.random.shuffle(colors)

    for i, cls in enumerate(classes):
        axes[0].plot(X_orig[class_idxs == cls, 0], X_orig[class_idxs == cls, 1], marker="o",
                        linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                        markeredgecolor='black', markeredgewidth=0.15)
        axes[1].plot(X_pretrain[class_idxs == cls, 0], X_pretrain[class_idxs == cls, 1], marker="o",
                        linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                        markeredgecolor='black', markeredgewidth=0.15)
        axes[2].plot(X_train[class_idxs == cls, 0], X_train[class_idxs == cls, 1], marker="o",
                        linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                        markeredgecolor='black', markeredgewidth=0.15)
    if legend:
        axes[2].legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0, fontsize=10, markerscale=2, frameon=False,
                       ncol=1, handletextpad=0.1, columnspacing=0.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename)

    return axes

def scGAE_scatter(filename, Y, class_idxs, legend=True, ran=False):
    Y = np.array(Y)
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    classes = list(np.unique(class_idxs))
    print(classes)
    markers = 'osD' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    if ran:
        np.random.shuffle(colors)

    for i, cls in enumerate(classes):
        mark = markers[i]
        print(i, mark)
        ax.plot(Y[class_idxs == cls, 0], Y[class_idxs == cls, 1], marker=mark,
                linestyle='', ms=4, label=str(cls), alpha=1, color=colors[i],
                markeredgecolor='black', markeredgewidth=0.15)
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0, fontsize=10, markerscale=2, frameon=False,
                  ncol=1, handletextpad=0.1, columnspacing=0.5)

    plt.xticks([])
    plt.yticks([])

    plt.savefig(filename)

    return ax


def dopca(X, dim=50):
    pcaten = PCA(n_components=dim)
    X_50 = pcaten.fit_transform(X)
    return X_50

def generate_random_pair(count, S, n_pairs):
    """
    Generate random pairwise constraints.
    """
    pair_nums = np.round(count.shape[0] * n_pairs)
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    i = randint(0, S.shape[0]-1)
    while pair_nums>0:
        ml_ind1.append(i)

        ml_ind2.append(np.argsort(S[i, :])[-2])

        cl_ind1.append(i)
        cl_ind2.append(np.argmin(S[i, :]))

        i = np.argmax(S[i, :])
        while i in ml_ind1:
            i = randint(0, S.shape[0]-1)
        pair_nums -= 1

    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)
