import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import h5py


topk_global = 30
topk_local = 5

def construct_graph(features, label, method, topk, fname):
    num = len(label)
    print(f"Number of samples: {num}")

    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(y) * np.linalg.norm(features))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], - (topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                continue
            if label[vv] != label[i]:
                counter += 1
            f.write('{} {}\n'.format(i, vv))
            A[i, vv] = 1
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))
    return A

# File paths and similarity metrics
file_path = 'data/LPS/lps_int2_2000.h5'
method = ['heat', 'cos', 'ncos', 'p']


with h5py.File(file_path, 'r') as file:
    x = np.array(file['X'])
    y = np.array(file['Y']).flatten()  # Make sure y is a one-dimensional array
    print("Data shape:", x.shape)

# Constructing an adjacency matrix of global features
fname_global = 'graph/LPS/lps_int2_2global_ncos.txt'
A_global = construct_graph(x, y, 'ncos', topk_global, fname_global)

# Constructing an adjacency matrix of local features
fname_local = 'graph/LPS/lps_int2_2local_ncos.txt'
A_local = construct_graph(x, y, 'ncos', topk_local, fname_local)


