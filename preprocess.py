import h5py
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True,
              normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    # Ensure that adata.X is the un-normalized count data
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=200)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata



file_path = 'data/LPS/lps_int2.h5'
save_path = 'data/LPS/lps_int2_2000.h5'

with h5py.File(file_path, 'r') as file:
    data = np.array(file['X'], dtype=np.float64)
    labels = np.array(file['Y'])
    print(data.shape)

    adata = sc.AnnData(data)
    adata.obs['labels'] = labels


    print("Shape of the original dataset:", data.shape)  # Cell x Gene
    print(labels.shape)
    print("Number of variables (genes) in the original dataset:", data.shape[1])     # Column: Genes
    print("Number of observations (cells) in the original dataset:", data.shape[0])  # Row: Cells

    # Normalization, selection of highly variable genes
    adata = normalize(adata, highly_genes = 2000)


# Save the normalized data to a new .h5 file
with h5py.File(save_path, 'w') as f_normalized:
    f_normalized.create_dataset('X', data=adata.X, compression="gzip", compression_opts=9)
    # 如果有标签或其他形式的附加信息，也保存
    f_normalized.create_dataset('Y', data=adata.obs['labels'], compression="gzip", compression_opts=9)

    f_normalized.create_dataset('highly_variable', data=adata.var['highly_variable'], compression="gzip", compression_opts=9)
    f_normalized.create_dataset('size_factors', data=adata.obs['size_factors'], compression="gzip", compression_opts=9)
    f_normalized.create_dataset('raw', data=adata.raw.X, compression="gzip", compression_opts=9)

# Print the normalized data format
print("Shape of the normalized dataset:", adata.X.shape)
print("Data type of the dataset:", adata.X.dtype)
print("Number of variables (genes):", adata.n_vars)
print("Number of observations (cells):", adata.n_obs)
print("Y", adata.obs['labels'])
print("highly_variable", adata.var['highly_variable'])
print("size_factors", adata.obs['size_factors'])
print("raw", adata.raw.X)
print(adata.raw.X.shape)










