# scG-cluster
We introduce a novel deep clustering method, scG-cluster, which leverages dual topology adaptive graph convolutional networks (TAGCNs) to enhance neighbor graph construction by integrating node distribution information and addressing the oversmoothing issue common in GCNs. The scG-cluster framework comprises data preprocessing and adjacency graph construction, dual TAGCN encoding, multi-task decoding, and an unsupervised clustering.

Extensive empirical evaluations on six distinct scRNA-seq datasets reveal that scG-cluster consistently surpasses current state-of-the-art methods in both clustering accuracy and scalability.
### 1.Data preprocessing
After obtaining the scRNA-seq data, we need to do preliminary processing of the gene expression data. After the data is preprocessed, it will be stored in the data catalog (preprocess the data using `preprocess`.py). Due to space limitations on github, we are unable to put the full data on there. Please download the detailed data file from the following website.
### 2.Generate graphs
We execute the `graph_function.py` file to generate the graphs needed for input and store them in the graph folder.
### 3.Pre-training and Training
To get better training results, we execute the `train.py` file. The model is first pre-trained, and the pre-trained weights are saved under the model file, followed by clustering training based on the initial clustering centers obtained from the pre-training.
### 4.Datasets
The specific data file can be downloaded from the following website:
- The 10×PBMC datasets from [10x Genomics](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k),
- The Mouse ES cells from [Figshare](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525),
- The Mouse bladder cells from [Figshare](https://figshare.com/s/865e694ad06d5857db4b),
- The Worm neuron cells from [Cole Trapnell Lab](https://cole-trapnell-lab.github.io/worm-rna/docs/),
- The GSE60361 from [Figshare](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361),
- The LPS datasets from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17721).
