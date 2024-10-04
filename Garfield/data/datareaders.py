"""
This module contains data readers for the training of Garfield model.
"""

import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr
from glob import glob

from anndata import AnnData
from muon import MuData
import muon as mu
import scanpy as sc


def read_mtx(path):
    """\
    Read mtx format data folder including:

        * matrix file: e.g. count.mtx or matrix.mtx or their gz format
        * barcode file: e.g. barcode.txt
        * feature file: e.g. feature.txt

    Parameters
    ----------
    path
        the path store the mtx files

    Return
    ----------
    AnnData
    """
    for filename in glob(path + '/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path + '/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.var = pd.DataFrame(index=gene)
        elif 'feature' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, 1].values
            adata.var = pd.DataFrame(index=gene)
    return adata

def read_scData(path, backed=False):
    """
    Read single cell dataset from single file

    Parameters
    ----------
    path
        the path store the file

    Return
    ------
    AnnData
    """
    if path is not None:
        path = path
    else:
        path = os.getcwd()

    if os.path.exists(path + '.h5ad'): # path=file name
        adata = sc.read_h5ad(path + '.h5ad', backed=backed)
    elif os.path.isdir(path):  # mtx format
        adata = read_mtx(path)
    elif os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
        elif path.endswith('.loom'):
            adata = sc.read_loom(path, sparse=False)
        elif path.endswith('.h5mu'):
            adata = mu.read(path, backed=backed)
        elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac'])):
            adata = mu.read(path, backed=backed)
    else:
        raise ValueError("File {} not exists".format(path))

    if not issparse(adata.X) and not backed:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata

def read_multi_scData(root):
    """
    Read single cell dataset from multiple files

    Parameters
    ----------
    root
        the root store the single-cell data files, each file represent one dataset

    Return
    ----------
    AnnData
    """
    if root.split('/')[-1] == '*':
        adata = []
        for root in sorted(glob(root)):
            adata.append(read_scData(root))
        return AnnData.concatenate(*adata, batch_key='sub_batch', index_unique=None)
    else:
        return read_scData(root)

def concat_data(
        data_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        index_unique=None,
        save=None
):
    """
    Concatenate multiple datasets along the observations axis with name ``batch_key``.

    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a “batch”.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    index_unique
        Make the index unique by joining the existing index names with the batch category, using index_unique='-', for instance. Provide None to keep existing indices.
    save
        Path to save the new merged AnnData. Default: None.

    Returns
    ----------
    New merged AnnData.
    """
    if len(data_list) == 1:
        return read_multi_scData(data_list[0])
    elif isinstance(data_list, str):
        return read_multi_scData(data_list)
    elif isinstance(data_list, (AnnData, MuData)):
        return data_list

    adata_list = []
    for root in data_list:
        if isinstance(root, (AnnData, MuData)):
            adata = root
        else:
            adata = read_multi_scData(root)
        adata_list.append(adata)

    if isinstance(data_list[0], AnnData):
        if batch_categories is None:
            batch_categories = list(map(str, range(len(adata_list))))
        else:
            assert len(adata_list) == len(batch_categories)
        # [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
        concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                     batch_categories=batch_categories, index_unique=index_unique)
    elif isinstance(data_list[0], MuData):
        concat = adata_list ## MuData

    if batch_key not in concat.obs:
        concat.obs[batch_key] = 'batch'
    concat.obs[batch_key] = concat.obs[batch_key].astype('category')

    if save:
        concat.write(save, compression='gzip')
    return concat



