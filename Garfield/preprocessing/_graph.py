"""
Functions for graph construction
"""
import numpy as np
import warnings
import scanpy as sc
import scipy.sparse as sp
import igraph as ig
import leidenalg
import pynndescent

from . import _utils as utils

def get_nearest_neighbors(query_arr, target_arr, svd_components=None, randomized_svd=False, svd_runs=1,
                          metric='correlation'):
    """
    For each row in query_arr, compute its nearest neighbor in target_arr.

    Parameters
    ----------
    query_arr: np.array of shape (n_samples1, n_features)
        The query data matrix.
    target_arr: np.array of shape (n_samples2, n_features)
        The target data matrix.
    svd_components: None or int, default=None
        If not None, will first conduct SVD to reduce the dimension
        of the vertically stacked version of query_arr and target_arr.
    randomized_svd: bool, default=False
        Whether to use randomized SVD.
    svd_runs: int, default=1
        Run multiple instances of SVD and select the one with the lowest Frobenious reconstruction error.
    metric: string, default='correlation'
        The metric to use in nearest neighbor search.

    Returns
    -------
    neighbors: np.array of shape (n_samples1)
        The i-th element is the index in target_arr to whom the i-th row of query_arr is closest to.
    dists: np.array of shape (n_samples1)
        The i-th element is the distance corresponding to neighbors[i].
    """
    query_arr = utils.convert_to_numpy(query_arr)
    target_arr = utils.convert_to_numpy(target_arr)
    arr = np.vstack([query_arr, target_arr])
    arr = utils.svd_embedding(
        arr=arr, n_components=svd_components,
        randomized=randomized_svd,
        n_runs=svd_runs
    )
    query_arr = arr[:query_arr.shape[0], :]
    pivot_arr = arr[query_arr.shape[0]:, :]
    # approximate nearest neighbor search
    index = pynndescent.NNDescent(pivot_arr, n_neighbors=100, metric=metric)
    neighbors, dists = index.query(query_arr, k=50)
    neighbors, dists = neighbors[:, 0], dists[:, 0]
    return neighbors, dists
