"""
Utility functions for matching
"""
import pandas as pd
import numpy as np
import pynndescent
from scipy.optimize import linear_sum_assignment

from . import _utils as utils

# modify from Maxfuse: https://github.com/shuxiaoc/maxfuse/blob/main/maxfuse/match_utils.py#L273
def match_cells(arr1, arr2, base_dist=None, wt_on_base_dist=0, verbose=True):
    """
    Get matching between arr1 and arr2 using linear assignment, the distance is 1 - Pearson correlation.

    Parameters
    ----------
    arr1: np.array of shape (n_samples1, n_features)
        The first data matrix
    arr2: np.array of shape (n_samples2, n_features)
        The second data matrix
    base_dist: None or np.ndarray of shape (n_samples1, n_samples2)
        Baseline distance matrix
    wt_on_base_dist: float between 0 and 1
        The final distance matrix to use is (1-wt_on_base_dist) * dist[arr1, arr2] + wt_on_base_dist * base_dist
    verbose: bool, default=True
        Whether to print the progress

    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their distance is vals[i]
    """
    if verbose:
        print('Start the matching process...', flush=True)
        print('Computing the distance matrix...', flush=True)
    dist = utils.cdist_correlation(arr1, arr2)
    if base_dist is not None:
        if verbose:
            print(
                f'Interpolating {1-wt_on_base_dist} * dist[arr1, arr2] + {wt_on_base_dist} * base_dist...',
                flush=True
            )
        dist = (1-wt_on_base_dist) * dist + wt_on_base_dist * base_dist
    if verbose:
        print('Solving linear assignment...', flush=True)
    rows, cols = linear_sum_assignment(dist)
    if verbose:
        print('Linear assignment completed!', flush=True)

    return rows, cols, np.array([dist[i, j] for i, j in zip(rows, cols)])


def get_initial_matching(
        arr1, arr2,
        randomized_svd=True,
        svd_runs=1,
        svd_components1=30, svd_components2=30,
        verbose=True
):
    """
    Assume the features of arr1 and arr2 are column-wise directly comparable,
    obtain a matching by minimizing the correlation distance between arr1 and arr2.

    Parameters
    ----------
    arr1: np.array of shape (n_samples1, n_features1)
        The first data matrix.
    arr2: np.array of shape (n_samples2, n_features2)
        The second data matrix.
    randomized_svd: bool, default=False
        Whether to use randomized svd.
    svd_runs: int, default=1
        Randomized SVD will result in different runs,
        so if randomized_svd=True, perform svd_runs many randomized SVDs, and pick the one with the
        smallest Frobenious reconstruction error.
        If randomized_svd=False, svd_runs is forced to be 1.
    svd_components1: None or int
        If None, then do not do SVD,
        else, number of components to keep when doing SVD de-noising for the first data matrix.
    svd_components2: None or int
        Same as svd_components1 but for the second data matrix.
    verbose: bool, default=True
        Whether to print the progress.

    Returns
    -------
    matching: list of length 3
        rows, cols, vals = matching,
        Each matched pair is rows[i], cols[i], their distance is vals[i].
    """
    # assert arr1.shape[1] == arr2.shape[1]

    arr1 = utils.convert_to_numpy(arr1)
    arr2 = utils.convert_to_numpy(arr2)
    arr1, arr2 = utils.drop_zero_variability_columns(arr_lst=[arr1, arr2])

    # 确保 SVD 分量数不超过特征数，如果超过则调整并发出警告
    if arr1.shape[1] < svd_components1 or arr2.shape[1] < svd_components2:
        svd_components1, svd_components2 = arr1.shape[1]-1, arr2.shape[1]-1
        # 发出警告信息
        print(
            f"Warning: svd_components1 and svd_components2 were set too large and have been adjusted to {svd_components1} and {svd_components2}, "
            "which are the number of features in arr1 and arr2 respectively."
        )

    # denoising
    if verbose:
        print("Denoising the data...", flush=True)
    arr1_svd = utils.svd_denoise(
        arr=arr1, n_components=svd_components1, randomized=randomized_svd,
        n_runs=svd_runs
    )
    arr2_svd = utils.svd_denoise(
        arr=arr2, n_components=svd_components2, randomized=randomized_svd,
        n_runs=svd_runs
    )

    # if verbose:
    #     print('Normalizing and reducing the dimension of the data...', flush=True)
    # arr1_svd = utils.svd_embedding(
    #     arr=arr1, n_components=svd_components1,
    #     randomized=randomized_svd, n_runs=svd_runs
    # )
    # arr2_svd = utils.svd_embedding(
    #     arr=arr2, n_components=svd_components2,
    #     randomized=randomized_svd, n_runs=svd_runs
    # )

    res = match_cells(arr1=arr1_svd, arr2=arr2_svd, verbose=verbose)
    if verbose:
        print('Initial matching completed!', flush=True)

    return res

def get_refined_matching_one_iter(
        init_matching, arr1, arr2,
        filter_prop=0,
        cca_components=15,
        cca_max_iter=2000,
        verbose=True
):
    """
    Run one iteration of CCA refinement.

    Parameters
    ----------
    init_matching: list
        init_matching[0][i], init_matching[1][i] is a matched pair,
        and init_matching[2][i] is the distance for this pair
    arr1: np.array of shape (n_samples1, n_features1)
        The first data matrix.
    arr2: np.array of shape (n_samples2, n_features2)
        The second data matrix.
    filter_prop: float, default=0
        Proportion of matched pairs to discard before feeding into refinement iterations.
    cca_components: int, default=15
        Number of CCA components.
    cca_max_iter: int, default=2000
        Maximum number of CCA iterations.
    verbose: bool, default=True
        Whether to print the

    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their distance is vals[i]
    """
    if verbose:
        print('Fitting CCA...', flush=True)
    arr1_cca, arr2_cca, _ = utils.cca_embedding(
        arr1=arr1, arr2=arr2,
        init_matching=init_matching, filter_prop=filter_prop, n_components=cca_components, max_iter=cca_max_iter
    )

    return match_cells(
        arr1=arr1_cca, arr2=arr2_cca, verbose=verbose
    )


def get_refined_matching(
        init_matching, arr1, arr2,
        randomized_svd=False, svd_runs=1,
        svd_components1=None, svd_components2=None,
        n_iters=3, filter_prop=0,
        cca_components=20,
        cca_max_iter=2000,
        verbose=True
):
    """
    Refinement of init_matching.

    Parameters
    ----------
    init_matching: list
        init_matching[0][i], init_matching[1][i] is a matched pair,
        and init_matching[2][i] is the distance for this pair.
    arr1: np.array of shape (n_samples1, n_features1)
        The first data matrix.
    arr2: np.array of shape (n_samples2, n_features2)
        The second data matrix.
    randomized_svd: bool, default=False
        Whether to use randomized SVD
    svd_runs: int, default=1
        Randomized SVD will result in different runs,
        so if randomized_svd=True, perform svd_runs many randomized SVDs, and pick the one with the
        smallest Frobenious reconstruction error.
        If randomized_svd=False, svd_runs is forced to be 1.
    svd_components1: None or int
        If None, then do not do SVD,
        else, number of components to keep when doing SVD de-noising for the first data matrix
        before feeding into CCA.
    svd_components2: None or int
        Same as svd_components1 but for the second data matrix.
    n_iters: int, default=3
        Number of refinement iterations.
    filter_prop: float, default=0
        Proportion of matched pairs to discard before feeding into refinement iterations.
    cca_components: int, default=15
        Number of CCA components.
    cca_max_iter: int, default=2000,
        Maximum number of CCA iterations.
    verbose: bool, default=True
        Whether to print the progress.

    Returns
    -------
    matching: list of length 3
        rows, cols, vals = matching,
        Each matched pair is rows[i], cols[i], their distance is vals[i].
    """
    ns = [len(x) for x in init_matching]
    assert ns[0] == ns[1] == ns[2]
    assert isinstance(n_iters, int) and n_iters >= 1
    assert 0 <= int(ns[0] * filter_prop) < ns[0]

    assert 1 <= cca_components <= min(arr1.shape[1], arr2.shape[1])

    # 确保 SVD 分量数不超过特征数，如果超过则调整并发出警告
    if arr1.shape[1] < cca_components or arr2.shape[1] < cca_components:
        cca_components = min(arr1.shape[1], arr2.shape[1])
        # 发出警告信息
        print(
            f"Warning: cca_components were set too large and have been adjusted to {cca_components}, "
            "which are the min number of features of arr1 and arr2."
        )

    if verbose:
        print('Normalizing and reducing the dimension of the data...', flush=True)
    arr1 = utils.svd_embedding(
        arr=arr1, n_components=svd_components1,
        randomized=randomized_svd, n_runs=svd_runs
    )
    arr2 = utils.svd_embedding(
        arr=arr2, n_components=svd_components2,
        randomized=randomized_svd, n_runs=svd_runs
    )

    # assert arr1.shape[1] == arr2.shape[1]
    # denoising
    # if verbose:
    #     print("Denoising the data...", flush=True)
    # arr1 = utils.svd_denoise(
    #     arr=arr1, n_components=svd_components1, randomized=randomized_svd,
    #     n_runs=svd_runs
    # )
    # arr2 = utils.svd_denoise(
    #     arr=arr2, n_components=svd_components2, randomized=randomized_svd,
    #     n_runs=svd_runs
    # )

    # prepare the distance matrix used in the initial matching
    cca_matching = init_matching
    # iterative refinement
    for it in range(n_iters):
        if verbose:
            print('Now at iteration {}:'.format(it), flush=True)
        cca_matching = get_refined_matching_one_iter(
            init_matching=cca_matching,
            arr1=arr1, arr2=arr2,
            filter_prop=filter_prop,
            cca_components=cca_components,
            cca_max_iter=cca_max_iter, verbose=verbose
        )

    if verbose:
        print('Refined matching completed!', flush=True)
    return cca_matching

def fit_svd_on_full_data(arr1, arr2, randomized_svd=False, svd_runs=1,
                         svd_components1=None, svd_components2=None):
    """Perform SVD on full self.active_arr1 and self.active_arr2 and save the functions that reduce the dimension
    of the data.
    """
    if svd_components1 is not None:
        u1, s1, vh1 = utils.robust_svd(
            arr=arr1, n_components=svd_components1,
            randomized=randomized_svd, n_runs=svd_runs
        )
        rotation_before_cca_on_active_arr1 = lambda arr: arr @ vh1.T
    else:
        rotation_before_cca_on_active_arr1 = lambda arr: arr

    if svd_components2 is not None:
        u2, s2, vh2 = utils.robust_svd(
            arr=arr2, n_components=svd_components2,
            randomized=randomized_svd, n_runs=svd_runs
        )
        rotation_before_cca_on_active_arr2 = lambda arr: arr @ vh2.T
    else:
        rotation_before_cca_on_active_arr2 = lambda arr: arr

    return rotation_before_cca_on_active_arr1, rotation_before_cca_on_active_arr2


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


def address_matching_redundancy(matching, direction='both'):
    """
    Make a potentially multiple-to-multiple matching to an one-to-one matching according to order.

    Parameters
    ----------
    matching: list of length three
        rows, cols, vals = matching: list
        Each matched pair of rows[i], cols[i], their score (the larger, the better) is vals[i]
    direction: 'left' or 'right' or 'both', default=both
        If None, do nothing;
        If (1, 2), then the redundancy is addressed by making matching
        an injective map from the first dataset to the second;
        if (2, 1), do the other way around.

    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their score is vals[i].
    """
    if direction == 'both':
        return matching
    res = [[], [], []]
    if direction == 'left':
        idx1_to_idx2 = dict()
        idx1_to_score = dict()
        for i, j, score in zip(matching[0], matching[1], matching[2]):
            if i not in idx1_to_idx2:
                idx1_to_idx2[i] = j
                idx1_to_score[i] = score
            elif score > idx1_to_score[i]:
                idx1_to_idx2[i] = j
                idx1_to_score[i] = score
        for idx1, idx2 in idx1_to_idx2.items():
            res[0].append(idx1)
            res[1].append(idx2)
            res[2].append(idx1_to_score[idx1])
    elif direction == 'right':
        idx2_to_idx1 = dict()
        idx2_to_score = dict()
        for i, j, score in zip(matching[0], matching[1], matching[2]):
            if j not in idx2_to_idx1:
                idx2_to_idx1[j] = i
                idx2_to_score[j] = score
            elif score > idx2_to_score[j]:
                idx2_to_idx1[j] = i
                idx2_to_score[j] = score
        for idx2, idx1 in idx2_to_idx1.items():
            res[0].append(idx1)
            res[1].append(idx2)
            res[2].append(idx2_to_score[idx2])
    else:
        raise NotImplementedError('direction must be in {both, left, right}.')

    return res
