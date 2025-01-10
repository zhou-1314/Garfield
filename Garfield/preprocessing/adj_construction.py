import warnings
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import networkx as nx
from scipy.spatial import distance

import scipy
from scipy.sparse import csr_matrix
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors

import scanpy as sc
from anndata import AnnData

from ._utils import reindex
from .match_utils import (
    get_initial_matching,
    get_refined_matching,
    fit_svd_on_full_data,
    address_matching_redundancy,
)
from . import _utils as utils
from . import _graph as graph

## Spatial graph construction
def graph_computing(adata, k_cutoff=None, model="Radius", batch_key=None, verbose=True):
    """\
    Construct the spatial neighbor networks.
    """
    assert model in ["mu_std", "Radius", "KNN"], "Invalid model specified"
    if verbose:
        print("------Calculating spatial graph...")

    edgeList = []

    if batch_key is not None and len(np.unique(adata.obs[batch_key])) > 1:
        sample_names = np.unique(adata.obs[batch_key])
    else:
        sample_names = [None]

    for sample_name in sample_names:
        if sample_name is not None:
            sub_adata = adata[adata.obs[batch_key] == sample_name].copy()
        else:
            sub_adata = adata.copy()

        coor = pd.DataFrame(sub_adata.obsm["spatial"])
        coor.index = sub_adata.obs.index
        coor.columns = ["imagerow", "imagecol"]

        if model == "mu_std":
            distMat = distance.cdist(
                coor, coor, "euclidean"
            )  # compute all distances at once
            for node_idx in range(sub_adata.shape[0]):
                nearest_indices = np.argsort(distMat[node_idx])[: k_cutoff + 1]
                nearest_dists = distMat[node_idx, nearest_indices[1 : k_cutoff + 1]]
                boundary = np.mean(nearest_dists) + np.std(nearest_dists)
                for j, dist in zip(nearest_indices[1:], nearest_dists):
                    weight = 1.0 if dist <= boundary else 0.0
                    edgeList.append((node_idx, j, weight))

        elif model == "Radius":
            nbrs = NearestNeighbors(radius=k_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            for it in range(indices.shape[0]):
                edgeList.extend(
                    list(zip([it] * indices[it].shape[0], indices[it], distances[it]))
                )

        elif model == "KNN":
            nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            for it in range(indices.shape[0]):
                edgeList.extend(
                    list(zip([it] * indices.shape[1], indices[it, :], distances[it, :]))
                )

    return edgeList


def edgeList2edgeDict(edgeList, node_size):
    graphdict = defaultdict(list)
    for start, end, _ in edgeList:
        graphdict[start].append(end)
    return {
        node: graphdict[node] for node in range(node_size)
    }  # ensure all nodes are present


def graph_construction(adata, mode, k, batch_key, verbose=True):
    edge_list = graph_computing(
        adata, k_cutoff=k, model=mode, batch_key=batch_key, verbose=verbose
    )
    graph_dict = edgeList2edgeDict(edge_list, adata.shape[0])
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
    return adj_org

def split_into_batches(
        shared_arr1, shared_arr2,
        max_outward_size=5000, matching_ratio=1,
        batching_scheme='pairwise',
        seed=None, verbose=True
):
    def _split(arr, n_batches, curr_seed):
        indices = list(np.random.RandomState(curr_seed).permutation(arr.shape[0]))
        res = []
        batch_size = int(len(indices) // n_batches)
        for b in range(n_batches):
            res.append(indices[b * batch_size:(b + 1) * batch_size])
        res[-1].extend(indices[n_batches * batch_size:])
        return res

    # cut arr2 into batches
    max_arr2_batch_size = int(max_outward_size * matching_ratio)
    n_batches2 = max(1, int(shared_arr2.shape[0] // max_arr2_batch_size))
    arr2_batch_size = int(shared_arr2.shape[0] // n_batches2)
    batch_to_indices2 = _split(
        arr=shared_arr2,
        n_batches=n_batches2,
        curr_seed=seed
    )
    # cut arr1 into batches
    max_arr1_batch_size = int(arr2_batch_size // matching_ratio)
    n_batches1 = max(1, int(shared_arr1.shape[0] // max_arr1_batch_size))
    arr1_batch_size = int(shared_arr1.shape[0] // n_batches1)
    if seed is not None:
        seed = seed + 1
    batch_to_indices1 = _split(
        arr=shared_arr1,
        n_batches=n_batches1,
        curr_seed=seed
    )

    # construct mapping between batches
    b1 = 0
    b2 = 0
    batch1_to_batch2 = []
    if batching_scheme == 'cyclic':
        for i in range(max(n_batches1, n_batches2)):
            batch1_to_batch2.append((b1, b2))
            b1 = (b1 + 1) % n_batches1
            b2 = (b2 + 1) % n_batches2
    elif batching_scheme == 'pairwise':
        for b1 in range(n_batches1):
            for b2 in range(n_batches2):
                batch1_to_batch2.append((b1, b2))

    if verbose:
        print(('The first data is split into {} batches, '
               'average batch size is {}, and max batch size is {}.').format(
            n_batches1, arr1_batch_size, len(batch_to_indices1[-1])
        ), flush=True)
        print(('The second data is split into {} batches, '
               'average batch size is {}, and max batch size is {}.').format(
            n_batches2, arr2_batch_size, len(batch_to_indices2[-1])
        ), flush=True)
        print('Batch to batch correspondence is:\n  {}.'.format(
            [str(i) + '<->' + str(j) for i, j in batch1_to_batch2]
        ), flush=True)

    if arr1_batch_size <= 1000:
        warnings.warn('Batch size for arr1 is <= 1000: '
                      'consider setting a smaller matching ratio for better performance.')

    return batch_to_indices1, batch_to_indices2, batch1_to_batch2

def find_initial_pivots(
    arr1,
    arr2,
    max_outward_size=5000,
    matching_ratio=1,
    batching_scheme='pairwise',
    seed=123,
    svd_components1=None,
    svd_components2=None,
    randomized_svd=False,
    svd_runs=1,
    verbose=True,
):
    """
    Perform initial matching.

    Parameters
    ----------
    svd_components1: None or int, default=None
        If not None, perform SVD to reduce the dimension of self.shared_arr1.
    svd_components2: None or int, default=None
        If not None, perform SVD to reduce the dimension of self.shared_arr2.
    randomized_svd: bool, default=False
        Whether to use randomized SVD.
    svd_runs: int, default=1
        Perform multiple runs of SVD and the one with lowest Frobenious reconstruction error is selected.
    verbose: bool, default=True
        Whether to print the progress.

    Returns
    -------
    None
    """
    init_matching = []
    arr1 = utils.convert_to_numpy(arr1)
    arr2 = utils.convert_to_numpy(arr2)

    if arr1.shape[0] < 10000 and arr2.shape[0] < 10000:
        # 不进行minibatch划分
        init_matching.append(
            get_initial_matching(
                arr1=arr1,
                arr2=arr2,
                randomized_svd=randomized_svd,
                svd_runs=svd_runs,
                svd_components1=svd_components1,
                svd_components2=svd_components2,
                verbose=False,
            )
        )

        if verbose:
            print("Init_matching done!", flush=True)
        return init_matching
    else:
        # divide minibatch
        batch_to_indices1, batch_to_indices2, batch1_to_batch2 = split_into_batches(
            arr1, arr2,
            max_outward_size=max_outward_size,
            matching_ratio=matching_ratio,
            batching_scheme=batching_scheme,
            seed=seed,
            verbose=verbose
        )

        for b1, b2 in batch1_to_batch2:
            if verbose:
                print(
                    'Now at batch {}<->{}...'.format(b1, b2),
                    flush=True
                )
            arr1_batch = arr1[batch_to_indices1[b1], :]
            arr2_batch = arr2[batch_to_indices2[b2], :]

            init_matching.append(
                get_initial_matching(
                    arr1=arr1_batch,
                    arr2=arr2_batch,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    svd_components1=svd_components1,
                    svd_components2=svd_components2,
                    verbose=False,
                )
            )

        if verbose:
            print("Init_matching done!", flush=True)
        return init_matching


def refine_pivots(
    init_matching,
    arr1,
    arr2,
    max_outward_size=5000,
    matching_ratio=1,
    batching_scheme='pairwise',
    seed=None,
    svd_components1=None,
    svd_components2=None,
    cca_components=None,
    filter_prop=0,
    n_iters=1,
    randomized_svd=False,
    svd_runs=1,
    cca_max_iter=2000,
    verbose=True,
):
    refined_matching = []
    arr1 = utils.convert_to_numpy(arr1)
    arr2 = utils.convert_to_numpy(arr2)

    if arr1.shape[0] < 10000 and arr2.shape[0] < 10000:
        # 不进行minibatch划分
        refined_matching.append(
            get_refined_matching(
                init_matching=init_matching[0],
                arr1=arr1,
                arr2=arr2,
                randomized_svd=randomized_svd,
                svd_runs=svd_runs,
                svd_components1=svd_components1,
                svd_components2=svd_components2,
                n_iters=n_iters,
                filter_prop=filter_prop,
                cca_components=cca_components,
                cca_max_iter=cca_max_iter,
                verbose=False,
            )
        )

        if verbose:
            print("Refined_matching done!", flush=True)
        return refined_matching
    else:
        # divide minibatch
        batch_to_indices1, batch_to_indices2, batch1_to_batch2 = split_into_batches(
            arr1, arr2,
            max_outward_size=max_outward_size,
            matching_ratio=matching_ratio,
            batching_scheme=batching_scheme,
            seed=seed,
            verbose=False
        )

        for batch_idx, (b1, b2) in enumerate(batch1_to_batch2):
            if verbose:
                print(
                    'Now at batch {}<->{}...'.format(b1, b2),
                    flush=True
                )
            arr1_batch = arr1[batch_to_indices1[b1], :]
            arr2_batch = arr2[batch_to_indices2[b2], :]

            refined_matching.append(
                get_refined_matching(
                    init_matching=init_matching[batch_idx],
                    arr1=arr1_batch,
                    arr2=arr2_batch,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    svd_components1=svd_components1,
                    svd_components2=svd_components2,
                    n_iters=n_iters,
                    filter_prop=filter_prop,
                    cca_components=cca_components,
                    cca_max_iter=cca_max_iter,
                    verbose=False,
                )
            )

        if verbose:
            print("Refined_matching done!", flush=True)
        return refined_matching


def filter_bad_matches(
    arr1,
    arr2,
    refined_matching=None,
    propagated_matching=None,
    target="pivot",
    max_outward_size=5000,
    matching_ratio=1,
    batching_scheme='pairwise',
    seed=None,
    filter_prop=0.0,
    randomized_svd=False,
    svd_runs=1,
    svd_components1=None,
    svd_components2=None,
    cca_components=20,
    cca_max_iter=2000,
    verbose=True,
):
    arr1_raw = arr1
    arr2_raw = arr2
    # conduct filtering
    n_remaining = 0
    if verbose:
        print("Begin filtering...", flush=True)
    if target == "pivot":
        matching_to_be_filtered = refined_matching
    elif target == "propagated":
        matching_to_be_filtered = propagated_matching
    else:
        raise ValueError("target must be in {'pivot', 'propagated'}.")

    remaining_indices_after_filtering = []
    idx2_to_indices1 = defaultdict(set)
    # record the locations that survive the filtering
    if arr1.shape[0] < 10000 and arr2.shape[0] < 10000:
        rows, cols, vals = matching_to_be_filtered[0]
        # anything with val <= thresh will be retained
        thresh = np.quantile(vals, 1 - filter_prop)
        remaining_indices_after_filtering.append(
            [i for i in range(len(vals)) if vals[i] <= thresh]
        )
        n_remaining += len(remaining_indices_after_filtering[0])
        for i in remaining_indices_after_filtering[0]:
            idx1, idx2 = rows[i], cols[i]
            idx2_to_indices1[idx2].add(idx1)
    else:
        # divide minibatch
        _, _, batch1_to_batch2 = split_into_batches(
            arr1, arr2,
            max_outward_size=max_outward_size,
            matching_ratio=matching_ratio,
            batching_scheme=batching_scheme,
            seed=seed,
            verbose=False
        )
        for batch_idx, (b1, b2) in enumerate(batch1_to_batch2):
            rows, cols, vals = matching_to_be_filtered[batch_idx]
            # anything with val <= thresh will be retained
            thresh = np.quantile(vals, 1 - filter_prop)
            remaining_indices_after_filtering.append(
                [i for i in range(len(vals)) if vals[i] <= thresh]
            )
            n_remaining += len(remaining_indices_after_filtering[batch_idx])
            for i in remaining_indices_after_filtering[batch_idx]:
                idx1, idx2 = rows[i], cols[i]
                idx2_to_indices1[idx2].add(idx1)

    # if target == "pivot":
    #     remaining_indices_in_refined_matching = remaining_indices_after_filtering #[0]
    # else:
    #     remaining_indices_in_propagated_matching = remaining_indices_after_filtering #[0]

    if verbose:
        print(
            "{}/{} pairs of matched cells remain after the filtering.".format(
                n_remaining,
                np.sum(
                    [
                        len(per_batch_matching[0])
                        for per_batch_matching in matching_to_be_filtered
                    ]
                ),
            )
        )

    # convert idx2_to_indices1 to a dict of lists for ease of later usage
    idx2_to_indices1 = {
        idx2: sorted(indices1) for idx2, indices1 in idx2_to_indices1.items()
    }

    # fit CCA on pivots
    if verbose:
        print("Fitting CCA on pivots...", flush=True)
    if svd_components1 >= min(arr1.shape):
        svd_components1 = min(arr1.shape) - 1
    if svd_components2 >= min(arr2.shape):
        svd_components2 = min(arr2.shape) - 1
    (
        rotation_before_cca_on_active_arr1,
        rotation_before_cca_on_active_arr2,
    ) = fit_svd_on_full_data(
        arr1,
        arr2,
        randomized_svd=randomized_svd,
        svd_runs=svd_runs,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
    )
    arr1_list, arr2_list = [], []
    for idx2, indices1 in idx2_to_indices1.items():
        arr1_list.append(
            arr1[indices1[0], :].toarray().squeeze()
        )  # np.mean(self.active_arr1[indices1, :], axis=0)
        arr2_list.append(arr2[idx2, :].toarray().squeeze())
    arr1 = rotation_before_cca_on_active_arr1(np.array(arr1_list))
    arr2 = rotation_before_cca_on_active_arr2(np.array(arr2_list))

    cca_on_pivots = CCA(n_components=cca_components, max_iter=cca_max_iter)
    cca_on_pivots.fit(arr1, arr2)

    if verbose:
        print("Scoring matched pairs...", flush=True)
    # transform the whole dataset
    arr1_raw = rotation_before_cca_on_active_arr1(arr1_raw)
    arr2_raw = rotation_before_cca_on_active_arr2(arr2_raw)
    arr1, arr2 = cca_on_pivots.transform(arr1_raw, arr2_raw)
    arr1 = utils.center_scale(arr1)
    arr2 = utils.center_scale(arr2)

    # add distances to idx2_to_indices1
    all_indices1, all_indices2 = [], []
    for idx2, indices1 in idx2_to_indices1.items():
        for idx1 in indices1:
            all_indices1.append(idx1)
            all_indices2.append(idx2)
    # compute all possible pairs of distances
    pearson_correlations = utils.pearson_correlation(
        arr1[all_indices1, :], arr2[all_indices2, :]
    )

    cnt = 0
    for idx2, indices1 in idx2_to_indices1.items():
        indices_and_scores = []
        for idx1 in indices1:
            indices_and_scores.append((idx1, pearson_correlations[cnt]))
            cnt += 1
        idx2_to_indices1[idx2] = indices_and_scores

    if target == "pivot":
        pivot2_to_pivots1 = idx2_to_indices1
        remaining_indices_in_refined_matching = remaining_indices_after_filtering
        return pivot2_to_pivots1, remaining_indices_in_refined_matching
    else:
        propidx2_to_propindices1 = idx2_to_indices1
        remaining_indices_in_propagated_matching = remaining_indices_after_filtering
        return propidx2_to_propindices1, remaining_indices_in_propagated_matching


def propagate(
    refined_matching=None,
    remaining_indices_in_refined_matching=None,
    arr1=None,
    arr2=None,
    svd_components1=None,
    svd_components2=None,
    max_outward_size=5000,
    matching_ratio=1,
    batching_scheme='pairwise',
    seed=None,
    metric="euclidean",
    randomized_svd=False,
    svd_runs=1,
    verbose=True,
):
    """
    For indices not in pivots, find their matches by nearest neighbor search.

    Parameters
    ----------
    svd_components1: None or int, default=None
        If not None, perform SVD to reduce the dimension of self.active_arr1
        before doing internal nearest neighbor search.
    svd_components2: None or int, default=None
        If not None, perform SVD to reduce the dimension of self.active_arr1
        before doing internal nearest neighbor search.
    wt1: float, default=0.7
        Weight to put on raw data of self.active_arr1 when doing smoothing.
    wt2: float, default=0.7
        Weight to put on raw data of self.active_arr2 when doing smoothing.
    metric: string, default='correlation'
        The metric to use in nearest neighbor search.
    randomized_svd: bool, default=False
        Whether to perform randomized SVD.
    svd_runs: int, default=1
        Perform multiple runs of SVD and the one with lowest Frobenious reconstruction error is selected.
    verbose: bool, default=True
        Whether to print the progress.

    Returns
    -------
    None
    """
    propagated_matching = []
    if arr1.shape[0] < 10000 and arr2.shape[0] < 10000:
        curr_propagated_matching = [[], [], []]
        curr_refined_matching = refined_matching[0]
        # get good pivot indices that survived pivot filtering
        existing_indices = curr_refined_matching[0][remaining_indices_in_refined_matching[0]]
        good_indices1 = curr_refined_matching[0][existing_indices]
        good_indices2 = curr_refined_matching[1][existing_indices]

        # get arrays that were used when doing refined matching
        # get remaining indices
        # propagation will only be done for those indices
        good_indices1_set = set(good_indices1)
        remaining_indices1 = [i for i in range(arr1.shape[0]) if i not in good_indices1_set]
        good_indices2_set = set(good_indices2)
        remaining_indices2 = [i for i in range(arr2.shape[0]) if i not in good_indices2_set]

        # propagate for remaining indices in arr1
        if len(remaining_indices1) > 0:
            # get 1-nearest-neighbors and the corresponding distances
            (
                remaining_indices1_nns,
                remaining_indices1_nn_dists,
            ) = graph.get_nearest_neighbors(
                query_arr=arr1[remaining_indices1, :],
                target_arr=arr1[good_indices1, :],
                svd_components=svd_components1,
                randomized_svd=randomized_svd,
                svd_runs=svd_runs,
                metric=metric,
            )
            matched_indices2 = good_indices2[remaining_indices1_nns]
            curr_propagated_matching[0].extend(remaining_indices1)
            curr_propagated_matching[1].extend(matched_indices2)
            curr_propagated_matching[2].extend(remaining_indices1_nn_dists)

        # propagate for remaining indices in arr2
        if len(remaining_indices2) > 0:
            # get 1-nearest-neighbors and the corresponding distances
            (
                remaining_indices2_nns,
                remaining_indices2_nn_dists,
            ) = graph.get_nearest_neighbors(
                query_arr=arr2[remaining_indices2, :],
                target_arr=arr2[good_indices2, :],
                svd_components=svd_components2,
                randomized_svd=randomized_svd,
                svd_runs=svd_runs,
                metric=metric,
            )
            matched_indices1 = good_indices1[remaining_indices2_nns]
            curr_propagated_matching[0].extend(matched_indices1)
            curr_propagated_matching[1].extend(remaining_indices2)
            curr_propagated_matching[2].extend(remaining_indices2_nn_dists)

        propagated_matching.append(
            (
                np.array(curr_propagated_matching[0]),
                np.array(curr_propagated_matching[1]),
                np.array(curr_propagated_matching[2]),
            )
        )
    else:
        # divide minibatch
        _, _, batch1_to_batch2 = split_into_batches(
            arr1, arr2,
            max_outward_size=max_outward_size,
            matching_ratio=matching_ratio,
            batching_scheme=batching_scheme,
            seed=seed,
            verbose=False
        )
        for batch_idx, (b1, b2) in enumerate(batch1_to_batch2):
            if verbose:
                print(
                    'Now at batch {}<->{}...'.format(b1, b2),
                    flush=True
                )
            curr_propagated_matching = [[], [], []]
            curr_refined_matching = refined_matching[batch_idx]
            # get good pivot indices that survived pivot filtering
            existing_indices = curr_refined_matching[0][remaining_indices_in_refined_matching[batch_idx]]
            good_indices1 = curr_refined_matching[0][existing_indices]
            good_indices2 = curr_refined_matching[1][existing_indices]

            # get arrays that were used when doing refined matching
            # get remaining indices
            # propagation will only be done for those indices
            good_indices1_set = set(good_indices1)
            remaining_indices1 = [i for i in range(arr1.shape[0]) if i not in good_indices1_set]
            good_indices2_set = set(good_indices2)
            remaining_indices2 = [i for i in range(arr2.shape[0]) if i not in good_indices2_set]

            # propagate for remaining indices in arr1
            if len(remaining_indices1) > 0:
                # get 1-nearest-neighbors and the corresponding distances
                (
                    remaining_indices1_nns,
                    remaining_indices1_nn_dists,
                ) = graph.get_nearest_neighbors(
                    query_arr=arr1[remaining_indices1, :],
                    target_arr=arr1[good_indices1, :],
                    svd_components=svd_components1,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    metric=metric,
                )
                matched_indices2 = good_indices2[remaining_indices1_nns]
                curr_propagated_matching[0].extend(remaining_indices1)
                curr_propagated_matching[1].extend(matched_indices2)
                curr_propagated_matching[2].extend(remaining_indices1_nn_dists)

            # propagate for remaining indices in arr2
            if len(remaining_indices2) > 0:
                # get 1-nearest-neighbors and the corresponding distances
                (
                    remaining_indices2_nns,
                    remaining_indices2_nn_dists,
                ) = graph.get_nearest_neighbors(
                    query_arr=arr2[remaining_indices2, :],
                    target_arr=arr2[good_indices2, :],
                    svd_components=svd_components2,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    metric=metric,
                )
                matched_indices1 = good_indices1[remaining_indices2_nns]
                curr_propagated_matching[0].extend(matched_indices1)
                curr_propagated_matching[1].extend(remaining_indices2)
                curr_propagated_matching[2].extend(remaining_indices2_nn_dists)

            propagated_matching.append(
                (
                    np.array(curr_propagated_matching[0]),
                    np.array(curr_propagated_matching[1]),
                    np.array(curr_propagated_matching[2]),
                )
            )

    if verbose:
        print("Done!", flush=True)

    return propagated_matching


def get_matching(
    pivot2_to_pivots1=None,
    propidx2_to_propindices1=None,
    arr1=None,
    arr2=None,
    direction="both",
    target="pivot",
):
    """
    Return a copy of the desired matching.

    Parameters
    ----------
    order: None or (1, 2) or (1, 2), default=None
        If (1, 2), then every cell in target arr1 has at least one match
        if (2, 1), then does the other way around,
        if None, then every cell in target arr1 and every cell in target arr2 both have at least one match
    target: 'pivot' or 'full_data'
        If 'pivot', then only return matching on pivots, else return matching on all the data.

    Returns
    -------
    A matching of format dict or list.
    """
    if target not in {"pivot", "full_data"}:
        raise ValueError("mode must be in {'pivot_only', 'full_data'}.")

    res = [[], [], []]
    for idx2, indices1_and_scores in pivot2_to_pivots1.items():
        for idx1, score in indices1_and_scores:
            res[0].append(idx1)
            res[1].append(idx2)
            res[2].append(score)
    if target == "pivot":
        return res
    elif target == "full_data":
        if direction == "left":
            # add propagated matching for non-pivot cells in the first dataset
            existing_indices1 = np.unique(res[0])
            remaining_indices1 = [
                i for i in range(arr1.shape[0]) if i not in existing_indices1
            ]
            propagated_idx1_to_indices2 = defaultdict(list)
            for idx2, indices1_and_scores in propidx2_to_propindices1.items():
                for idx1, score in indices1_and_scores:
                    propagated_idx1_to_indices2[idx1].append((idx2, score))
            for idx1 in remaining_indices1:
                if idx1 in propagated_idx1_to_indices2:
                    for idx2, score in propagated_idx1_to_indices2[idx1]:
                        res[0].append(idx1)
                        res[1].append(idx2)
                        res[2].append(score)
        elif direction == "right":
            # add propagated matching for non-pivot cells in the second dataset
            existing_indices2 = np.unique(res[1])
            remaining_indices2 = [
                i for i in range(arr2.shape[0]) if i not in existing_indices2
            ]
            for idx2 in remaining_indices2:
                if idx2 in propidx2_to_propindices1:
                    for idx1, score in propidx2_to_propindices1[idx2]:
                        res[0].append(idx1)
                        res[1].append(idx2)
                        res[2].append(score)
        elif direction == "both":
            # first do order (1, 2) and then do order (2, 1)
            # add propagated matching for non-pivot cells in the first dataset
            existing_indices1 = np.unique(res[0])
            remaining_indices1 = [
                i for i in range(arr1.shape[0]) if i not in existing_indices1
            ]
            propagated_idx1_to_indices2 = defaultdict(list)
            for idx2, indices1_and_scores in propidx2_to_propindices1.items():
                for idx1, score in indices1_and_scores:
                    propagated_idx1_to_indices2[idx1].append((idx2, score))
            for idx1 in remaining_indices1:
                if idx1 in propagated_idx1_to_indices2:
                    for idx2, score in propagated_idx1_to_indices2[idx1]:
                        res[0].append(idx1)
                        res[1].append(idx2)
                        res[2].append(score)

            # add propagated matching for non-pivot cells in the second dataset
            existing_indices2 = np.unique(res[1])
            remaining_indices2 = [
                i for i in range(arr2.shape[0]) if i not in existing_indices2
            ]
            for idx2 in remaining_indices2:
                if idx2 in propidx2_to_propindices1:
                    for idx1, score in propidx2_to_propindices1[idx2]:
                        res[0].append(idx1)
                        res[1].append(idx2)
                        res[2].append(score)

        else:
            raise NotImplementedError("direction must be `both` or `left` or `right`.")
    else:
        raise NotImplementedError("target must be in {'pivot', 'full_data'}.")

    return address_matching_redundancy(matching=res, direction=direction)


## Main function
def create_adj(
    rna,
    atac,
    rna_adata_shared,
    atac_adata_shared,
    data_type="Paired",
    rna_n_top_features=3000,
    atac_n_top_features=10000,
    batch_key=None,
    max_outward_size=5000,
    matching_ratio=1,
    batching_scheme='cyclic', # cyclic or pairwise
    seed=42, # None
    svd_components1=30,
    svd_components2=30,
    cca_components=20,
    cca_max_iter=2000,
    randomized_svd=False,
    filter_prop_initial=0,
    filter_prop_refined=0.3,
    filter_prop_propagated=0,
    n_iters=1,
    svd_runs=1,
    verbose=True,
):

    # Finding initial pivots on all genes or peaks
    # 统计计算时间
    # import time
    # start_time = time.time()
    print("Finding initial pivots...")
    initial_pivots = find_initial_pivots(
        rna_adata_shared.X,
        atac_adata_shared.X,
        max_outward_size=max_outward_size,
        matching_ratio=matching_ratio,
        batching_scheme=batching_scheme,
        seed=seed,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
        randomized_svd=randomized_svd,
        svd_runs=svd_runs,
        verbose=verbose,
    )
    # print(
    #     "Finding initial pivots costs {:.2f} seconds.".format(time.time() - start_time)
    # )

    # Finding variable features for RNA adata
    if type(rna_n_top_features) == int:
        if rna_n_top_features > len(rna.var_names):
            rna_n_top_features = len(rna.var_names)
        if batch_key is not None:
            sc.pp.highly_variable_genes(
                rna, n_top_genes=rna_n_top_features, batch_key=batch_key
            )
        else:
            sc.pp.highly_variable_genes(rna, n_top_genes=rna_n_top_features)
            # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        rna = rna[:, rna.var.highly_variable].copy()
    elif type(rna_n_top_features) != int:
        if isinstance(rna_n_top_features, str):
            if os.path.isfile(rna_n_top_features):
                rna_n_top_features = np.loadtxt(rna_n_top_features, dtype=str)
        rna = reindex(rna, rna_n_top_features)

    # Finding variable features for ATAC adata
    if (
        type(atac_n_top_features) == int
        and atac_n_top_features > 0
        and atac_n_top_features < atac.shape[1]
    ):
        if batch_key is not None:
            sc.pp.highly_variable_genes(
                atac,
                n_top_genes=atac_n_top_features,
                batch_key=batch_key,
                inplace=False,
                subset=True,
            )
        else:
            sc.pp.highly_variable_genes(
                atac, n_top_genes=atac_n_top_features, inplace=False, subset=True
            )
    elif type(atac_n_top_features) != int:
        if isinstance(atac_n_top_features, str):
            if os.path.isfile(atac_n_top_features):
                atac_n_top_features = np.loadtxt(atac_n_top_features, dtype=str)
        atac = reindex(atac, atac_n_top_features)

    # Refining pivots
    print("Refining pivots...")
    fine_pivots = refine_pivots(
        initial_pivots,
        rna.X,
        atac.X,
        max_outward_size=max_outward_size,
        matching_ratio=matching_ratio,
        batching_scheme=batching_scheme,
        seed=seed,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
        cca_components=cca_components,
        filter_prop=filter_prop_initial,
        n_iters=n_iters,
        randomized_svd=randomized_svd,
        svd_runs=svd_runs,
        cca_max_iter=cca_max_iter,
        verbose=verbose,
    )

    # Filtering bad matches
    print("Filter_bad_matches on pivots matching...")
    pivot2_to_pivots1, remaining_indices_in_refined_matching = filter_bad_matches(
        rna.X,
        atac.X,
        refined_matching=fine_pivots,
        target="pivot",
        filter_prop=filter_prop_refined,
        max_outward_size=max_outward_size,
        matching_ratio=matching_ratio,
        batching_scheme=batching_scheme,
        seed=seed,
        randomized_svd=randomized_svd,
        svd_runs=svd_runs,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
        cca_components=cca_components,
        cca_max_iter=cca_max_iter,
        verbose=verbose,
    )

    # Propagating matching
    print("Propagating matching...")
    propagated_matching = propagate(
        refined_matching=fine_pivots,
        remaining_indices_in_refined_matching=remaining_indices_in_refined_matching,
        arr1=rna.X,
        arr2=atac.X,
        max_outward_size=max_outward_size,
        matching_ratio=matching_ratio,
        batching_scheme=batching_scheme,
        seed=seed,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
        metric="euclidean",
        randomized_svd=randomized_svd,
        svd_runs=svd_runs,
        verbose=verbose,
    )

    # Final filtering of bad matches
    print("Filter_bad_matches on propagated matching...")
    (
        propidx2_to_propindices1,
        remaining_indices_in_propagated_matching,
    ) = filter_bad_matches(
        rna.X,
        atac.X,
        propagated_matching=propagated_matching,
        target="propagated",
        filter_prop=filter_prop_propagated,
        randomized_svd=randomized_svd,
        max_outward_size=max_outward_size,
        matching_ratio=matching_ratio,
        batching_scheme=batching_scheme,
        seed=seed,
        svd_runs=svd_runs,
        svd_components1=svd_components1,
        svd_components2=svd_components2,
        cca_components=cca_components,
        cca_max_iter=cca_max_iter,
        verbose=verbose,
    )

    # Generating final matching
    final_cellmatching = get_matching(
        pivot2_to_pivots1,
        propidx2_to_propindices1,
        arr1=rna.X,
        arr2=atac.X,
        direction="both",
        target="full_data",
    )
    df = pd.DataFrame(
        list(zip(final_cellmatching[0], final_cellmatching[1], final_cellmatching[2])),
        columns=["mod1_indx", "mod2_indx", "score"],
    )

    # Preparing pivot table and converting to sparse matrix format
    if data_type == "Paired":
        all_mod1_indx = pd.Index(range(rna.shape[0]))
        all_mod2_indx = pd.Index(range(atac.shape[0]))
        pivot_df = (
            df.pivot(index="mod1_indx", columns="mod2_indx", values="score")
            .reindex(index=all_mod1_indx, columns=all_mod2_indx)
            .fillna(0.0)
        )
        sparse_matrix = csr_matrix(pivot_df)
    elif data_type == "UnPaired" or data_type == "multi-modal":
        # df2 = df.copy()
        all_mod1_indx = pd.Index(range(rna.shape[0]))
        all_mod2_indx = pd.Index(range(atac.shape[0]))
        # all_mod1_indx = pd.Index(range((rna.shape[0] + atac.shape[0])))
        # all_mod2_indx = pd.Index(range((rna.shape[0] + atac.shape[0])))
        # df['mod1_indx'] += atac.shape[0]  # 将RNA索引偏移
        # df['mod2_indx'] += rna.shape[0]  # 将ATAC索引偏移
        pivot_df = (
            df.pivot(index="mod1_indx", columns="mod2_indx", values="score")
            .reindex(index=all_mod1_indx, columns=all_mod2_indx)
            .fillna(0.0)
        )

        # df2['mod1_indx'] += atac.shape[0]  # 将RNA索引偏移
        # df2['mod2_indx'] += rna.shape[0]  # 将ATAC索引偏移
        # pivot_df2 = df2.pivot(index='mod1_indx', columns='mod2_indx', values='score').reindex(index=all_mod1_indx, columns=all_mod2_indx).fillna(0.)
        # combined_pivot_df = pivot_df.add(pivot_df2, fill_value=0)

        ## 增加对角线的weight
        # 设置主对角线的值为1
        # np.fill_diagonal(combined_pivot_df.values, 1)

        # compute all possible pairs of distances for rna and atac
        # pearson_correlations_rna = utils.pearson_correlation(rna.X, rna.X)
        # pearson_correlations_atac = utils.pearson_correlation(atac.X, atac.X)
        # # Generate row and column indices
        # row_indices = np.arange(len(pearson_correlations_rna)).astype(int)
        # col_indices = np.arange(len(correlation_values)).astype(int)
        #
        # # Construct the output array with row, col, value
        # output_array = np.column_stack((row_indices, col_indices, pearson_correlations_rna))

        sparse_matrix = csr_matrix(pivot_df)

    return sparse_matrix
