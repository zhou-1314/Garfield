"""
This module contains helper functions for the ´models´ subpackage.
"""

import logging
import os
import pickle
import dill
from collections import OrderedDict
from typing import Optional, Tuple, Literal

from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse import isspmatrix_csr
from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata
import anndata as ad
from anndata import AnnData, concat
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import KNeighborsTransformer

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_model_with_fallback(model, file_path):
    try:
        # 尝试使用 pickle 保存
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved successfully using pickle at {file_path}")
    except (AttributeError, pickle.PicklingError) as e:
        # 如果 pickle 保存失败，捕获异常并使用 joblib
        print(f"Pickle failed with error: {e}, switching to dill...")
        with open(file_path, "wb") as f:
            dill.dump(model, f)
        print(f"Model saved successfully using dill at {file_path}")


def load_model_with_fallback(file_path):
    try:
        # 尝试使用 pickle 加载
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully using pickle from {file_path}")
    except (AttributeError, pickle.UnpicklingError, EOFError) as e:
        # 如果 pickle 加载失败，捕获异常并使用 joblib
        print(f"Pickle failed with error: {e}, switching to dill...")
        with open(file_path, "rb") as f:
            model = dill.load(f)
        print(f"Model loaded successfully using dill from {file_path}")

    return model


def load_saved_files(
    dir_path: str,
    query_adata: Optional[AnnData] = None,
    ref_adata_name: str = "adata_ref.h5ad",
    batch_key: Optional[str] = None,
    map_location: Optional[Literal["cpu", "cuda"]] = None,
) -> Tuple[OrderedDict, dict, np.ndarray, ad.AnnData]:
    """
    Helper to load saved model files.

    Parts of the implementation are adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_utils.py#L55
    (01.10.2022)

    Parameters
    ----------
    dir_path:
        Path where the saved model files are stored.
    query_adata:
        Query anndata object.
    ref_adata_name:
        Name of the reference anndata object.
    batch_key:
        Batch key for the reference anndata object.
    map_location:
        Memory location where to map the model files to.

    Returns
    ----------
    model_state_dict:
        The stored model state dict.
    var_names:
        The stored variable names.
    attr_dict:
        The stored attributes.
    adata_concat:
        The concatenated anndata object.
    """
    attr_path = os.path.join(dir_path, "attr.pkl")
    adata_path = os.path.join(dir_path, ref_adata_name)
    var_names_path = os.path.join(dir_path, "var_names.csv")
    model_path = os.path.join(dir_path, "model_params.pt")

    if os.path.exists(adata_path):
        adata_ref = ad.read(adata_path)
        adata_ref.X = (
            adata_ref.layers["counts"].copy()
            if "counts" in adata_ref.layers.keys()
            else adata_ref.X
        )
    else:
        raise ValueError("Dir path contains no saved reference anndata")

    var_names = np.genfromtxt(var_names_path, delimiter=",", dtype=str)
    if query_adata is not None:
        query_adata = validate_var_names(query_adata, var_names)

        if batch_key is None:
            raise ValueError("batch_key is required when query_adata is provided.")

        # 检查 batch_key 是否存在于 adata_ref.obs
        if batch_key not in adata_ref.obs:
            raise ValueError(f"The column '{batch_key}' does not exist in adata_ref.obs.")

        # 给 query_adata 添加 batch_key 并标记为新 batch
        if batch_key not in query_adata.obs:
            new_batch_label = "new_batch"  # 可以根据需要动态设置
            query_adata.obs[batch_key] = new_batch_label

        # 合并数据集
        adata_concat = anndata.concat(
            [adata_ref, query_adata],
            label="projection",
            keys=["reference", "query"],
            index_unique=None,
            join="outer",
        )
        # 验证合并后的结果是否包含 sample_col
        if batch_key not in adata_concat.obs:
            raise ValueError(f"The column '{batch_key}' is missing in adata_concat.obs.")

        del adata_concat.obsm['garfield_latent'] # remove garfield_latent
    else:
        adata_concat = adata_ref

    model_state_dict = torch.load(model_path, map_location=map_location)
    attr_dict = load_model_with_fallback(attr_path)
    # with open(attr_path, "rb") as handle:
    #     attr_dict = pickle.load(handle)
    return model_state_dict, var_names, attr_dict, adata_concat


def validate_var_names(adata, source_var_names):
    # Warning for gene percentage
    user_var_names = adata.var_names
    try:
        percentage = (
            len(user_var_names.intersection(source_var_names)) / len(user_var_names)
        ) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(
                f"WARNING: Query shares {percentage}% of its genes with the reference."
                "This may lead to inaccuracy in the results."
            )
    except Exception:
        logger.warning("WARNING: Something is wrong with the reference genes.")

    user_var_names = user_var_names.astype(str)
    new_adata = adata

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in source_var_names:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print(
            "Query data is missing expression data of ",
            len(ref_genes_not_in_query),
            " genes which were contained in the reference dataset.",
        )
        print("The missing information will be filled with zeroes.")

        filling_X = np.zeros((len(adata), len(ref_genes_not_in_query)))
        if isinstance(adata.X, csr_matrix):
            filling_X = csr_matrix(filling_X)  # support csr sparse matrix
            new_target_X = hstack((adata.X, filling_X))
        else:
            new_target_X = np.concatenate((adata.X, filling_X), axis=1)
        new_target_vars = adata.var_names.tolist() + ref_genes_not_in_query
        new_adata = AnnData(new_target_X, dtype="float32")
        new_adata.var_names = new_target_vars
        new_adata.obs = adata.obs.copy()

    if len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)) > 0:
        print(
            "Query data contains expression data of ",
            len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)),
            " genes that were not contained in the reference dataset. This information "
            "will be removed from the query data object for further processing.",
        )

        # remove unseen gene information and order anndata
        new_adata = new_adata[:, source_var_names].copy()

    print(new_adata)
    return new_adata


def weighted_knn_trainer(train_adata, train_adata_emb, n_neighbors=50):
    """
    Trains a weighted KNN classifier on ``train_adata``.

    Parameters
    ----------
    train_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
    train_adata_emb: str
        Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be
        used
    n_neighbors: int
        Number of nearest neighbors in KNN classifier.
    """
    print(f"Weighted KNN with n_neighbors = {n_neighbors} ... ")
    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm.keys():
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise ValueError(
            "train_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    k_neighbors_transformer.fit(train_emb)
    return k_neighbors_transformer


def weighted_knn_transfer(
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    knn_model,
    threshold=1,
    pred_unknown=False,
    mode="package",
):
    """
    Annotates ``query_adata`` cells with an input trained weighted KNN classifier.

    Parameters
    ----------
    query_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    query_adata_emb: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: :class:`pd.DataFrame`
        obs of ref Anndata
    label_keys: str
        Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    knn_model: :class:`~sklearn.neighbors._graph.KNeighborsTransformer`
        knn model trained on reference adata with weighted_knn_trainer function
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown: bool
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    mode: str
        Has to be one of "paper" or "package". If mode is set to "package",
        uncertainties will be 1 - P(pred_label), otherwise it will be 1 - P(true_label).
    """
    if not type(knn_model) == KNeighborsTransformer:
        raise ValueError(
            "knn_model should be of type sklearn.neighbors._graph.KNeighborsTransformer!"
        )

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            "query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )
    cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]

    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    for i in range(len(weights)):
        for j in cols:
            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0
            for candidate_label in unique_labels:
                candidate_prob = weights[
                    i, y_train_labels[top_k_indices[i]] == candidate_label
                ].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label

            if pred_unknown:
                if best_prob >= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label

            if mode == "package":
                uncertainties.iloc[i][j] = max(1 - best_prob, 0)

            else:
                raise Exception("Inquery Mode!")

            pred_labels.iloc[i][j] = pred_label

    print("Label transfer finished!", flush=True)

    return pred_labels, uncertainties


