import os
import scanpy as sc
import muon as mu
import anndata as ad

from anndata import AnnData
from muon import MuData

import episcanpy as epi
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import issparse, csr
from sklearn.preprocessing import MaxAbsScaler

# from ._pca import select_pcs_features
from ._utils import gene_scores, TFIDF_LSI
from .adj_construction import create_adj

CHUNK_SIZE = 20000

def batch_scale(adata, use_rep='X', batch_key='batch', chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data

    Parameters
    ----------
    adata
        AnnData
    use_rep
        use '.X' or '.obsm'
    chunk_size
        chunk large data into small chunks

    """
    for b in adata.obs[batch_key].unique():
        idx = np.where(adata.obs[batch_key] == b)[0]
        if use_rep == 'X':
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.X[idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                    adata.X[idx[i * chunk_size:(i + 1) * chunk_size]])
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.obsm[use_rep][idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.obsm[use_rep][idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                    adata.obsm[use_rep][idx[i * chunk_size:(i + 1) * chunk_size]])

    return adata

import logging

logger = logging.getLogger(__name__)

def reindex(adata, genes):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing

    Return
    ------
    AnnData
    """
    # Warning for gene percentage
    user_var_names = adata.var_names
    user_var_names = user_var_names.astype(str)
    try:
        percentage = (len(user_var_names.intersection(genes)) / len(user_var_names)) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(f"WARNING: Query shares {percentage}% of its genes with the reference."
                           "This may lead to inaccuracy in the results.")
    except Exception:
        logger.warning("WARNING: Something is wrong with the reference genes.")

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in genes:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print("Query data is missing expression data of ",
              len(ref_genes_not_in_query),
              " genes which were contained in the reference dataset.")

    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        new_X[:, idx] = adata[:, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def preprocessing_rna(
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        used_hvgs: bool = True,
        used_pca_graph: bool = False,
        rna_n_top_features=2000,  # or gene list
        n: int = 15,
        batch_key: str = 'batch',
        metric: str = 'euclidean',
        svd_solver: str = 'arpack',
        keep_mt: bool = False,
        backed: bool = False
):
    """
    Preprocessing single-cell RNA-seq data

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 600.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    rna_n_top_features
        Number of highly-variable genes to keep. Default: 2000.

    Return
    -------
    The AnnData object after preprocessing.
    """
    if min_features is None: min_features = 600
    if rna_n_top_features is None: rna_n_top_features = 2000
    if target_sum is None: target_sum = 10000

    # Preprocessing
    # if not issparse(adata.X):
    if type(adata.X) != csr.csr_matrix and (not backed) and (not adata.isbacked):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    if not keep_mt:
        # Filtering out MT genes
        adata = adata[:, [gene for gene in adata.var_names
                          if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    # Filtering cells
    sc.pp.filter_cells(adata, min_genes=min_features)

    # Filtering features
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalizing total per cell
    if adata.X.max() < 50:
        print('Warning: adata.X may have already been normalized, do not normalize, please check.')
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # Log1p transforming
        sc.pp.log1p(adata)

    # Finding variable features for RNA adata
    adata_hvg = adata.copy()
    if used_hvgs:
        if type(rna_n_top_features) == int:
            if rna_n_top_features > len(adata_hvg.var_names):
                rna_n_top_features = len(adata_hvg.var_names)
            if batch_key is not None:
                sc.pp.highly_variable_genes(adata_hvg, n_top_genes=rna_n_top_features, batch_key=batch_key)
            else:
                sc.pp.highly_variable_genes(adata_hvg, n_top_genes=rna_n_top_features)
                # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()
        elif type(rna_n_top_features) != int:
            if isinstance(rna_n_top_features, str):
                if os.path.isfile(rna_n_top_features):
                    rna_n_top_features = np.loadtxt(rna_n_top_features, dtype=str)
            adata_hvg = reindex(adata_hvg, rna_n_top_features)

    # scale data, clip values exceeding standard deviation 10.
    # if adata.X.min() < 0:
    #     print('Warning: adata.X may have already been scaled, do not scale, please check.')
    # else:
    #     sc.pp.scale(adata, max_value=10)
    # PCA
    sc.tl.pca(adata_hvg, svd_solver=svd_solver)

    if used_pca_graph:
        # use scanpy functions to do the graph construction
        sc.pp.neighbors(adata_hvg, n_neighbors=n, metric=metric, use_rep='X_pca')

    return adata, adata_hvg

def preprocessing_atac(adata: AnnData,
                       data_type: str = None,
                       genome: str = None,
                       use_gene_weigt: bool = True,
                       use_top_pcs: bool = False,
                       used_binarize: bool = False,
                       used_hvgs: bool = True,
                       used_lsi_graph: bool = False,
                       min_features: int = 100,
                       min_cells: int = 3,
                       atac_n_top_features=100000,  # or gene list
                       n: int = 15,
                       batch_key: str = 'batch',
                       metric: str = 'euclidean',
                       n_components: int = 50
                       ):
    """
    Preprocess scCAS data matrix.

    Parameters
    ----------
    adata :  AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks/regions.
    filter_rate : float, optional
        Proportion for feature selection, by default 0.01
    """
    if min_features is None: min_features = 100
    if atac_n_top_features is None: atac_n_top_features = 100000

    # Preprocessing
    # if type(adata.X) != csr.csr_matrix:
    #     adata.X = scipy.sparse.csr_matrix(adata.X)

    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata)
    # adata.raw = adata

    ## TFIDF & LSI
    TFIDF_LSI(adata, n_comps=n_components, binarize=used_binarize, random_state=0)

    if used_lsi_graph:
        # use scanpy functions to do the graph construction
        sc.pp.neighbors(adata, n_neighbors=n, metric=metric, use_rep='X_lsi')

    ## HVP
    adata_hvg = adata.copy()
    if used_hvgs:
        if type(atac_n_top_features) == int and atac_n_top_features > 0 and atac_n_top_features < adata_hvg.shape[1]:
            sc.pp.highly_variable_genes(adata_hvg, n_top_genes=atac_n_top_features, batch_key=batch_key,
                                        inplace=False, subset=True)
        elif type(atac_n_top_features) != int:
            if isinstance(atac_n_top_features, str):
                if os.path.isfile(atac_n_top_features):
                    atac_n_top_features = np.loadtxt(atac_n_top_features, dtype=str)
            adata_hvg = reindex(adata_hvg, atac_n_top_features)

    return adata, adata_hvg


### preprocessing main function
def preprocessing(
        adata: [AnnData, MuData],
        profile: str = 'RNA',
        data_type: str = 'Paired',
        batch_key: str = 'batch',
        weight = 0.8,
        genome: str = None,
        use_gene_weigt: bool = True,
        use_top_pcs: bool = False,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = None,
        rna_n_top_features=None,  # or gene list
        atac_n_top_features=None,  # or gene list
        svd_components_rna: int = 30,
        svd_components_atac: int = 30,
        cca_components: int = 20,
        cca_max_iter: int = 2000,
        randomized_svd: bool = False,
        filter_prop_initial: int = 0,
        filter_prop_refined: int = 0.3,
        filter_prop_propagated: int = 0,
        n_iters: int = 1,
        svd_runs: int = 1,
        n: int = 15,
        metric: str = 'euclidean',
        svd_solver: str = 'arpack',
        n_components: int = 50,
        keep_mt: bool = False,
        backed: bool = False,
        verbose: bool = True
):
    """
    Preprocessing single-cell data

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    profile
        Specify the single-cell profile type, RNA or ATAC, Default: RNA.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 100.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    log
        If log, record each operation in the log file. Default: None.

    Return
    -------
    The AnnData object after preprocessing.

    """
    if profile == 'RNA':
        return preprocessing_rna(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            used_hvgs=True,
            used_pca_graph=True,
            rna_n_top_features=rna_n_top_features,
            n=n,
            batch_key=batch_key,
            metric=metric,
            svd_solver=svd_solver,
            keep_mt=keep_mt,
            backed=backed,
        )
    elif profile == 'ATAC':
        return preprocessing_atac(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            used_hvgs=True,
            used_lsi_graph=True,
            atac_n_top_features=atac_n_top_features,
            n=n,
            batch_key=batch_key,
            metric=metric,
            n_components=n_components
        )
    elif profile == 'multi-modal':
        # UnPaired TO DO
        assert data_type in ['Paired', 'UnPaired'], 'Data_type must be "Paired", or "UnPaired".'
        rna_adata = adata.mod['rna']
        atac_adata = adata.mod['atac']

        if data_type == 'Paired':
            rna_adata, rna_adata_hvg = preprocessing_rna(
                rna_adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                used_hvgs=True,
                used_pca_graph=True,
                rna_n_top_features=rna_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=keep_mt,
                backed=backed,
            )
            atac_adata, atac_adata_hvg = preprocessing_atac(
                atac_adata,
                data_type,
                min_features=min_features,
                min_cells=min_cells,
                used_hvgs=True,
                used_lsi_graph=True,
                atac_n_top_features=atac_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                n_components=n_components
            )
            ## Concatenating different modalities
            adata_paired = ad.concat([rna_adata_hvg, atac_adata_hvg], axis=1)

            ## the .obs layer is empty now, and we need to repopulate it
            rna_cols = rna_adata.obs.columns
            atac_cols = atac_adata.obs.columns

            rnaobs = rna_adata.obs.copy()
            rnaobs.columns = ["rna:" + x for x in rna_cols]
            atacobs = atac_adata.obs.copy()
            atacobs.columns = ["atac:" + x for x in atac_cols]
            adata_paired.obs = pd.merge(rnaobs, atacobs, left_index=True, right_index=True)

            ## 先将 scATAC 转换为基因活性矩阵
            print('Convert peak to gene activity matrix, please wait.', flush=True)
            print('`genome` parameter should be set correctly', flush=True)
            print("Choose from {‘hg19’, ‘hg38’, ‘mm9’, ‘mm10’}", flush=True)
            adata_CG_atac = gene_scores(atac_adata, genome=genome, use_gene_weigt=use_gene_weigt, use_top_pcs=use_top_pcs)

            ## 交集
            common_genes = set(rna_adata.var_names).intersection(set(adata_CG_atac.var_names))
            print('There are {} common genes in RNA and ATAC datasets'.format(len(common_genes)))
            rna_adata_shared = rna_adata[:, list(common_genes)]
            atac_adata_shared = adata_CG_atac[:, list(common_genes)]

            # 通过cell matching 构建组学间的图结构
            print('To start performing cell matching for adjacency matrix of the graph, please wait...', flush=True)
            # adata_paired.obsp['connectivities']
            inter_connect = create_adj(rna_adata, atac_adata, rna_adata_shared, atac_adata_shared,
                                                             data_type=data_type,
                                                             rna_n_top_features=rna_n_top_features,
                                                             atac_n_top_features=atac_n_top_features,
                                                             batch_key=batch_key, svd_components1=svd_components_rna,
                                                             svd_components2=svd_components_atac,
                                                             cca_components=cca_components, cca_max_iter=cca_max_iter,
                                                             randomized_svd=randomized_svd,
                                                             filter_prop_initial=filter_prop_initial,
                                                             filter_prop_refined=filter_prop_refined,
                                                             filter_prop_propagated=filter_prop_propagated,
                                                             n_iters=n_iters, svd_runs=svd_runs, verbose=verbose)

            # 你可以在一个字典中保留原始的.obsp数据
            # 设置权重
            # w = weight #0.8  # 你可以根据实际需要调整这个权重
            # Iterate over keys in rna_obsp (assuming atac_obsp has the same keys)
            for key in rna_adata_hvg.obsp.keys():
                # 计算加权平均的连通性矩阵
                intra_connect = weight * rna_adata_hvg.obsp[key] + (1 - weight) * atac_adata_hvg.obsp[key]
                # adata_paired.obsp[key + '_combined'] = combined_obsp

            adata_paired.obsp['connectivities'] = inter_connect + intra_connect
            # adata_paired.uns['obsp_rna'] = rna_adata.obsp
            # adata_paired.uns['obsp_atac'] = atac_adata.obsp

            return adata_paired
    else:
        raise ValueError("Not support profile: `{}` yet".format(profile))