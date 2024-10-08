import os
import scanpy as sc
import anndata as ad

from anndata import AnnData
from muon import MuData

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import issparse, csr
from sklearn.preprocessing import MaxAbsScaler

# from ._pca import select_pcs_features
from ._utils import reindex, gene_scores, TFIDF_LSI, clr_normalize_each_cell
from .adj_construction import create_adj, graph_construction

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


def preprocessing_rna(
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        used_hvgs: bool = True,
        used_pca_graph: bool = True,
        rna_n_top_features=2000,  # or gene list
        n_components: int = 50,
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
        print('Warning: adata_rna.X may have already been normalized, do not normalize, please check.')
        adata.layers['norm_data'] = adata.X.copy()
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # Log1p transforming
        sc.pp.log1p(adata)
        adata.layers['norm_data'] = adata.X.copy()

    # Finding variable features for RNA adata
    if used_hvgs:
        if type(rna_n_top_features) == int:
            if rna_n_top_features > len(adata.var_names):
                rna_n_top_features = len(adata.var_names)
            if batch_key is not None:
                sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features, batch_key=batch_key)
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features)
                # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        elif type(rna_n_top_features) != int:
            if isinstance(rna_n_top_features, str):
                if os.path.isfile(rna_n_top_features):
                    rna_n_top_features = np.loadtxt(rna_n_top_features, dtype=str)
            adata = reindex(adata, rna_n_top_features)

    ## only HVGs
    use_highly_variable = "highly_variable" in adata.var
    adata_hvg = adata[:, adata.var["highly_variable"]].copy() if use_highly_variable else adata.copy()

    # scale data, clip values exceeding standard deviation 10.
    if adata_hvg.X.min() < 0:
        print('Warning: adata.X may have already been scaled, do not scale, please check.')
    else:
        sc.pp.scale(adata_hvg, max_value=10)
    # PCA
    sc.tl.pca(adata_hvg, svd_solver=svd_solver, n_comps=n_components)
    # 返回norm_data
    adata_hvg.X = adata_hvg.layers['norm_data'].copy() if "norm_data" in adata_hvg.layers.keys() else adata_hvg.X.copy()
    adata_hvg.obsm['feat'] = adata_hvg.obsm['X_pca'].copy()

    if used_pca_graph:
        # use scanpy functions to do the graph construction
        sc.pp.neighbors(adata_hvg, n_neighbors=n, metric=metric, use_rep='X_pca')

    return adata, adata_hvg

def preprocessing_atac(adata: AnnData,
                       used_hvgs: bool = True,
                       used_lsi_norm: bool = False,
                       used_lsi_graph: bool = True,
                       min_features: int = 100,
                       min_cells: int = 3,
                       atac_n_top_features=100000,  # or gene list
                       n: int = 6,
                       batch_key: str = 'batch',
                       metric: str = 'euclidean',
                       n_components: int = 50
                       ):
    """
    Preprocess scATAC data matrix.

    Parameters
    ----------
    adata :  AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks/regions.
    filter_rate : float, optional
        Proportion for feature selection, by default 0.01
    """
    if min_features is None: min_features = 100
    if min_cells is None: min_cells = 3
    if atac_n_top_features is None: atac_n_top_features = 10000

    # Preprocessing
    if type(adata.X) != csr.csr_matrix:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    sc.pp.filter_cells(adata, min_genes=min_features)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # adata.raw = adata

    ## TFIDF & LSI
    if used_lsi_norm:
        TFIDF_LSI(adata, n_comps=n_components, binarize=False, random_state=0)
        adata.layers['norm_data'] = adata.X.copy()
    else:
        # Normalizing total per cell
        if adata.X.max() < 2:
            print('Warning: adata_rna.X may have already been normalized, do not normalize, please check.')
            adata.layers['norm_data'] = adata.X.copy()
        else:
            sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
            # sc.pp.normalize_total(adata)
            # Log1p transforming
            sc.pp.log1p(adata)
            adata.layers['norm_data'] = adata.X.copy()

    # Finding variable features for ATAC adata
    if used_hvgs:
        if type(atac_n_top_features) == int:
            if atac_n_top_features > len(adata.var_names):
                atac_n_top_features = len(adata.var_names)
            if batch_key is not None:
                sc.pp.highly_variable_genes(adata, n_top_genes=atac_n_top_features, batch_key=batch_key)
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=atac_n_top_features)
                # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        elif type(atac_n_top_features) != int:
            if isinstance(atac_n_top_features, str):
                if os.path.isfile(atac_n_top_features):
                    atac_n_top_features = np.loadtxt(atac_n_top_features, dtype=str)
            adata = reindex(adata, atac_n_top_features)

    ## only HVGs
    use_highly_variable = "highly_variable" in adata.var
    adata_hvg = adata[:, adata.var["highly_variable"]].copy() if use_highly_variable else adata.copy()

    if used_lsi_graph:
        if not used_lsi_norm:
            # scale data, clip values exceeding standard deviation 10.
            if adata_hvg.X.min() < 0:
                print('Warning: adata.X may have already been scaled, do not scale, please check.')
            else:
                sc.pp.scale(adata_hvg, max_value=10)
            # PCA
            sc.tl.pca(adata_hvg, n_comps=n_components)
            # 返回norm_data
            adata_hvg.X = adata_hvg.layers['norm_data'].copy()
            adata_hvg.obsm['feat'] = adata_hvg.obsm['X_pca'].copy()

            # use scanpy functions to do the graph construction
            sc.pp.neighbors(adata_hvg, n_neighbors=n, metric=metric, use_rep='X_pca')
        else:
            # 返回norm_data
            adata_hvg.X = adata_hvg.layers['norm_data'].copy() if "norm_data" in adata_hvg.layers.keys() else adata_hvg.X.copy()
            adata_hvg.obsm['feat'] = adata_hvg.obsm['X_lsi'].copy()

            # use scanpy functions to do the graph construction
            sc.pp.neighbors(adata_hvg, n_neighbors=n, metric=metric, use_rep='X_lsi')

    return adata, adata_hvg

def preprocessing_adt(
        adata: AnnData,
        used_hvgs: bool = False,
        used_pca_graph: bool = True,
        rna_n_top_features=2000,  # or gene list
        n: int = 15,
        batch_key: str = 'batch',
        metric: str = 'euclidean',
        svd_solver: str = 'arpack',
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
    # Preprocessing

    # if not issparse(adata.X):
    if type(adata.X) != csr.csr_matrix and (not backed) and (not adata.isbacked):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # Normalizing total per cell
    if adata.X.max() < 50:
        print('Warning: adata_adt.X may have already been normalized, do not normalize, please check.')
    else:
        adata = clr_normalize_each_cell(adata)
        adata.layers['norm_data'] = adata.X.copy()

    # Finding variable features for RNA adata
    if used_hvgs:
        if type(rna_n_top_features) == int:
            if rna_n_top_features > len(adata.var_names):
                rna_n_top_features = len(adata.var_names)
            if batch_key is not None:
                sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features, batch_key=batch_key)
            else:
                sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features)
                # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        elif type(rna_n_top_features) != int:
            if isinstance(rna_n_top_features, str):
                if os.path.isfile(rna_n_top_features):
                    rna_n_top_features = np.loadtxt(rna_n_top_features, dtype=str)
            adata = reindex(adata, rna_n_top_features)

    ## only HVGs
    use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]].copy() if use_highly_variable else adata.copy()

    if used_pca_graph:
        # scale data, clip values exceeding standard deviation 10.
        if adata_use.X.min() < 0:
            print('Warning: adata.X may have already been scaled, do not scale, please check.')
        else:
            sc.pp.scale(adata_use, max_value=10)
        # PCA
        sc.tl.pca(adata_use, svd_solver=svd_solver)
        # use scanpy functions to do the graph construction
        sc.pp.neighbors(adata_use, n_neighbors=n, metric=metric, use_rep='X_pca')
        # 返回norm_data
        if adata.X.max() < 50:
            pass
        else:
            adata_use.X = adata_use.layers['norm_data'].copy() if "norm_data" in adata_use.layers.keys() else adata_use.X.copy()
        adata_use.obsm['feat'] = adata_use.obsm['X_pca'].copy()
    else:
        sc.pp.neighbors(adata_use, n_neighbors=n, metric=metric, use_rep='X')

    return adata, adata_use

### preprocessing main function
def preprocessing(
        adata: [AnnData, MuData],
        profile: str = 'RNA',
        data_type: str = 'Paired',
        sub_data_type: str = ['rna', 'atac'],
        batch_key: str = 'batch',
        weight = 0.8,
        used_hvgs: bool = True,
        graph_const_method: str = None,
        genome: str = None,
        use_gene_weigt: bool = True,
        use_top_pcs: bool = False,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = None,
        rna_n_top_features=None,  # or gene list
        atac_n_top_features=None,  # or gene list
        n_components: int = 50,
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
            used_hvgs=used_hvgs,
            used_pca_graph=True,
            rna_n_top_features=rna_n_top_features,
            n_components=n_components,
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
            used_hvgs=used_hvgs,
            used_lsi_graph=True,
            atac_n_top_features=atac_n_top_features,
            n_components=n_components,
            n=n,
            batch_key=batch_key,
            metric=metric
        )
    elif profile == 'ADT':
        return preprocessing_adt(
            adata,
            used_hvgs=used_hvgs,
            used_pca_graph=True,
            rna_n_top_features=rna_n_top_features,  # or gene list
            n=n,
            batch_key=batch_key,
            metric=metric,
            svd_solver=svd_solver,
            backed=backed
        )
    elif profile == 'multi-modal':
        # UnPaired TO DO
        assert data_type in ['Paired', 'UnPaired'], 'Data_type must be "Paired", or "UnPaired".'
        assert sub_data_type[0] in ['rna', 'atac', 'adt'] and sub_data_type[1] in ['rna', 'atac', 'adt'], \
            'Sub_data_type must be "rna", "atac", or "adt".'

        if len(sub_data_type) == 2:
            if sub_data_type[0] == 'rna' and sub_data_type[1] == 'atac':
                rna_adata = adata.mod['rna']
                atac_adata = adata.mod['atac']
                adt_adata = None
            elif sub_data_type[0] == 'rna' and sub_data_type[1] == 'adt':
                rna_adata = adata.mod['rna']
                atac_adata = None
                adt_adata = adata.mod['adt']
        else:
            ValueError('The length of sub_data_type must be 2, such as: ["rna", "atac"] or ["rna", "adt"].')

        if data_type == 'Paired':
            rna_adata, rna_adata_hvg = preprocessing_rna(
                rna_adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                used_hvgs=used_hvgs,
                used_pca_graph=True,
                rna_n_top_features=rna_n_top_features,
                n_components=n_components,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=keep_mt,
                backed=backed,
            )
            if atac_adata is not None:
                atac_adata, atac_adata_hvg = preprocessing_atac(
                    atac_adata,
                    min_features=min_features,
                    min_cells=min_cells,
                    used_hvgs=used_hvgs,
                    used_lsi_graph=True,
                    atac_n_top_features=atac_n_top_features,
                    n_components=n_components,
                    n=n,
                    batch_key=batch_key,
                    metric=metric
                )
                ## Concatenating different modalities
                adata_paired = ad.concat([rna_adata_hvg, atac_adata_hvg], axis=1)
                # 假设 obsm 的每个值都是形状相同的矩阵，可以通过 np.hstack 或其他方法进行合并
                rna_obsm = rna_adata_hvg.obsm
                atac_obsm = atac_adata_hvg.obsm
                # 只需要 feat key
                # combined_obsm = {key: np.hstack((rna_obsm[key], atac_obsm[key])) for key in rna_obsm.keys()}
                combined_obsm = np.hstack((rna_obsm['feat'], atac_obsm['feat']))
                # 将合并后的 obsm 信息添加到 adata_paired，但先确保索引匹配
                adata_paired.obsm['feat'] = pd.DataFrame(
                    combined_obsm,
                    index=adata_paired.obs.index  # 确保索引一致
                )

                ## the .obs layer is empty now, and we need to repopulate it
                rna_cols = rna_adata.obs.columns
                atac_cols = atac_adata.obs.columns

                rnaobs = rna_adata.obs.copy()
                rnaobs.columns = ["rna:" + x for x in rna_cols]
                atacobs = atac_adata.obs.copy()
                atacobs.columns = ["atac:" + x for x in atac_cols]
                adata_paired.obs = pd.merge(rnaobs, atacobs, left_index=True, right_index=True)

                ## 先将 scATAC 转换为基因活性矩阵
                # 定义保存文件的路径
                cache_path = "adata_ATAC_cache.h5ad"
                # 检查是否已经存在缓存文件
                if os.path.exists(cache_path):
                    # 如果缓存文件存在，直接加载
                    print("Gene activity matrix has been calculated, and loading cached adata_CG_atac object...")
                    adata_CG_atac = sc.read_h5ad(cache_path)
                else:
                    # 如果缓存文件不存在，执行耗时计算生成新的 adata
                    print('Convert peak to gene activity matrix, this might take a while...', flush=True)
                    print('`genome` parameter should be set correctly', flush=True)
                    print("Choose from {‘hg19’, ‘hg38’, ‘mm9’, ‘mm10’}", flush=True)
                    # 生成新的 adata_CG_atac 对象
                    adata_CG_atac = gene_scores(atac_adata, genome=genome, use_gene_weigt=use_gene_weigt, use_top_pcs=use_top_pcs)
                    # 将 adata_new 保存到缓存文件中
                    adata_CG_atac.write_h5ad(cache_path)
                    print(f"adata object cached at: {cache_path}")

                ## 交集
                common_genes = set(rna_adata.var_names).intersection(set(adata_CG_atac.var_names))
                print('There are {} common genes in RNA and ATAC datasets'.format(len(common_genes)))
                rna_adata_shared = rna_adata[:, list(common_genes)]
                atac_adata_shared = adata_CG_atac[:, list(common_genes)]

                # 通过cell matching 构建组学间的图结构
                print('To start performing cell matching for adjacency matrix of the graph, please wait...', flush=True)
                inter_connect = create_adj(rna_adata,
                                           atac_adata,
                                           rna_adata_shared,
                                           atac_adata_shared,
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

                # 设置权重
                # Iterate over keys in rna_obsp (assuming atac_obsp has the same keys)
                for key in rna_adata_hvg.obsp.keys():
                    # 计算加权平均的连通性矩阵
                    intra_connect = rna_adata_hvg.obsp[key] + atac_adata_hvg.obsp[key]

                adata_paired.obsp['connectivities'] = weight * inter_connect + (1 - weight) * intra_connect

                return adata_paired

            elif adt_adata is not None:
                adt_adata, adt_adata_hvg = preprocessing_adt(
                    adt_adata,
                    used_hvgs=used_hvgs,
                    used_pca_graph=True,
                    rna_n_top_features=rna_n_top_features,  # or gene list
                    n=n,
                    batch_key=batch_key,
                    metric=metric,
                    svd_solver=svd_solver,
                    backed=backed
                )
                ## Concatenating different modalities
                adata_paired = ad.concat([rna_adata_hvg, adt_adata_hvg], axis=1)

                ## the .obs layer is empty now, and we need to repopulate it
                rna_cols = rna_adata.obs.columns
                adt_cols = adt_adata.obs.columns

                rnaobs = rna_adata.obs.copy()
                rnaobs.columns = ["rna:" + x for x in rna_cols]
                adtobs = adt_adata.obs.copy()
                adtobs.columns = ["adt:" + x for x in adt_cols]
                adata_paired.obs = pd.merge(rnaobs, adtobs, left_index=True, right_index=True)

                ## 交集
                common_genes = set(rna_adata.var_names).intersection(set(adt_adata.var_names))
                print('There are {} common genes in RNA and ADT datasets'.format(len(common_genes)))
                rna_adata_shared = rna_adata[:, list(common_genes)]
                adt_adata_shared = adt_adata[:, list(common_genes)]

                ## 通过cell matching 构建组学间的图结构
                print('To start performing cell matching for adjacency matrix of the graph, please wait...', flush=True)
                inter_connect = create_adj(rna_adata,
                                           adt_adata,
                                           rna_adata_shared,
                                           adt_adata_shared,
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

                # Iterate over keys in rna_obsp (assuming adt_obsp has the same keys)
                for key in rna_adata_hvg.obsp.keys():
                    # 计算加权平均的连通性矩阵
                    intra_connect = rna_adata_hvg.obsp[key] + adt_adata_hvg.obsp[key]

                adata_paired.obsp['connectivities'] = weight * inter_connect + (1 - weight) * intra_connect

                return adata_paired

    elif profile == 'spatial':
        assert data_type in ['single-modal', 'multi-modal'], 'Data_type must be "single-modal", or "multi-modal".'

        if data_type == 'single-modal':
            _, rna_adata_hvg = preprocessing_rna(
                adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                used_hvgs=used_hvgs,
                used_pca_graph=True,
                rna_n_top_features=rna_n_top_features,
                n_components=n_components,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=True,
                backed=backed,
            )
            # Construct Spatial Graph
            if graph_const_method == 'mu_std':
                spatial_adj = graph_construction(adata, mode='mu_std', k=n, batch_key=batch_key)
                # spatial_adj = graph_dict #.toarray()
            elif graph_const_method == 'Radius':
                spatial_adj = graph_construction(adata, mode='Radius', k=n, batch_key=batch_key)
            elif graph_const_method == 'KNN':
                spatial_adj = graph_construction(adata, mode='KNN', k=n, batch_key=batch_key)
            elif graph_const_method == 'Squidpy':
                import squidpy as sq
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n)
                # Make adjacency matrix symmetric
                adata.obsp['spatial_connectivities'] = (
                    adata.obsp['spatial_connectivities'].maximum(
                        adata.obsp['spatial_connectivities'].T))
                spatial_adj = adata.obsp['spatial_connectivities'] #.toarray()

            # Ensure adj is a csr_matrix
            if not isinstance(spatial_adj, csr_matrix):
                spatial_adj = csr_matrix(spatial_adj)

            expr_adj = rna_adata_hvg.obsp['connectivities']
            # Ensure adj is a csr_matrix
            if not isinstance(expr_adj, csr_matrix):
                expr_adj = csr_matrix(expr_adj)

            # Validate adjacency matrix symmetry
            if (spatial_adj.getnnz() != spatial_adj.T.getnnz()):
                raise ImportError("The spatial_adj adjacency matrix has to be symmetric.")
            # Validate adjacency matrix symmetry
            if (expr_adj.getnnz() != expr_adj.T.getnnz()):
                raise ImportError("The expr_adj adjacency matrix has to be symmetric.")

            # 计算加权平均的连通性矩阵
            rna_adata_hvg.obsp['spatial_connectivities'] = spatial_adj
            rna_adata_hvg.obsp['connectivities'] = weight * spatial_adj + (1 - weight) * expr_adj

            return rna_adata_hvg

        elif data_type == 'multi-modal':
            assert sub_data_type[0] in ['rna', 'atac', 'adt'] and sub_data_type[1] in ['rna', 'atac', 'adt'], 'Sub_data_type must be "rna", "atac", or "adt".'

            if len(sub_data_type) == 2:
                if sub_data_type[0] == 'rna' and sub_data_type[1] == 'atac':
                    rna_adata = adata.mod['rna']
                    atac_adata = adata.mod['atac']
                    adt_adata = None
                elif sub_data_type[0] == 'rna' and sub_data_type[1] == 'adt':
                    rna_adata = adata.mod['rna']
                    atac_adata = None
                    adt_adata = adata.mod['adt']
            else:
                ValueError('The length of sub_data_type must be 2, such as: ["rna", "atac"] or ["rna", "adt"].')

            rna_adata, rna_adata_hvg = preprocessing_rna(
                rna_adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                used_hvgs=used_hvgs,
                used_pca_graph=True,
                rna_n_top_features=rna_n_top_features,
                n_components=n_components,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=True,
                backed=backed,
            )
            if atac_adata is not None:
                atac_adata, atac_adata_hvg = preprocessing_atac(
                    atac_adata,
                    min_features=min_features,
                    min_cells=min_cells,
                    used_hvgs=used_hvgs,
                    used_lsi_graph=True,
                    atac_n_top_features=atac_n_top_features,
                    n_components=n_components,
                    n=n,
                    batch_key=batch_key,
                    metric=metric
                )
                ## Concatenating different modalities
                adata_paired = ad.concat([rna_adata_hvg, atac_adata_hvg], axis=1)
                # 假设 obsm 的每个值都是形状相同的矩阵，可以通过 np.hstack 或其他方法进行合并
                rna_obsm = rna_adata_hvg.obsm
                atac_obsm = atac_adata_hvg.obsm
                # 只需要 feat key
                # combined_obsm = {key: np.hstack((rna_obsm[key], atac_obsm[key])) for key in rna_obsm.keys()}
                combined_obsm = np.hstack((rna_obsm['feat'], atac_obsm['feat']))
                # 将合并后的 obsm 信息添加到 adata_paired，但先确保索引匹配
                adata_paired.obsm['feat'] = pd.DataFrame(
                    combined_obsm,
                    index=adata_paired.obs.index  # 确保索引一致
                )
                adata_paired.obsm['spatial'] = rna_adata.obsm['spatial']

                ## the .obs layer is empty now, and we need to repopulate it
                rna_cols = rna_adata.obs.columns
                atac_cols = atac_adata.obs.columns

                rnaobs = rna_adata.obs.copy()
                rnaobs.columns = ["rna:" + x for x in rna_cols]
                atacobs = atac_adata.obs.copy()
                atacobs.columns = ["atac:" + x for x in atac_cols]
                adata_paired.obs = pd.merge(rnaobs, atacobs, left_index=True, right_index=True)

                ## 先将 scATAC 转换为基因活性矩阵
                if len(atac_adata.var_names) > 50000:
                    # 定义保存文件的路径
                    cache_path = "adata_ATAC_cache.h5ad"
                    # 检查是否已经存在缓存文件
                    if os.path.exists(cache_path):
                        # 如果缓存文件存在，直接加载
                        print("Gene activity matrix has been calculated, and loading cached adata_CG_atac object...")
                        adata_CG_atac = sc.read_h5ad(cache_path)
                    else:
                        # 如果缓存文件不存在，执行耗时计算生成新的 adata
                        print('Convert peak to gene activity matrix, this might take a while...', flush=True)
                        print('`genome` parameter should be set correctly', flush=True)
                        print("Choose from {‘hg19’, ‘hg38’, ‘mm9’, ‘mm10’}", flush=True)
                        # 生成新的 adata_CG_atac 对象
                        adata_CG_atac = gene_scores(atac_adata, genome=genome, use_gene_weigt=use_gene_weigt,
                                                    use_top_pcs=use_top_pcs)
                        # 将 adata_new 保存到缓存文件中
                        adata_CG_atac.write_h5ad(cache_path)
                        print(f"adata object cached at: {cache_path}")

                    ## 交集
                    common_genes = set(rna_adata.var_names).intersection(set(adata_CG_atac.var_names))
                    print('There are {} common genes in RNA and ATAC datasets'.format(len(common_genes)))
                    rna_adata_shared = rna_adata[:, list(common_genes)]
                    atac_adata_shared = adata_CG_atac[:, list(common_genes)]
                else:
                    ## 交集
                    common_genes = set(rna_adata.var_names).intersection(set(atac_adata.var_names))
                    print('There are {} common genes in RNA and ATAC datasets'.format(len(common_genes)))
                    rna_adata_shared = rna_adata[:, list(common_genes)]
                    atac_adata_shared = atac_adata[:, list(common_genes)]

                # 通过cell matching 构建组学间的图结构
                print('To start performing cell matching for adjacency matrix of the graph, please wait...', flush=True)
                inter_connect = create_adj(rna_adata,
                                           atac_adata,
                                           rna_adata_shared,
                                           atac_adata_shared,
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

                # Construct Spatial Graph
                if graph_const_method == 'mu_std':
                    spatial_adj = graph_construction(rna_adata_hvg, mode='mu_std', k=n, batch_key=batch_key)
                elif graph_const_method == 'Radius':
                    spatial_adj = graph_construction(adata, mode='Radius', k=n, batch_key=batch_key)
                elif graph_const_method == 'KNN':
                    spatial_adj = graph_construction(adata, mode='KNN', k=n, batch_key=batch_key)
                elif graph_const_method == 'Squidpy':
                    import squidpy as sq
                    sq.gr.spatial_neighbors(rna_adata_hvg, coord_type="generic", n_neighs=n)
                    spatial_adj = rna_adata_hvg.obsp['spatial_connectivities'] #.toarray()

                # Ensure adj is a csr_matrix
                if not isinstance(spatial_adj, csr_matrix):
                    spatial_adj = csr_matrix(spatial_adj)
                if not isinstance(inter_connect, csr_matrix):
                    inter_connect = csr_matrix(inter_connect)

                # Iterate over keys in rna_obsp (assuming adt_obsp has the same keys)
                # for key in rna_adata_hvg.obsp.keys():
                #     # 计算加权平均的连通性矩阵
                #     intra_connect = rna_adata_hvg.obsp[key] + atac_adata_hvg.obsp[key]

                # 计算加权平均的连通性矩阵
                adata_paired.obsp['spatial_connectivities'] = spatial_adj
                adata_paired.obsp['connectivities'] = weight * spatial_adj + (1 - weight) * inter_connect

                return adata_paired

            elif adt_adata is not None:
                adt_adata, adt_adata_hvg = preprocessing_adt(
                    adt_adata,
                    used_hvgs=used_hvgs,
                    used_pca_graph=True,
                    rna_n_top_features=rna_n_top_features,  # or gene list
                    n=n,
                    batch_key=batch_key,
                    metric=metric,
                    svd_solver=svd_solver,
                    backed=backed
                )
                ## Concatenating different modalities
                adata_paired = ad.concat([rna_adata_hvg, adt_adata_hvg], axis=1)
                # 假设 obsm 的每个值都是形状相同的矩阵，可以通过 np.hstack 或其他方法进行合并
                rna_obsm = rna_adata_hvg.obsm
                adt_obsm = adt_adata_hvg.obsm
                # 只需要 feat key
                # combined_obsm = {key: np.hstack((rna_obsm[key], atac_obsm[key])) for key in rna_obsm.keys()}
                combined_obsm = np.hstack((rna_obsm['feat'], adt_obsm['feat']))
                # 将合并后的 obsm 信息添加到 adata_paired，但先确保索引匹配
                adata_paired.obsm['feat'] = pd.DataFrame(
                    combined_obsm,
                    index=adata_paired.obs.index  # 确保索引一致
                )
                adata_paired.obsm['spatial'] = rna_adata.obsm['spatial']

                ## the .obs layer is empty now, and we need to repopulate it
                rna_cols = rna_adata.obs.columns
                adt_cols = adt_adata.obs.columns

                rnaobs = rna_adata.obs.copy()
                rnaobs.columns = ["rna:" + x for x in rna_cols]
                adtobs = adt_adata.obs.copy()
                adtobs.columns = ["adt:" + x for x in adt_cols]
                adata_paired.obs = pd.merge(rnaobs, adtobs, left_index=True, right_index=True)

                ## 交集
                common_genes = set(rna_adata.var_names).intersection(set(adt_adata.var_names))
                print('There are {} common genes in RNA and ADT datasets'.format(len(common_genes)))
                rna_adata_shared = rna_adata[:, list(common_genes)]
                adt_adata_shared = adt_adata[:, list(common_genes)]

                ## 通过cell matching 构建组学间的图结构
                print('To start performing cell matching for adjacency matrix of the graph, please wait...', flush=True)
                inter_connect = create_adj(rna_adata_hvg,
                                           adt_adata,
                                           rna_adata_shared,
                                           adt_adata_shared,
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

                # Construct Spatial Graph
                if graph_const_method == 'mu_std':
                    spatial_adj = graph_construction(rna_adata_hvg, mode='mu_std', k=n, batch_key=batch_key)
                elif graph_const_method == 'Radius':
                    spatial_adj = graph_construction(adata, mode='Radius', k=n, batch_key=batch_key)
                elif graph_const_method == 'KNN':
                    spatial_adj = graph_construction(adata, mode='KNN', k=n, batch_key=batch_key)
                elif graph_const_method == 'Squidpy':
                    import squidpy as sq
                    sq.gr.spatial_neighbors(rna_adata_hvg, coord_type="generic", n_neighs=n)
                    spatial_adj = rna_adata_hvg.obsp['spatial_connectivities'] #.toarray()

                # Ensure adj is a csr_matrix
                if not isinstance(spatial_adj, csr_matrix):
                    spatial_adj = csr_matrix(spatial_adj)
                if not isinstance(inter_connect, csr_matrix):
                    inter_connect = csr_matrix(inter_connect)

                # 计算加权平均的连通性矩阵
                adata_paired.obsp['spatial_connectivities'] = spatial_adj
                adata_paired.obsp['connectivities'] = weight * spatial_adj + (1 - weight) * inter_connect
                return adata_paired
    else:
        raise ValueError("Not support profile: `{}` yet".format(profile))