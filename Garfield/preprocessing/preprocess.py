import scanpy as sc
import muon as mu
import anndata as ad

from anndata import AnnData
from muon import MuData

import episcanpy as epi
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import issparse, csr

# from ._pca import select_pcs_features
from ._utils import gene_scores


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
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log1p transforming
    sc.pp.log1p(adata)

    # scale data, clip values exceeding standard deviation 10.
    sc.pp.scale(adata, max_value=10)

    adata.raw = adata
    # Finding variable features
    if type(rna_n_top_features) == int and rna_n_top_features > 0:
        if rna_n_top_features > len(adata.var_names):
            rna_n_top_features = len(adata.var_names)
        if batch_key is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features, batch_key=batch_key)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=rna_n_top_features)
        adata = adata[:, adata.var.highly_variable].copy()
    elif type(rna_n_top_features) != int:
        if isinstance(rna_n_top_features, str):
            if os.path.isfile(rna_n_top_features):
                rna_n_top_features = np.loadtxt(n_top_features, dtype=str)
        adata = reindex(adata, rna_n_top_features)

    ## PCA
    sc.tl.pca(adata, svd_solver=svd_solver)

    # use scanpy functions to do the graph construction
    sc.pp.neighbors(adata, n_neighbors=n, metric=metric, use_rep='X_pca')

    return adata

def preprocessing_atac(adata: AnnData,
                       data_type: str = None,
                       genome: str = None,
                       use_gene_weigt: bool = True,
                       use_top_pcs: bool = True,
                       min_features: int = 100,
                       min_cells: int = 3,
                       atac_n_top_features=100000,  # or gene list
                       n: int = 15,
                       batch_key: str = 'batch',
                       metric: str = 'euclidean',
                       method='umap',
                       if_bi: int = 0,
                       n_components: int = 50,
                       svd_solver: str = 'arpack'
                       ):
    """
    Preprocess scCAS data matrix.

    Parameters
    ----------
    adata :  AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks/regions.
    filter_rate : float, optional
        Proportion for feature selection, by default 0.01
    """
    if if_bi == 1:
        adata.X.data = np.ones(adata.X.data.shape[0], dtype=np.int8)

    if min_features is None: min_features = 100
    if atac_n_top_features is None: atac_n_top_features = 100000

    # Preprocessing
    if type(adata.X) != csr.csr_matrix:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    epi.pp.filter_cells(adata, min_features=min_features)
    epi.pp.filter_features(adata, min_cells=min_cells)
    adata.raw = adata

    ## HVP
    if type(atac_n_top_features) == int and atac_n_top_features > 0 and atac_n_top_features < adata.shape[1]:
        # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
        adata = epi.pp.select_var_feature(adata, nb_features=atac_n_top_features, show=False, copy=True)
    elif type(atac_n_top_features) != int:
        if isinstance(atac_n_top_features, str):
            if os.path.isfile(atac_n_top_features):
                atac_n_top_features = np.loadtxt(atac_n_top_features, dtype=str)
        adata = reindex(adata, atac_n_top_features)

    adata.layers['binary'] = adata.X.copy()
    epi.pp.normalize_total(adata)
    adata.layers['normalised'] = adata.X.copy()
    epi.pp.log1p(adata)
    epi.pp.lazy(adata, svd_solver=svd_solver, n_neighbors=n, metric=metric, method=method)

    ## only for HVP
    adata = adata[:, adata.var.highly_variable].copy()

    ## gene_scores convert peak to gene names
    if data_type == 'UnPaired':
        print('`genome` parameter should be set correctly')
        print("Choose from {‘hg19’, ‘hg38’, ‘mm9’, ‘mm10’}")
        adata = gene_scores(adata, genome=genome, use_gene_weigt=use_gene_weigt, use_top_pcs=use_top_pcs)

    return adata

### preprocessing main function
def preprocessing(
        adata: [AnnData, MuData],
        profile: str = 'RNA',
        data_type: str = 'Paired',
        genome: str = None,
        use_gene_weigt: bool = True,
        use_top_pcs: bool = True,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = None,
        rna_n_top_features=None,  # or gene list
        atac_n_top_features=None,  # or gene list
        n: int = 15,
        batch_key: str = 'batch',
        metric: str = 'euclidean',
        method: str = 'umap',
        svd_solver: str = 'arpack',
        if_bi: int = 0,
        n_components: int = 50,
        keep_mt: bool = False,
        backed: bool = False
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
            atac_n_top_features=atac_n_top_features,
            n=n,
            batch_key=batch_key,
            metric=metric,
            method=method,
            if_bi=if_bi,
            n_components=n_components,
            svd_solver=svd_solver
        )
    elif profile == 'muData':
        assert data_type in ['Paired', 'UnPaired'], 'Data_type must be "Paired", or "UnPaired".'
        rna_adata = adata.mod['rna']
        atac_adata = adata.mod['atac']

        if data_type == 'Paired':
            rna_adata = preprocessing_rna(
                rna_adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                rna_n_top_features=rna_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=keep_mt,
                backed=backed,
            )
            atac_adata = preprocessing_atac(
                atac_adata,
                min_features=min_features,
                min_cells=min_cells,
                atac_n_top_features=atac_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                method=method,
                if_bi=if_bi,
                n_components=n_components,
                svd_solver=svd_solver
            )
            ## Concatenating different modalities
            adata_paired = ad.concat([rna_adata, atac_adata], axis=1)

            ## the .obs layer is empty now, and we need to repopulate it
            rna_cols = rna_adata.obs.columns
            atac_cols = atac_adata.obs.columns

            rnaobs = rna_adata.obs.copy()
            rnaobs.columns = ["rna:" + x for x in rna_cols]
            atacobs = atac_adata.obs.copy()
            atacobs.columns = ["atac:" + x for x in atac_cols]
            adata_paired.obs = pd.merge(rnaobs, atacobs, left_index=True, right_index=True)

            ##
            # 你可以在一个字典中保留原始的.obsp数据
            adata_paired.uns['obsp_rna'] = rna_adata.obsp
            adata_paired.uns['obsp_atac'] = atac_adata.obsp
            return adata_paired

        elif data_type == 'UnPaired':
            rna_adata = preprocessing_rna(
                rna_adata,
                min_features=min_features,
                min_cells=min_cells,
                target_sum=target_sum,
                rna_n_top_features=rna_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                svd_solver=svd_solver,
                keep_mt=keep_mt,
                backed=backed,
            )
            ## 先将 scATAC 转换为基因活性矩阵
            atac_adata = preprocessing_atac(
                atac_adata,
                data_type,
                genome,
                use_gene_weigt,
                use_top_pcs,
                min_features=min_features,
                min_cells=min_cells,
                atac_n_top_features=atac_n_top_features,
                n=n,
                batch_key=batch_key,
                metric=metric,
                method=method,
                if_bi=if_bi,
                n_components=n_components,
                svd_solver=svd_solver
            )
            ## Concatenate datasets, by modality
            adata_unpaired = ad.concat([rna_adata, atac_adata], axis=0)

            ## the .obs layer is empty now, and we need to repopulate it
            rna_cols = rna_adata.obs.columns
            atac_cols = atac_adata.obs.columns

            rnaobs = rna_adata.obs.copy()
            rnaobs.columns = ["rna:" + x for x in rna_cols]
            atacobs = atac_adata.obs.copy()
            atacobs.columns = ["atac:" + x for x in atac_cols]
            adata_unpaired.obs = pd.merge(rnaobs, atacobs, left_index=True, right_index=True)

            ##
            # 你可以在一个字典中保留原始的.obsp数据
            adata_unpaired.uns['obsp_rna'] = rna_adata.obsp
            adata_unpaired.uns['obsp_atac'] = atac_adata.obsp
            return adata_unpaired
    else:
        raise ValueError("Not support profile: `{}` yet".format(profile))