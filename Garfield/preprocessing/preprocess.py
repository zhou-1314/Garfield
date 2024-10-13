from anndata import AnnData
from mudata import MuData

# read data
from ..data.datareaders import concat_data

# preprocessing
from ..preprocessing.preprocess_utils import preprocessing


def DataProcess(
    adata_list,
    profile,
    data_type=None,
    sub_data_type=None,
    sample_col="batch",
    genome=None,
    weight=None,
    graph_const_method=None,
    use_gene_weight=True,
    user_cache_path=None,
    use_top_pcs=False,
    used_hvg=True,
    min_features=100,
    min_cells=3,
    keep_mt=False,
    target_sum=1e4,
    rna_n_top_features=3000,
    atac_n_top_features=10000,
    n_components=50,
    n_neighbors=15,
    metric="correlation",
    svd_solver="arpack",
):
    """
    Processes single or multi-modal data (e.g., RNA, ATAC, ADT, spatial) with optional preprocessing steps
    such as normalization, feature selection, and dimensionality reduction.

    Parameters
    ----------
    adata_list : list of AnnData or MuData objects
        List of AnnData or MuData objects to be concatenated and processed.
    profile : str
        Data profile type, e.g., 'RNA', 'ATAC', 'ADT', 'multi-modal', or 'spatial'.
    data_type : str, optional
        Type of data being processed, e.g., 'single-cell', 'bulk'. Default is None.
    sub_data_type : list[str], optional
        List of sub-data types for multi-modal data, e.g., ['rna', 'atac'] or ['rna', 'adt']. Default is None.
    sample_col : str, optional
        Column in the dataset used to indicate batch or sample groupings. Default is 'batch'.
    genome : str, optional
        Reference genome for the dataset. Default is None.
    weight : float or None, optional
        Weight for certain data processing steps, such as graph construction. Default is None.
    graph_const_method : str, optional
        Method for constructing the graph if applicable, e.g., 'knn'. Default is None.
    use_gene_weight : bool, optional
        Whether to use gene weights in the preprocessing steps. Default is True.
    user_cache_path : str, optional
        Path to the user's cache directory. Default is None.
    use_top_pcs : bool, optional
        Whether to use the top principal components during dimensionality reduction. Default is False.
    used_hvg : bool, optional
        Whether to use highly variable genes (HVG) for the analysis. Default is True.
    min_features : int, optional
        Minimum number of features required for a cell to be included. Default is 100.
    min_cells : int, optional
        Minimum number of cells required for a feature to be included. Default is 3.
    keep_mt : bool, optional
        Whether to keep mitochondrial genes in the dataset. Default is False.
    target_sum : float, optional
        Target sum for normalization. Default is 1e4.
    rna_n_top_features : int, optional
        Number of top features to keep for RNA data. Default is 3000.
    atac_n_top_features : int, optional
        Number of top features to keep for ATAC data. Default is 10000.
    n_components : int, optional
        Number of components for dimensionality reduction (e.g., PCA). Default is 50.
    n_neighbors : int, optional
        Number of neighbors for graph-based algorithms. Default is 15.
    metric : str, optional
        Distance metric to use in graph construction. Default is 'correlation'.
    svd_solver : str, optional
        Solver to use for singular value decomposition (SVD). Default is 'arpack'.

    Returns
    ----------
    AnnData or MuData
        Preprocessed single or multi-modal data based on the specified profile and sub_data_type.
    """
    # load data
    adata = concat_data(
        adata_list,
        batch_categories=None,
        join="inner",
        batch_key=sample_col,  # 'batch'
        index_unique=None,
        save=None,
    )
    if isinstance(adata, AnnData):
        if adata.X.max() < 50:
            print(
                "Warning: adata.X may have already been normalized, adata.X must be `counts`, please check."
            )
        else:
            adata.layers["counts"] = adata.X.copy()
    elif isinstance(adata, MuData):
        if adata.mod["rna"].X.max() < 50:
            print(
                "Warning: adata.X may have already been normalized, adata.X must be `counts`, please check."
            )
        else:
            adata.mod["rna"].layers["counts"] = adata.mod["rna"].X.copy()

    # RNA ATAC ADT
    if profile in ["RNA", "ATAC", "ADT"]:

        ## 预处理
        _, adata_hvg = preprocessing(
            adata,
            profile=profile,
            data_type=data_type,
            genome=genome,
            use_gene_weight=use_gene_weight,
            use_top_pcs=use_top_pcs,
            used_hvgs=used_hvg,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            rna_n_top_features=rna_n_top_features,
            atac_n_top_features=atac_n_top_features,
            n_components=n_components,
            n=n_neighbors,
            batch_key=sample_col,
            metric=metric,
            svd_solver=svd_solver,
            keep_mt=keep_mt,
        )

        return adata_hvg

    ### Paired multi-modal
    elif profile == "multi-modal":
        if len(sub_data_type) == 2:
            if sub_data_type[0] == "rna" and sub_data_type[1] == "atac":
                rna_adata = adata.mod["rna"].copy()
                atac_adata = adata.mod["atac"].copy()
                mdata = MuData({"rna": rna_adata, "atac": atac_adata})
            elif sub_data_type[0] == "rna" and sub_data_type[1] == "adt":
                rna_adata = adata.mod["rna"].copy()
                adt_adata = adata.mod["adt"].copy()
                mdata = MuData({"rna": rna_adata, "adt": adt_adata})
        else:
            ValueError(
                'The length of sub_data_type must be 2, such as: ["rna", "atac"] or ["rna", "adt"].'
            )
        del adata

        ## 预处理
        merged_adata = preprocessing(
            mdata,
            profile=profile,
            data_type=data_type,
            sub_data_type=sub_data_type,
            genome=genome,
            weight=weight,
            use_gene_weight=use_gene_weight,
            use_top_pcs=use_top_pcs,
            user_cache_path=user_cache_path,
            used_hvgs=used_hvg,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            rna_n_top_features=rna_n_top_features,
            atac_n_top_features=atac_n_top_features,
            n_components=n_components,
            n=n_neighbors,
            batch_key=sample_col,
            metric=metric,
            svd_solver=svd_solver,
            keep_mt=keep_mt,
        )

        return merged_adata

    ### spatial single- or multi-modal
    elif profile == "spatial":
        ## 预处理
        merged_adata = preprocessing(
            adata,
            profile=profile,
            data_type=data_type,
            sub_data_type=sub_data_type,
            genome=genome,
            weight=weight,
            graph_const_method=graph_const_method,
            use_gene_weight=use_gene_weight,
            use_top_pcs=use_top_pcs,
            used_hvgs=used_hvg,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            rna_n_top_features=rna_n_top_features,
            atac_n_top_features=atac_n_top_features,
            n_components=n_components,
            n=n_neighbors,
            batch_key=sample_col,
            metric=metric,
            svd_solver=svd_solver,
            keep_mt=keep_mt,
        )

        return merged_adata

    else:
        return "Unknown input data type."
