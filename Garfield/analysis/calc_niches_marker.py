"""
This module contains helper functions for the ´analysis´ subpackage.
"""

import scipy.sparse as sp
from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from anndata import AnnData


## identify marker genes for each niche
## This is based on Ni Huang code for the marker discovery
# also found https://github.com/Teichlab/sctk/blob/master/sctk/_markers.py
def calc_marker_stats(ad, groupby, genes=None, use_raw=True, inplace=False, partial=False):
    """
    Calculate marker statistics for grouped data.

    Parameters
    ----------
    ad : AnnData
        AnnData object containing expression data.
    groupby : str
        Column in `ad.obs` used for grouping cells. Must be categorical.
    genes : list, optional
        List of genes to subset for calculations. If None, all genes are used.
    use_raw : bool, True
        Which representation of data to use ('raw' or normalized). Default is 'raw'.
    inplace : bool, optional
        Whether to modify the AnnData object in place. Default is False.
    partial : bool, optional
        If True, calculate only fraction and mean statistics; skip additional computations.
        Default is False.

    Returns
    -------
    tuple or None
        If `inplace` is False, returns a tuple of DataFrames: (frac_df, mean_df, stats_df).
        Otherwise, modifies `ad` in place and returns None.
    """
    if ad.obs[groupby].dtype.name != 'category':
        raise ValueError('"%s" is not categorical' % groupby)
    n_grp = ad.obs[groupby].cat.categories.size
    if n_grp < 2:
        raise ValueError('"%s" must contain at least 2 categories' % groupby)
    if use_raw and 'raw' in ad.layers.keys():
        X = ad.raw.X
        var_names = ad.raw.var_names.values
    else:
        X = ad.X
        var_names = ad.var_names.values
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    if genes:
        v_idx = var_names.isin(genes)
        X = X[:, v_idx]
        var_names = var_names[v_idx]
    # 检测是否需要标准化
    if hasattr(ad, 'uns') or 'normalized' in ad.uns.get('status', {}):
        if X.mean() < 50 or (X.max() > 1 and X.max() < 100):
            ad.uns['status'] = {'normalized': True}
        else:
            X = normalize(X, norm='max', axis=0)
            ad.uns['status'] = {'normalized': True}

    k_nonzero = X.sum(axis=0).A1 > 0
    X = X[:, np.where(k_nonzero)[0]]
    var_names = var_names[k_nonzero]

    n_var = var_names.size
    x = np.arange(n_var)

    grp_indices = {k: g.index.values for k, g in ad.obs.reset_index().groupby(groupby, sort=False)}

    frac_df = pd.DataFrame({k: (X[idx, :] > 0).mean(axis=0).A1 for k, idx in grp_indices.items()}, index=var_names)
    mean_df = pd.DataFrame({k: X[idx, :].mean(axis=0).A1 for k, idx in grp_indices.items()}, index=var_names)

    if partial:
        stats_df = None
    else:
        frac_order = np.apply_along_axis(np.argsort, axis=1, arr=frac_df.values)
        y1 = frac_order[:, n_grp - 1]
        y2 = frac_order[:, n_grp - 2]
        y3 = frac_order[:, n_grp - 3] if n_grp > 2 else y2
        top_frac_grps = frac_df.columns.values[y1]
        top_fracs = frac_df.values[x, y1]
        frac_diffs = top_fracs - frac_df.values[x, y2]
        max_frac_diffs = top_fracs - frac_df.values[x, y3]

        mean_order = np.apply_along_axis(np.argsort, axis=1, arr=mean_df.values)
        y1 = mean_order[:, n_grp - 1]
        y2 = mean_order[:, n_grp - 2]
        y3 = mean_order[:, n_grp - 3] if n_grp > 2 else y2
        top_mean_grps = mean_df.columns.values[y1]
        top_means = mean_df.values[x, y1]
        mean_diffs = top_means - mean_df.values[x, y2]
        max_mean_diffs = top_means - mean_df.values[x, y3]

        stats_df = pd.DataFrame({
            'top_frac_group': top_frac_grps, 'top_frac': top_fracs, 'frac_diff': frac_diffs,
            'max_frac_diff': max_frac_diffs,
            'top_mean_group': top_mean_grps, 'top_mean': top_means, 'mean_diff': mean_diffs,
            'max_mean_diff': max_mean_diffs
        }, index=var_names)

        # stats_df['top_frac_group'] = stats_df['top_frac_group'].astype('category')
        # stats_df['top_frac_group'].cat.reorder_categories(list(ad.obs[groupby].cat.categories), inplace=True)
        # 修复 reorder_categories
        stats_df['top_frac_group'] = stats_df['top_frac_group'].astype('category')
        stats_df['top_frac_group'] = stats_df['top_frac_group'].cat.reorder_categories(list(ad.obs[groupby].cat.categories))

    if inplace:
        if use_raw:
            ad.raw.varm[f'frac_{groupby}'] = frac_df
            ad.raw.varm[f'mean_{groupby}'] = mean_df
            if not partial:
                ad.raw.var = pd.concat([ad.raw.var, stats_df], axis=1)
        else:
            ad.varm[f'frac_{groupby}'] = frac_df
            ad.varm[f'mean_{groupby}'] = mean_df
            if not partial:
                ad.var = pd.concat([ad.var, stats_df], axis=1)
    else:
        return frac_df, mean_df, stats_df


def filter_marker_stats(data, use_raw=True, min_frac_diff=0.1, min_mean_diff=0.1,
                        max_next_frac=0.9, max_next_mean=0.95, strict=False, how='or'):
    """
    Filter marker statistics based on thresholds.

    Parameters
    ----------
    data : AnnData or DataFrame
        Data containing marker statistics.
    use_raw : bool, optional
        Which data representation to use ('raw' or processed). Default is 'raw'.
    min_frac_diff : float, optional
        Minimum difference in fraction to consider a marker valid. Default is 0.1.
    min_mean_diff : float, optional
        Minimum difference in mean expression to consider a marker valid. Default is 0.1.
    max_next_frac : float, optional
        Maximum fraction difference for non-top markers. Default is 0.9.
    max_next_mean : float, optional
        Maximum mean difference for non-top markers. Default is 0.95.
    strict : bool, optional
        If True, use stricter thresholds for filtering. Default is False.
    how : str, optional
        Logical operation to combine fraction and mean filters ('or' or 'and'). Default is 'or'.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing marker statistics.
    """
    columns = ['top_frac_group', 'top_frac', 'frac_diff', 'max_frac_diff',
               'top_mean_group', 'top_mean', 'mean_diff', 'max_mean_diff']
    if isinstance(data, anndata.AnnData):
        stats_df = data.raw.var[columns] if use_raw else data.var[columns]
    elif isinstance(data, pd.DataFrame):
        stats_df = data[columns]
    else:
        raise ValueError('Invalid input, must be an AnnData or DataFrame')
    frac_diff = stats_df.frac_diff if strict else stats_df.max_frac_diff
    mean_diff = stats_df.mean_diff if strict else stats_df.max_mean_diff
    same_group = stats_df.top_frac_group == stats_df.top_mean_group
    meet_frac_requirement = (frac_diff >= min_frac_diff) & (stats_df.top_frac - frac_diff <= max_next_frac)
    meet_mean_requirement = (mean_diff >= min_mean_diff) & (stats_df.top_mean - mean_diff <= max_next_mean)
    if how == 'or':
        filtered = stats_df.loc[same_group & (meet_frac_requirement | meet_mean_requirement)]
    else:
        filtered = stats_df.loc[same_group & (meet_frac_requirement & meet_mean_requirement)]
    if strict:
        filtered = filtered.sort_values(['top_frac_group', 'mean_diff', 'frac_diff'], ascending=[True, False, False])
    else:
        filtered = filtered.sort_values(['top_frac_group', 'mean_diff', 'frac_diff'], ascending=[True, False, False])
    # filtered['top_frac_group'] = filtered['top_frac_group'].astype('category')
    # filtered['top_frac_group'].cat.reorder_categories(list(stats_df['top_frac_group'].cat.categories), inplace=True)
    # 修复 reorder_categories
    filtered['top_frac_group'] = filtered['top_frac_group'].astype('category')
    filtered['top_frac_group'] = filtered['top_frac_group'].cat.reorder_categories(list(stats_df['top_frac_group'].cat.categories))

    return filtered


def aggregate_top_markers(ad, mks, groupby, n_genes=100, use_raw=True, **kwargs):
    """
    Aggregate top marker genes.

    Parameters
    ----------
    ad : AnnData
        AnnData object containing expression data.
    mks : DataFrame
        DataFrame containing marker statistics.
    groupby : str
        Column in `ad.obs` used for grouping cells.
    n_genes : int, optional
        Number of top genes to include in the test. Default is 100.
    use_raw : bool, optional
        Whether to use raw expression values for the test. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to the rank_genes_groups function.

    Returns
    -------
    DataFrame
        Merged DataFrame with marker statistics and differential expression results.
    """
    genes = top_markers(mks, top_n=n_genes)
    aux_ad = anndata.AnnData(
        X=ad.raw.X if use_raw else ad.X,
        obs=ad.obs.copy(),
        var=ad.raw.var.copy() if use_raw else ad.var.copy()
    )
    aux_ad = aux_ad[:, genes].copy()
    sc.tl.rank_genes_groups(aux_ad, groupby=groupby, n_genes=n_genes, use_raw=False, **kwargs)
    de_tbl = extract_de_table(aux_ad.uns['rank_genes_groups'])
    return mks.reset_index().rename(columns={'index': 'genes', 'top_frac_group': 'cluster'}).merge(
        de_tbl[['cluster', 'genes', 'logfoldchanges', 'pvals', 'pvals_adj']], how='left'
    )


def extract_de_table(de_dict):
    """
    Extract a differential expression table from an AnnData.uns dictionary.

    Parameters
    ----------
    de_dict : dict
        Dictionary containing differential expression results from AnnData.

    Returns
    -------
    DataFrame
        DataFrame containing differential expression statistics, including cluster, rank,
        gene names, and relevant metrics (e.g., logfoldchanges, p-values).
    """
    if de_dict['params']['method'] == 'logreg':
        requested_fields = ('scores',)
    else:
        requested_fields = ('scores', 'logfoldchanges', 'pvals', 'pvals_adj',)
    gene_df = _recarray_to_dataframe(de_dict['names'], 'genes')[
        ['cluster', 'rank', 'genes']]
    gene_df['ref'] = de_dict['params']['reference']
    gene_df = gene_df[['cluster', 'ref', 'rank', 'genes']]
    de_df = pd.DataFrame({
        field: _recarray_to_dataframe(de_dict[field], field)[field]
        for field in requested_fields if field in de_dict
    })
    de_tbl = gene_df.merge(de_df, left_index=True, right_index=True)
    de_tbl = de_tbl.loc[de_tbl.genes.astype(str) != 'nan', :]
    return de_tbl


def _recarray_to_dataframe(array, field_name):
    return pd.DataFrame(array).reset_index().rename(
        columns={'index': 'rank'}).melt(
        id_vars='rank', var_name='cluster', value_name=field_name)


def top_markers(df, top_n=5, groupby='top_frac_group'):
    return df.groupby(groupby).head(top_n).index.to_list()


# enrichment utils
