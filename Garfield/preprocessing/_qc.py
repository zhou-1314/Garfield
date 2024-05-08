"""Quality Control"""
import numpy as np
from scipy.sparse import (
    issparse,
    csr_matrix,
    diags,
)
import re

def filter_genes(adata,
                 min_n_cells=3,
                 max_n_cells=None,
                 min_pct_cells=None,
                 max_pct_cells=None,
                 min_n_counts=None,
                 max_n_counts=None,
                 expr_cutoff=1,
                 verbose=True):
    """Filter out features based on different metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_n_cells: `int`, optional (default: 5)
        Minimum number of cells expressing one feature
    min_pct_cells: `float`, optional (default: None)
        Minimum percentage of cells expressing one feature
    min_n_counts: `int`, optional (default: None)
        Minimum number of read count for one feature
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff.
        If greater than expr_cutoff,the feature is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {{'rna','atac'}},case insensitive

    Returns
    -------
    updates `adata` with a subset of features that pass the filtering.
    updates `adata` with the following fields if cal_qc() was not performed.
    n_counts: `pandas.Series` (`adata.var['n_counts']`,dtype `int`)
       The number of read count each gene has.
    n_cells: `pandas.Series` (`adata.var['n_cells']`,dtype `int`)
       The number of cells in which each gene is expressed.
    pct_cells: `pandas.Series` (`adata.var['pct_cells']`,dtype `float`)
       The percentage of cells in which each gene is expressed.
    """

    feature = 'genes'
    if(not issparse(adata.X)):
        adata.X = csr_matrix(adata.X)

    if('n_counts' in adata.var_keys()):
        n_counts = adata.var['n_counts']
    else:
        n_counts = np.sum(adata.X, axis=0).A1
        adata.var['n_counts'] = n_counts
    if('n_cells' in adata.var_keys()):
        n_cells = adata.var['n_cells']
    else:
        n_cells = np.sum(adata.X >= expr_cutoff, axis=0).A1
        adata.var['n_cells'] = n_cells
    if('pct_cells' in adata.var_keys()):
        pct_cells = adata.var['pct_cells']
    else:
        pct_cells = n_cells/adata.shape[0]
        adata.var['pct_cells'] = pct_cells

    if verbose:
        print('Before filtering: ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    if(sum(list(map(lambda x: x is None,
                    [min_n_cells, min_pct_cells, min_n_counts,
                     max_n_cells, max_pct_cells, max_n_counts,
                     ]))) == 6):
        print('No filtering')
    else:
        feature_subset = np.ones(len(adata.var_names), dtype=bool)
        if(min_n_cells is not None and verbose):
            print('Filter '+feature+' based on min_n_cells')
            feature_subset = (n_cells >= min_n_cells) & feature_subset
        if(max_n_cells is not None and verbose):
            print('Filter '+feature+' based on max_n_cells')
            feature_subset = (n_cells <= max_n_cells) & feature_subset
        if(min_pct_cells is not None and verbose):
            print('Filter '+feature+' based on min_pct_cells')
            feature_subset = (pct_cells >= min_pct_cells) & feature_subset
        if(max_pct_cells is not None and verbose):
            print('Filter '+feature+' based on max_pct_cells')
            feature_subset = (pct_cells <= max_pct_cells) & feature_subset
        if(min_n_counts is not None and verbose):
            print('Filter '+feature+' based on min_n_counts')
            feature_subset = (n_counts >= min_n_counts) & feature_subset
        if(max_n_counts is not None and verbose):
            print('Filter '+feature+' based on max_n_counts')
            feature_subset = (n_counts <= max_n_counts) & feature_subset
        adata._inplace_subset_var(feature_subset)
        if verbose:
            print('After filtering out low-expressed '+feature+': ')
            print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    return None

def cal_qc_rna(adata, expr_cutoff=1):
    """Calculate quality control metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff.
        If greater than expr_cutoff,the feature is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {'rna','atac'},case insensitive
    Returns
    -------
    updates `adata` with the following fields.
    n_counts: `pandas.Series` (`adata.var['n_counts']`,dtype `int`)
       The number of read count each gene has.
    n_cells: `pandas.Series` (`adata.var['n_cells']`,dtype `int`)
       The number of cells in which each gene is expressed.
    pct_cells: `pandas.Series` (`adata.var['pct_cells']`,dtype `float`)
       The percentage of cells in which each gene is expressed.
    n_counts: `pandas.Series` (`adata.obs['n_counts']`,dtype `int`)
       The number of read count each cell has.
    n_genes: `pandas.Series` (`adata.obs['n_genes']`,dtype `int`)
       The number of genes expressed in each cell.
    pct_genes: `pandas.Series` (`adata.obs['pct_genes']`,dtype `float`)
       The percentage of genes expressed in each cell.
    n_peaks: `pandas.Series` (`adata.obs['n_peaks']`,dtype `int`)
       The number of peaks expressed in each cell.
    pct_peaks: `pandas.Series` (`adata.obs['pct_peaks']`,dtype `int`)
       The percentage of peaks expressed in each cell.
    pct_mt: `pandas.Series` (`adata.obs['pct_mt']`,dtype `float`)
       the percentage of counts in mitochondrial genes
    """

    if(not issparse(adata.X)):
        adata.X = csr_matrix(adata.X)

    n_counts = adata.X.sum(axis=0).A1
    adata.var['n_counts'] = n_counts
    n_cells = (adata.X >= expr_cutoff).sum(axis=0).A1
    adata.var['n_cells'] = n_cells
    adata.var['pct_cells'] = n_cells/adata.shape[0]

    n_counts = adata.X.sum(axis=1).A1
    adata.obs['n_counts'] = n_counts
    n_features = (adata.X >= expr_cutoff).sum(axis=1).A1
    adata.obs['n_genes'] = n_features
    adata.obs['pct_genes'] = n_features/adata.shape[1]
    r = re.compile("^MT-", flags=re.IGNORECASE)
    mt_genes = list(filter(r.match, adata.var_names))
    if(len(mt_genes) > 0):
        n_counts_mt = adata[:, mt_genes].X.sum(axis=1).A1
        adata.obs['pct_mt'] = n_counts_mt/n_counts
    else:
        adata.obs['pct_mt'] = 0

def filter_cells_rna(adata,
                     min_n_genes=None,
                     max_n_genes=None,
                     min_pct_genes=None,
                     max_pct_genes=None,
                     min_n_counts=None,
                     max_n_counts=None,
                     expr_cutoff=0.5,
                     verbose=True):
    """Filter out cells for RNA-seq based on different metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_n_genes: `int`, optional (default: None)
        Minimum number of genes expressed
    min_pct_genes: `float`, optional (default: None)
        Minimum percentage of genes expressed
    min_n_counts: `int`, optional (default: None)
        Minimum number of read count for one cell
    expr_cutoff: `float`, optional (default: 0.5)
        Expression cutoff.
        If greater than expr_cutoff,the gene is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {{'rna','atac'}},case insensitive
    Returns
    -------
    updates `adata` with a subset of cells that pass the filtering.
    updates `adata` with the following fields if cal_qc() was not performed.
    n_counts: `pandas.Series` (`adata.obs['n_counts']`,dtype `int`)
       The number of read count each cell has.
    n_genes: `pandas.Series` (`adata.obs['n_genes']`,dtype `int`)
       The number of genes expressed in each cell.
    pct_genes: `pandas.Series` (`adata.obs['pct_genes']`,dtype `float`)
       The percentage of genes expressed in each cell.
    n_peaks: `pandas.Series` (`adata.obs['n_peaks']`,dtype `int`)
       The number of peaks expressed in each cell.
    pct_peaks: `pandas.Series` (`adata.obs['pct_peaks']`,dtype `int`)
       The percentage of peaks expressed in each cell.
    """

    if(not issparse(adata.X)):
        adata.X = csr_matrix(adata.X)
    if('n_counts' in adata.obs_keys()):
        n_counts = adata.obs['n_counts']
    else:
        n_counts = np.sum(adata.X, axis=1).A1
        adata.obs['n_counts'] = n_counts

    if('n_genes' in adata.obs_keys()):
        n_genes = adata.obs['n_genes']
    else:
        n_genes = np.sum(adata.X >= expr_cutoff, axis=1).A1
        adata.obs['n_genes'] = n_genes
    if('pct_genes' in adata.obs_keys()):
        pct_genes = adata.obs['pct_genes']
    else:
        pct_genes = n_genes/adata.shape[1]
        adata.obs['pct_genes'] = pct_genes

    if verbose:
        print('before filtering: ')
        print(f"{adata.shape[0]} cells,  {adata.shape[1]} genes")
    if(sum(list(map(lambda x: x is None,
                    [min_n_genes,
                     min_pct_genes,
                     min_n_counts,
                     max_n_genes,
                     max_pct_genes,
                     max_n_counts]))) == 6):
        print('No filtering')
    else:
        cell_subset = np.ones(len(adata.obs_names), dtype=bool)
        if(min_n_genes is not None and verbose):
            print('filter cells based on min_n_genes')
            cell_subset = (n_genes >= min_n_genes) & cell_subset
        if(max_n_genes is not None  and verbose):
            print('filter cells based on max_n_genes')
            cell_subset = (n_genes <= max_n_genes) & cell_subset
        if(min_pct_genes is not None  and verbose):
            print('filter cells based on min_pct_genes')
            cell_subset = (pct_genes >= min_pct_genes) & cell_subset
        if(max_pct_genes is not None  and verbose):
            print('filter cells based on max_pct_genes')
            cell_subset = (pct_genes <= max_pct_genes) & cell_subset
        if(min_n_counts is not None  and verbose):
            print('filter cells based on min_n_counts')
            cell_subset = (n_counts >= min_n_counts) & cell_subset
        if(max_n_counts is not None  and verbose):
            print('filter cells based on max_n_counts')
            cell_subset = (n_counts <= max_n_counts) & cell_subset
        adata._inplace_subset_obs(cell_subset)
        if verbose:
            print('after filtering out low-quality cells: ')
            print(f"{adata.shape[0]} cells,  {adata.shape[1]} genes")
    return None