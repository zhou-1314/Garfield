# modified by https://github.com/JinmiaoChenLab/scTM/blob/main/sctm/analysis.py
import gseapy as gp
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel, delayed

def get_enrichr_geneset(organism="Human"):
    avail_organisms = ["Human", "Mouse", "Yeast", "Fly", "Fish", "Worm"]
    if organism not in avail_organisms:
        raise ValueError(f"available organism are {avail_organisms}")
    return gp.get_library_name(organism=organism)


def get_niche_enrichr(mks, geneset, niche_column='cluster',
                      niches="all", organism="human", topn_genes=200):
    """
    Perform Enrichr analysis on top genes for each niche derived from aggregated marker statistics (non-parallel version).

    Parameters
    ----------
    mks : DataFrame
        DataFrame containing marker statistics (from aggregate_top_markers).
    geneset : list
        List of gene sets to use for enrichment analysis.
    niche_column : str, optional
        The column in `mks` that represents the niche/cluster (default is 'cluster').
    niches : str or list, optional
        The niches to perform enrichment analysis on. Default is "all" which considers all unique values in the `niche_column` of `mks`.
    organism : str, optional
        Organism for Enrichr analysis (default is "human").
    topn_genes : int, optional
        Number of top genes to use for enrichment analysis for each niche (default is 20).

    Returns
    -------
    dict
        A dictionary with niches as keys and corresponding Enrichr results as values.
    """

    # Extract the niches from the specified niche column
    if niches == "all":
        niches = mks[niche_column].unique()
    elif not isinstance(niches, list):
        niches = [niches]

    # Validate niches exist in mks
    for niche in niches:
        if niche not in mks[niche_column].values:
            raise KeyError(f"Niche '{niche}' not found in mks['{niche_column}']")

    # Prepare the gene list for each niche
    niche_data = {}
    for niche in niches:
        # Get the top N genes for this niche (nlargest on 'logfoldchanges' to prioritize high fold change markers)
        top_genes = mks[mks[niche_column] == niche].nlargest(topn_genes, 'logfoldchanges')['genes'].tolist()
        niche_data[niche] = top_genes

    # Perform Enrichr analysis for each niche sequentially (non-parallel)
    enrichr = {}
    for niche in niches:
        niche_genes = niche_data[niche]
        niche_enrichr = gp.enrichr(
            gene_list=niche_genes, gene_sets=geneset,
            organism=organism, outdir=None
        )
        enrichr[niche] = niche_enrichr.results

    return enrichr


# 并行版本
def get_fast_niche_enrichr(mks, geneset, niche_column='cluster', niches="all",
                           organism="human", topn_genes=20, n_jobs=4):
    """
    Perform Enrichr analysis on top genes for each niche derived from aggregated marker statistics.

    Parameters
    ----------
    mks : DataFrame
        DataFrame containing marker statistics (from aggregate_top_markers).
    geneset : list
        List of gene sets to use for enrichment analysis.
    niche_column : str, optional
        The column in `mks` that represents the niche/cluster (default is 'cluster').
    niches : str or list, optional
        The niches to perform enrichment analysis on. Default is "all" which considers all unique values in the `niche_column` of `mks`.
    organism : str, optional
        Organism for Enrichr analysis (default is "human").
    topn_genes : int, optional
        Number of top genes to use for enrichment analysis for each niche (default is 20).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1, no parallelism).

    Returns
    -------
    dict
        A dictionary with niches as keys and corresponding Enrichr results as values.
    """

    # Extract the niches from the specified niche column
    if niches == "all":
        niches = mks[niche_column].unique()
    elif not isinstance(niches, list):
        niches = [niches]

    # Validate niches exist in mks
    for niche in niches:
        if niche not in mks[niche_column].values:
            raise KeyError(f"Niche '{niche}' not found in mks['{niche_column}']")

    # Prepare the gene list for each niche
    niche_data = {}
    for niche in niches:
        # Get the top N genes for this niche (nlargest on 'logfoldchanges' to prioritize high fold change markers)
        top_genes = mks[mks[niche_column] == niche].nlargest(topn_genes, 'logfoldchanges')['genes'].tolist()
        niche_data[niche] = top_genes

    # Function to perform Enrichr analysis for each niche in parallel (if n_jobs > 1)
    def enrichr_for_niche(niche):
        niche_genes = niche_data[niche]
        niche_enrichr = gp.enrichr(
            gene_list=niche_genes, gene_sets=geneset, organism=organism, outdir=None
        )
        return niche, niche_enrichr.results

    # Use multiprocessing if n_jobs > 1
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(enrichr_for_niche, niches)
    else:
        # Single-threaded execution
        results = [enrichr_for_niche(niche) for niche in niches]

    # Convert results to dictionary format
    enrichr = {niche: result for niche, result in results}

    return enrichr

## GSEA
def get_niche_gsea(
    mks,
    geneset,
    niche_column='cluster',
    niches="all",
    geneset_size=[5, 500],
    permutations=1000,
    n_jobs=20
):
    """
    Perform GSEA analysis for each niche based on the full ranked gene list from marker statistics.

    Parameters
    ----------
    mks : DataFrame
        DataFrame containing marker statistics (from aggregate_top_markers).
    geneset : list
        List of gene sets to use for GSEA analysis.
    niche_column : str, optional
        The column in `mks` that represents the niche/cluster (default is 'cluster').
    niches : str or list, optional
        The niches to perform GSEA analysis on. Default is "all" which considers all unique values in the `niche_column` of `mks`.
    geneset_size : list of int, optional
        Minimum and maximum size of genesets to use in GSEA analysis (default is [5, 500]).
    permutations : int, optional
        Number of permutations to use in GSEA analysis (default is 1000).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 20).

    Returns
    -------
    dict
        A dictionary with niches as keys and corresponding GSEA results as values.
    """

    # Extract the niches from the specified niche column
    if niches == "all":
        niches = mks[niche_column].unique()
    elif not isinstance(niches, list):
        niches = [niches]

    # Validate niches exist in mks
    for niche in niches:
        if niche not in mks[niche_column].values:
            raise KeyError(f"Niche '{niche}' not found in mks['{niche_column}']")

    def process_niche(niche, mks, geneset, geneset_size, permutations):
        # Filter data for the specific niche and rank genes by logfoldchanges
        niche_data = mks[mks[niche_column] == niche]
        ranked_genes = niche_data.sort_values(by='logfoldchanges', ascending=False)
        rank = pd.Series(ranked_genes['logfoldchanges'].values, index=ranked_genes['genes'])

        # Perform GSEA
        results = gp.prerank(
            rnk=rank,
            gene_sets=geneset,
            permutations=permutations,
            min_size=geneset_size[0],
            max_size=geneset_size[1],
            threads=1,
            verbose=True,
        )
        return results.res2d

    # Perform GSEA analysis for each niche (parallel or non-parallel)
    gsea_results = Parallel(n_jobs=n_jobs)(
        delayed(process_niche)(niche, mks, geneset, geneset_size, permutations)
        for niche in niches
    )

    # Combine the results into a dictionary
    gseas = {}
    for i, niche in enumerate(niches):
        gseas[niche] = gsea_results[i]

    return gseas
