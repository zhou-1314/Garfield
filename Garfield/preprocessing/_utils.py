import warnings
import io
import pybedtools
import pkgutil
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
import scipy.sparse
from scipy.sparse import (
    csr_matrix,
    coo_matrix,
)
from collections import defaultdict, Counter
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.linalg import svds
from sklearn.cross_decomposition import CCA
from sklearn.utils.extmath import randomized_svd

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
        percentage = (
            len(user_var_names.intersection(genes)) / len(user_var_names)
        ) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(
                f"WARNING: Query shares {percentage}% of its genes with the reference."
                "This may lead to inaccuracy in the results."
            )
    except Exception:
        logger.warning("WARNING: Something is wrong with the reference genes.")

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in genes:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print(
            "Query data is missing expression data of ",
            len(ref_genes_not_in_query),
            " genes which were contained in the reference dataset.",
        )

    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print("There are {} gene in selected genes".format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        new_X[:, idx] = adata[:, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={"var_names": genes})
    return adata


def get_centroids(sub_data, labels):
    """
    Compute the centroids (cluster mean) of adata.

    Parameters
    ----------
    sub_data: scanpy object of each sample
    labels: np.array of shape (n_samples,)
        Cluster labels of each sample, coded from 0, ..., num_clusters-1

    Returns
    -------
    centroids: np.array of shape (n_centroids, n_features)
        Matrix of cluster centroids, the i-th column is the centroid of the i-th cluster
    """
    arr = sub_data.X
    arr_raw = sub_data.layers["counts"]
    meta = sub_data.obs
    cluster_label_to_indices = defaultdict(list)
    for i, l in enumerate(labels):
        cluster_label_to_indices[l].append(i)

    unique_labels = sorted(cluster_label_to_indices.keys())
    if not all(i == l for i, l in enumerate(unique_labels)):
        raise ValueError("labels must be coded in integers from 0, ..., n_clusters-1.")

    ## 对于每个簇，计算其均值，返回标准化的矩阵
    centroids = np.empty((len(unique_labels), arr.shape[1]))
    for curr_label, indices in cluster_label_to_indices.items():
        centroids[curr_label, :] = arr[indices, :].mean(axis=0)

    ## 对于每个簇，计算其均值，返回raw counts的矩阵
    counts = np.empty((len(unique_labels), arr.shape[1]))
    for curr_label, indices in cluster_label_to_indices.items():
        counts[curr_label, :] = arr_raw[indices, :].mean(axis=0)

    meta_info = np.empty(
        (len(unique_labels), meta.shape[1]), dtype=object
    )  # dtype=object 的意思是在 NumPy 数组中使用 Python 对象作为数据类型
    for curr_label, indices in cluster_label_to_indices.items():
        for col_idx in range(meta.shape[1]):
            if is_numeric_dtype(meta.iloc[indices, col_idx].dtype):
                # 如果是数值型数据
                meta_info[curr_label, col_idx] = meta.iloc[indices, col_idx].mean(
                    axis=0
                )
            else:
                # 如果是字符型或其他类型 则通过投票找到最可能的label
                true_labels = meta.iloc[:, col_idx]  # .tolist()
                res = summarize_clustering(labels, curr_label, true_labels)
                meta_info[curr_label, col_idx] = res

    meta_info = pd.DataFrame(meta_info, columns=sub_data.obs.columns)

    for i, column in enumerate(meta_info.columns):
        if type(meta_info.iloc[:, i][0]) == list:
            meta_info.iloc[:, i] = sum(meta_info.iloc[:, i], [])
        else:
            meta_info.iloc[:, i] = meta_info.iloc[:, i]

    # use scanpy functions to do the adata_sub construction
    adata_sub = ad.AnnData(centroids, obs=meta_info, dtype=np.float32)
    adata_raw = ad.AnnData(counts, dtype=np.float32)
    adata_sub.layers["counts"] = adata_raw.X.copy()
    adata_sub.var_names = sub_data.var_names
    del adata_raw
    return adata_sub


## summarize_clustering
def summarize_clustering(clustering, curr_label, true_labels):
    """
    Compute the majority cell type for each cluster.

    Parameters
    ----------
    clustering: np.array of shape (n_samples,)
        Clustering labels, coded from 0, 1, ..., n_clusters.
    true_labels: np.array of shape (n_samples,)
        Groundtruth labels.

    Returns
    -------
    np.array of shape (n_clusters,)
        The majority voting results.
    """
    res = []
    cluster_label_to_indices = defaultdict(list)
    for i, l in enumerate(clustering):
        cluster_label_to_indices[l].append(i)

    # 使用字典推导式提取curr_label的子字典
    #     if len(curr_label) > 1:
    #         sub_dict = {key: cluster_label_to_indices[key] for key in curr_label}
    #     else:
    sub_dict = {curr_label: cluster_label_to_indices[curr_label]}

    for key in sub_dict.keys():
        curr_indices = sub_dict[key]  # sub_dict.get(key)
        curr_true_labels = true_labels.iloc[curr_indices]
        # majority voting
        # randomly break the ties
        curr_true_labels = np.random.permutation(curr_true_labels)
        counter = Counter(curr_true_labels)
        most_common_element, _ = counter.most_common(1)[0]
        res.append(most_common_element)

    return res


def cdist_correlation(arr1, arr2):
    """Calculate pair-wise 1 - Pearson correlation between X and Y.

    Parameters
    ----------
    arr1: np.array of shape (n_samples1, n_features)
        First dataset.
    arr2: np.array of shape (n_samples2, n_features)
        Second dataset.

    Returns
    -------
    array-like of shape (n_samples1, n_samples2)
        The (i, j)-th entry is 1 - Pearson correlation between i-th row of arr1 and j-th row of arr2.
    """
    n, p = arr1.shape
    m, p2 = arr2.shape
    assert p2 == p

    arr1 = (arr1.T - np.mean(arr1, axis=1)).T
    arr2 = (arr2.T - np.mean(arr2, axis=1)).T

    arr1 = (arr1.T / np.sqrt(1e-6 + np.sum(arr1**2, axis=1))).T
    arr2 = (arr2.T / np.sqrt(1e-6 + np.sum(arr2**2, axis=1))).T

    return 1 - arr1 @ arr2.T


def convert_to_numpy(arr):
    if isinstance(arr, csr_matrix):
        return arr.toarray()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError("Unsupported data type.")


def drop_zero_variability_columns(arr_lst: list, tol=1e-8):
    """
    Drop columns for which its standard deviation is zero in any one of the arrays in arr_list.

    Parameters
    ----------
    arr_lst: list of np.array
        List of arrays
    tol: float, default=1e-8
        Any number less than tol is considered as zero

    Returns
    -------
    List of np.array where no column has zero standard deviation
    """
    bad_columns = set()
    for arr in arr_lst:
        curr_std = np.std(arr, axis=0)
        for col in np.nonzero(np.abs(curr_std) < tol)[0]:
            bad_columns.add(col)
    good_columns = [i for i in range(arr_lst[0].shape[1]) if i not in bad_columns]
    return [arr[:, good_columns] for arr in arr_lst]


def robust_svd(arr, n_components, randomized=False, n_runs=1):
    """
    Do deterministic or randomized SVD on arr.

    Parameters
    ----------
    arr: np.array
        The array to do SVD on
    n_components: int
        Number of SVD components
    randomized: bool, default=False
        Whether to run randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error

    Returns
    -------
    u, s, vh: np.array
        u @ np.diag(s) @ vh is the reconstruction of the original arr
    """
    if randomized:
        best_err = float("inf")
        u, s, vh = None, None, None
        for _ in range(n_runs):
            curr_u, curr_s, curr_vh = randomized_svd(
                arr, n_components=n_components, random_state=None
            )
            curr_err = np.sum((arr - curr_u @ np.diag(curr_s) @ curr_vh) ** 2)
            if curr_err < best_err:
                best_err = curr_err
                u, s, vh = curr_u, curr_s, curr_vh
        assert u is not None and s is not None and vh is not None
    else:
        if n_runs > 1:
            warnings.warn("Doing deterministic SVD, n_runs reset to one.")
        u, s, vh = svds(arr * 1.0, k=n_components)  # svds can not handle integer values
    return u, s, vh


def svd_denoise(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute best rank-n_components approximation of arr by SVD.

    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error

    Returns
    -------
    arr: array_like of shape (n_samples, n_features)
        Rank-n_comopnents approximation of the input arr.
    """
    if n_components is None:
        return arr
    u, s, vh = robust_svd(
        arr, n_components=n_components, randomized=randomized, n_runs=n_runs
    )
    return u @ np.diag(s) @ vh


def svd_embedding(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute rank-n_components SVD embeddings of arr.

    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error

    Returns
    -------
    embeddings: array_like of shape (n_samples, n_components)
        Rank-n_comopnents SVD embedding of arr.
    """
    n_components = min(n_components, min(arr.shape) - 1)  # 确保 n_components 小于矩阵最小维度

    if n_components is None:
        return arr
    u, s, vh = robust_svd(
        arr, n_components=n_components, randomized=randomized, n_runs=n_runs
    )
    return u @ np.diag(s)


def center_scale(arr):
    """
    Column-wise center and scale by standard deviation.

    Parameters
    ----------
    arr: np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    Center and scaled version of arr.
    """
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def filter_bad_matches(matching, filter_prop=0.1):
    """
    Filter bad matches according to the distances of matched pairs.

    Parameters
    ----------
    matching: list
        rows, cols, vals = init_matching, where each matched pair is (rows[i], cols[i]),
        and their distance is vals[i]
    filter_prop: float
        Matched pairs with distance in top filter_prop are discarded
    Returns
    -------
    rows, cols, vals: list
        Each matched pair of rows[i], cols[i], their distance is vals[i]
    """
    init_rows, init_cols, init_vals = matching
    thresh = np.quantile(init_vals, 1 - filter_prop)
    rows = []
    cols = []
    vals = []
    for i, j, val in zip(init_rows, init_cols, init_vals):
        if val < thresh:
            rows.append(i)
            cols.append(j)
            vals.append(val)
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float32),
    )


def pearson_correlation(arr1, arr2):
    """Calculate the vector of pearson correlations between each row of arr1 and arr2.

    Parameters
    ----------
    arr1: np.array of shape (n_samples, n_features)
        First dataset.
    arr2: np.array of shape (n_samples, n_features)
        Second dataset.

    Returns
    -------
    np.array of shape (n_samples,), the i-th entry is the pearson correlation between arr1[i, :] and arr2[i, :].
    """
    n, p = arr1.shape
    m, p2 = arr2.shape
    assert n == m and p2 == p

    arr1 = (arr1.T - np.mean(arr1, axis=1)).T
    arr2 = (arr2.T - np.mean(arr2, axis=1)).T

    arr1 = (arr1.T / np.sqrt(1e-6 + np.sum(arr1**2, axis=1))).T
    arr2 = (arr2.T / np.sqrt(1e-6 + np.sum(arr2**2, axis=1))).T

    return np.sum(arr1 * arr2, axis=1)


def cca_embedding(arr1, arr2, init_matching, filter_prop, n_components, max_iter=2000):
    """
    Filter bad matched pairs, align arr1 and arr2 using init_matching, fit CCA, and get CCA embeddings of arr1 and arr2.

    Parameters
    ----------
    arr1: np.ndarray of shape (n_samples1, n_features1)
        The first data matrix
    arr2: np.ndarray of shape (n_samples2, n_features2)
        The second data matrix
    init_matching: list
        rows, cols, vals = init_matching, where each matched pair is (rows[i], cols[i]),
        and their distance is vals[i]
    filter_prop: float
        Matched pairs with distance in top filter_prop are discarded when fitting CCA
    n_components: int
        Number of components to keep when fitting CCA
    max_iter: int, default=2000
        Maximum number of iterations for CCA

    Returns
    -------
    arr1_cca: np.array of shape (n_samples1, n_components)
    arr2_cca: np.array of shape (n_samples2, n_components)
    canonical_correlations: np.array of shape (n_components,)
    """

    # filter bad matched pairs
    arr1_indices, arr2_indices, _ = filter_bad_matches(init_matching, filter_prop)

    # align
    arr1_aligned = arr1[arr1_indices, :]
    arr2_aligned = arr2[arr2_indices, :]

    # cca
    cca = CCA(n_components=n_components, max_iter=max_iter)
    cca.fit(arr1_aligned, arr2_aligned)
    arr1_aligned_cca, arr2_aligned_cca = cca.transform(arr1_aligned, arr2_aligned)
    arr1_aligned_cca = center_scale(arr1_aligned_cca)
    arr2_aligned_cca = center_scale(arr2_aligned_cca)

    canonical_correlations = np.corrcoef(
        arr1_aligned_cca, arr2_aligned_cca, rowvar=False
    ).diagonal(offset=n_components)
    arr1_cca, arr2_cca = cca.transform(arr1, arr2)
    arr1_cca = center_scale(arr1_cca)
    arr2_cca = center_scale(arr2_cca)

    return arr1_cca, arr2_cca, canonical_correlations


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    def seurat_clr(x):
        """A modified CLR function that supports sparse input"""
        if scipy.sparse.issparse(x):
            # Only take non-zero entries for calculation
            nonzero_vals = x.data
            log_nonzero_vals = np.log1p(nonzero_vals)
            s = np.sum(log_nonzero_vals)
            cell_count = x.shape[1]
            exp = np.exp(s / cell_count)
            x.data = np.log1p(nonzero_vals / exp)
            return x
        else:
            # Dense array handling (as fallback)
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    if scipy.sparse.issparse(adata.X):
        # Apply the CLR function to each row of the sparse matrix
        adata.X = scipy.sparse.vstack([seurat_clr(row) for row in adata.X])
    else:
        # Apply the CLR function to each row of the dense matrix
        adata.X = np.apply_along_axis(seurat_clr, 1, adata.X)

    return adata


def tfidf(X, n_components, binarize=True, random_state=0):
    import sklearn

    sc_count = np.copy(X)
    if binarize:
        sc_count = np.where(sc_count < 1, sc_count, 1)

    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    normed_count = tfidf.fit_transform(sc_count)

    lsi = sklearn.decomposition.TruncatedSVD(
        n_components=n_components, random_state=random_state
    )
    lsi_r = lsi.fit_transform(normed_count)

    X_lsi = lsi_r[:, 1:]

    return normed_count, X_lsi


def TFIDF_LSI(adata, n_comps=50, binarize=True, random_state=0):
    """
    Computes LSI based on a TF-IDF transformation of the data from MultiMap. Putative dimensionality
    reduction for scATAC-seq data. Adds an ``.obsm['X_lsi']`` field to the object it was ran on.

    Input
    -----
    adata : ``AnnData``
        The object to run TFIDF + LSI on. Will use ``.X`` as the input data.
    n_comps : ``int``
        The number of components to generate. Default: 50
    binarize : ``bool``
        Whether to binarize the data prior to the computation. Often done during scATAC-seq
        processing. Default: True
    random_state : ``int``
        The seed to use for randon number generation. Default: 0
    """

    # this is just a very basic wrapper for the non-adata function
    import scipy

    if scipy.sparse.issparse(adata.X):
        adata.X, adata.obsm["X_lsi"] = tfidf(
            adata.X.todense(),
            n_components=n_comps,
            binarize=binarize,
            random_state=random_state,
        )
    else:
        adata.X, adata.obsm["X_lsi"] = tfidf(
            adata.X, n_components=n_comps, binarize=binarize, random_state=random_state
        )


## Predict gene scores based on chromatin accessibility
## issue https://www.biostars.org/p/114460/ 需要确保bedtools被安装了。
def _uniquify(seq, sep="-"):
    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """

    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += sep + str(dups[val][1])

    return seq


class GeneScores:
    """A class used to represent gene scores

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        adata,
        genome,
        gene_anno=None,
        tss_upstream=1e5,
        tss_downsteam=1e5,
        gb_upstream=5000,
        cutoff_weight=1,
        use_top_pcs=False,
        use_precomputed=True,
        use_gene_weight=True,
        min_w=1,
        max_w=5,
    ):
        """
        Parameters
        ----------
        adata: `Anndata`
            Input anndata
        genome : `str`
            The genome name
        """
        self.adata = adata
        self.genome = genome
        self.gene_anno = gene_anno
        self.tss_upstream = tss_upstream
        self.tss_downsteam = tss_downsteam
        self.gb_upstream = gb_upstream
        self.cutoff_weight = cutoff_weight
        self.use_top_pcs = use_top_pcs
        self.use_precomputed = use_precomputed
        self.use_gene_weight = use_gene_weight
        self.min_w = min_w
        self.max_w = max_w

    def _read_gene_anno(self):
        """Read in gene annotation

        Parameters
        ----------

        Returns
        -------

        """
        assert self.genome in [
            "hg19",
            "hg38",
            "mm9",
            "mm10",
        ], "`genome` must be one of ['hg19','hg38','mm9','mm10']"

        bin_str = pkgutil.get_data(
            "Garfield", f"data/gene_anno/{self.genome}_genes.bed"
        )
        gene_anno = pd.read_csv(
            io.BytesIO(bin_str),
            encoding="utf8",
            sep="\t",
            header=None,
            names=["chr", "start", "end", "symbol", "strand"],
        )
        self.gene_anno = gene_anno
        return self.gene_anno

    def _extend_tss(self, pbt_gene):
        """Extend transcription start site in both directions

        Parameters
        ----------

        Returns
        -------

        """
        ext_tss = pbt_gene
        if ext_tss["strand"] == "+":
            ext_tss.start = max(0, ext_tss.start - self.tss_upstream)
            ext_tss.end = max(ext_tss.end, ext_tss.start + self.tss_downsteam)
        else:
            ext_tss.start = max(0, min(ext_tss.start, ext_tss.end - self.tss_downsteam))
            ext_tss.end = ext_tss.end + self.tss_upstream
        return ext_tss

    def _extend_genebody(self, pbt_gene):
        """Extend gene body upstream

        Parameters
        ----------

        Returns
        -------

        """
        ext_gb = pbt_gene
        if ext_gb["strand"] == "+":
            ext_gb.start = max(0, ext_gb.start - self.gb_upstream)
        else:
            ext_gb.end = ext_gb.end + self.gb_upstream
        return ext_gb

    def _weight_genes(self):
        """Weight genes

        Parameters
        ----------

        Returns
        -------

        """
        gene_anno = self.gene_anno
        gene_size = gene_anno["end"] - gene_anno["start"]
        w = 1 / gene_size
        w_scaled = (self.max_w - self.min_w) * (w - min(w)) / (
            max(w) - min(w)
        ) + self.min_w
        return w_scaled

    def cal_gene_scores(self):
        """Calculate gene scores

        Parameters
        ----------

        Returns
        -------

        """
        adata = self.adata
        if self.gene_anno is None:
            gene_ann = self._read_gene_anno()
        else:
            gene_ann = self.gene_anno

        df_gene_ann = gene_ann.copy()
        df_gene_ann.index = _uniquify(df_gene_ann["symbol"].values)
        if self.use_top_pcs:
            mask_p = adata.var["top_pcs"]
        else:
            mask_p = pd.Series(True, index=adata.var_names)
        df_peaks = adata.var[mask_p][["chr", "start", "end"]].copy()

        if "gene_scores" not in adata.uns_keys():
            print("Gene scores are being calculated for the first time")
            print("`use_precomputed` has been ignored")
            self.use_precomputed = False

        if self.use_precomputed:
            print("Using precomputed overlap")
            df_overlap_updated = adata.uns["gene_scores"]["overlap"].copy()
        else:
            # add the fifth column
            # so that pybedtool can recognize the sixth column as the strand
            df_gene_ann_for_pbt = df_gene_ann.copy()
            df_gene_ann_for_pbt["score"] = 0
            df_gene_ann_for_pbt = df_gene_ann_for_pbt[
                ["chr", "start", "end", "symbol", "score", "strand"]
            ]
            df_gene_ann_for_pbt["id"] = range(df_gene_ann_for_pbt.shape[0])

            df_peaks_for_pbt = df_peaks.copy()
            df_peaks_for_pbt["id"] = range(df_peaks_for_pbt.shape[0])

            pbt_gene_ann = pybedtools.BedTool.from_dataframe(df_gene_ann_for_pbt)
            pbt_gene_ann_ext = pbt_gene_ann.each(self._extend_tss)
            pbt_gene_gb_ext = pbt_gene_ann.each(self._extend_genebody)

            pbt_peaks = pybedtools.BedTool.from_dataframe(df_peaks_for_pbt)

            # peaks overlapping with extended TSS
            pbt_overlap = pbt_peaks.intersect(pbt_gene_ann_ext, wa=True, wb=True)
            df_overlap = pbt_overlap.to_dataframe(
                names=[x + "_p" for x in df_peaks_for_pbt.columns]
                + [x + "_g" for x in df_gene_ann_for_pbt.columns]
            )
            # peaks overlapping with gene body
            pbt_overlap2 = pbt_peaks.intersect(pbt_gene_gb_ext, wa=True, wb=True)
            df_overlap2 = pbt_overlap2.to_dataframe(
                names=[x + "_p" for x in df_peaks_for_pbt.columns]
                + [x + "_g" for x in df_gene_ann_for_pbt.columns]
            )

            # add distance and weight for each overlap
            df_overlap_updated = df_overlap.copy()
            df_overlap_updated["dist"] = 0

            for i, x in enumerate(df_overlap["symbol_g"].unique()):
                # peaks within the extended TSS
                df_overlap_x = df_overlap[df_overlap["symbol_g"] == x].copy()
                # peaks within the gene body
                df_overlap2_x = df_overlap2[df_overlap2["symbol_g"] == x].copy()
                # peaks that are not intersecting with the promoter
                # and gene body of gene x
                id_overlap = df_overlap_x.index[
                    ~np.isin(df_overlap_x["id_p"], df_overlap2_x["id_p"])
                ]
                mask_x = df_gene_ann["symbol"] == x
                range_x = df_gene_ann[mask_x][["start", "end"]].values.flatten()
                if df_overlap_x["strand_g"].iloc[0] == "+":
                    df_overlap_updated.loc[id_overlap, "dist"] = pd.concat(
                        [
                            abs(df_overlap_x.loc[id_overlap, "start_p"] - (range_x[1])),
                            abs(
                                df_overlap_x.loc[id_overlap, "end_p"]
                                - max(0, range_x[0] - self.gb_upstream)
                            ),
                        ],
                        axis=1,
                        sort=False,
                    ).min(axis=1)
                else:
                    df_overlap_updated.loc[id_overlap, "dist"] = pd.concat(
                        [
                            abs(
                                df_overlap_x.loc[id_overlap, "start_p"]
                                - (range_x[1] + self.gb_upstream)
                            ),
                            abs(df_overlap_x.loc[id_overlap, "end_p"] - (range_x[0])),
                        ],
                        axis=1,
                        sort=False,
                    ).min(axis=1)

                n_batch = int(df_gene_ann_for_pbt.shape[0] / 5)
                if i % n_batch == 0:
                    print(f"Processing: {i/df_gene_ann_for_pbt.shape[0]:.1%}")
            df_overlap_updated["dist"] = df_overlap_updated["dist"].astype(float)

            adata.uns["gene_scores"] = dict()
            adata.uns["gene_scores"]["overlap"] = df_overlap_updated.copy()

        df_overlap_updated["weight"] = np.exp(
            -(df_overlap_updated["dist"].values / self.gb_upstream)
        )
        mask_w = df_overlap_updated["weight"] < self.cutoff_weight
        df_overlap_updated.loc[mask_w, "weight"] = 0
        # construct genes-by-peaks matrix
        mat_GP = csr_matrix(
            coo_matrix(
                (
                    df_overlap_updated["weight"],
                    (df_overlap_updated["id_g"], df_overlap_updated["id_p"]),
                ),
                shape=(df_gene_ann.shape[0], df_peaks.shape[0]),
            )
        )
        # adata_GP = ad.AnnData(X=csr_matrix(mat_GP),
        #                       obs=df_gene_ann,
        #                       var=df_peaks)
        # adata_GP.layers['weight'] = adata_GP.X.copy()
        if self.use_gene_weight:
            gene_weights = self._weight_genes()
            gene_scores = adata[:, mask_p].X * (mat_GP.T.multiply(gene_weights))
        else:
            gene_scores = adata[:, mask_p].X * mat_GP.T
        adata_CG_atac = ad.AnnData(
            gene_scores, obs=adata.obs.copy(), var=df_gene_ann.copy()
        )
        return adata_CG_atac


def gene_scores(
    adata,
    genome,
    gene_anno=None,
    tss_upstream=1e5,
    tss_downsteam=1e5,
    gb_upstream=5000,
    cutoff_weight=1,
    use_top_pcs=True,
    use_precomputed=True,
    use_gene_weight=True,
    min_w=1,
    max_w=5,
):
    """
    Calculate gene scores of scATACseq data

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genome : `str`
        Reference genome. Choose from {'hg19', 'hg38', 'mm9', 'mm10'}
    gene_anno : `pandas.DataFrame`, optional (default: None)
        Dataframe of gene annotation.
        If None, built-in gene annotation will be used depending on `genome`;
        If provided, custom gene annotation will be used instead.
    tss_upstream : `int`, optional (default: 1e5)
        The number of base pairs upstream of TSS
    tss_downsteam : `int`, optional (default: 1e5)
        The number of base pairs downstream of TSS
    gb_upstream : `int`, optional (default: 5000)
        The number of base pairs upstream by which gene body is extended.
        Peaks within the extended gene body are given the weight of 1.
    cutoff_weight : `float`, optional (default: 1)
        Weight cutoff for peaks
    use_top_pcs : `bool`, optional (default: True)
        If True, only peaks associated with top PCs will be used
    use_precomputed : `bool`, optional (default: True)
        If True, overlap bewteen peaks and genes
        (stored in `adata.uns['gene_scores']['overlap']`) will be imported
    use_gene_weight : `bool`, optional (default: True)
        If True, for each gene, the number of peaks assigned to it
        will be rescaled based on gene size
    min_w : `int`, optional (default: 1)
        The minimum weight for each gene.
        Only valid if `use_gene_weight` is True
    max_w : `int`, optional (default: 5)
        The maximum weight for each gene.
        Only valid if `use_gene_weight` is True

    Returns
    -------
    adata_new: AnnData
        Annotated data matrix.
        Stores #cells x #genes gene score matrix

    updates `adata` with the following fields.
    overlap: `pandas.DataFrame`, (`adata.uns['gene_scores']['overlap']`)
        Dataframe of overlap between peaks and genes
    """
    GS = GeneScores(
        adata,
        genome,
        gene_anno=gene_anno,
        tss_upstream=tss_upstream,
        tss_downsteam=tss_downsteam,
        gb_upstream=gb_upstream,
        cutoff_weight=cutoff_weight,
        use_top_pcs=use_top_pcs,
        use_precomputed=use_precomputed,
        use_gene_weight=use_gene_weight,
        min_w=min_w,
        max_w=max_w,
    )
    adata_CG_atac = GS.cal_gene_scores()
    return adata_CG_atac
