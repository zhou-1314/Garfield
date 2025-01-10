from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def calc_neighbor_prop(
        adata,
        batch_key='replicates',
        celltype_key='Cluster',
        n_neighbors=25,
        spatial_key='spatial',
        output_key=None
):
    """
    Normalize the cell type abundance based on the nearest neighbors for each batch in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the spatial data and cell type information.
    batch_key : str, optional
        The key in `adata.obs` to identify different batches (default is 'replicates').
    celltype_key : str, optional
        The key in `adata.obs` representing the cell types (default is 'Cluster').
    n_neighbors : int, optional
        The number of nearest neighbors to consider for each cell (default is 25).
    spatial_key : str, optional
        The key in `adata.obsm` containing the spatial coordinates (default is 'spatial').
    output_key : str, optional
        The key to store the normalized cell type abundances in `adata.obsm` (default is None).

    Returns
    -------
    adata : AnnData
        The updated AnnData object with normalized cell type abundances added to `obsm`.
    """

    # Create storage for normalized cell type abundances
    cell_counts = {}

    # Loop through unique batches
    for b in adata.obs[batch_key].unique():
        # Get spatial coordinates and cell types for the current batch
        batch_data = adata[adata.obs[batch_key] == b]
        X = batch_data.obsm[spatial_key]
        celltypes = batch_data.obs[celltype_key].astype(str).values
        cellnames = batch_data.obs_names

        # Compute the nearest neighbors
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X)
        knn_indices = knn.kneighbors(X, return_distance=False)
        knn_celltypes = celltypes[knn_indices]  # Get the cell types of nearest neighbors

        # Process each cell in the batch
        for i in range(len(cellnames)):
            # Count the types of neighboring cells
            unique, counts = np.unique(knn_celltypes[i, :], return_counts=True)
            cell_counts[cellnames[i]] = dict(zip(unique, counts))

            # Get the total number of neighbors
            total_neighbors = sum(counts)

            # Normalize the counts (abundance)
            if total_neighbors > 0:
                normalized_counts = counts / total_neighbors  # Normalize to range [0, 1]
                cell_counts[cellnames[i]] = dict(zip(unique, normalized_counts))

    # Store the results in `obsm`
    if output_key is None:
        output_key = f'k{n_neighbors}_neighbours_celltype_normalized'

    adata.obsm[output_key] = pd.DataFrame(cell_counts).T.fillna(0)

    return adata
