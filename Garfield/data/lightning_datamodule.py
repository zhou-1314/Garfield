"""
PyTorch Lightning DataModule for Garfield, handling PyG graph data preparation
and DDP-safe dataloader creation.
"""
import math
import warnings
from typing import Optional

import torch
import pytorch_lightning as pl
from anndata import AnnData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from .dataprocessors import prepare_data
from .dual_loader import DualGraphDataLoader

# Ignore UserWarnings from PyTorch Geometric
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class GarfieldDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Garfield.

    Handles data preparation, train/val/test splits, and creation of
    DDP-safe PyG dataloaders for distributed training.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    label_name : str, optional
        Column name for labels (e.g., batch information).
    used_pca_feat : bool
        Whether to use PCA features or raw expression as node features.
    adj_key : str
        Key in adata.obsp for adjacency matrix (e.g., 'connectivities').
    edge_val_ratio : float
        Proportion of edges for validation.
    edge_test_ratio : float
        Proportion of edges for testing.
    node_val_ratio : float
        Proportion of nodes for validation.
    node_test_ratio : float
        Proportion of nodes for testing.
    augment_type : str
        Augmentation type ('svd', 'dropout', None).
    num_neighbors : int
        Number of neighbors to sample per hop.
    loaders_n_hops : int
        Number of hops for neighbor sampling.
    edge_batch_size : int
        Batch size for edge-level tasks.
    node_batch_size : int, optional
        Batch size for node-level tasks (auto-computed if None).
    num_workers : int
        Number of dataloader workers (default: 4).
    persistent_workers : bool
        Keep workers alive between epochs (default: False).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        adata: AnnData,
        label_name: Optional[str],
        used_pca_feat: bool,
        adj_key: str,
        edge_val_ratio: float,
        edge_test_ratio: float,
        node_val_ratio: float,
        node_test_ratio: float,
        augment_type: str,
        num_neighbors: int,
        loaders_n_hops: int,
        edge_batch_size: int,
        node_batch_size: Optional[int],
        num_workers: int = 4,
        persistent_workers: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Store parameters
        self.adata = adata
        self.label_name = label_name
        self.used_pca_feat = used_pca_feat
        self.adj_key = adj_key
        self.edge_val_ratio = edge_val_ratio
        self.edge_test_ratio = edge_test_ratio
        self.node_val_ratio = node_val_ratio
        self.node_test_ratio = node_test_ratio
        self.augment_type = augment_type
        self.num_neighbors = num_neighbors
        self.loaders_n_hops = loaders_n_hops
        self.edge_batch_size = edge_batch_size
        self.node_batch_size = node_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.seed = seed

        # Data objects (populated in setup())
        self.node_masked_data = None
        self.edge_train_data = None
        self.edge_val_data = None
        self.n_nodes_train = None
        self.n_nodes_val = None
        self.n_edges_train = None
        self.n_edges_val = None

    def setup(self, stage: Optional[str] = None):
        """
        Prepare data splits.

        Called on every process in distributed training. Each rank runs this
        independently to create its own data objects.

        Parameters
        ----------
        stage : str, optional
            'fit', 'validate', 'test', or 'predict'.
        """
        if stage == 'fit' or stage is None:
            print("\n--- DATA PREPARATION (Lightning DataModule) ---")

            # Prepare data splits (deterministic with seed)
            data_dict = prepare_data(
                adata=self.adata,
                label_name=self.label_name,
                used_pca_feat=self.used_pca_feat,
                adj_key=self.adj_key,
                edge_val_ratio=self.edge_val_ratio,
                edge_test_ratio=self.edge_test_ratio,
                node_val_ratio=self.node_val_ratio,
                node_test_ratio=self.node_test_ratio,
            )

            self.node_masked_data = data_dict["node_masked_data"]
            self.edge_train_data = data_dict["edge_train_data"]
            self.edge_val_data = data_dict["edge_val_data"]

            # Calculate dataset sizes
            self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
            self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
            self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
            self.n_edges_val = (self.edge_val_data.edge_label_index.size(1)
                               if self.edge_val_data is not None else 0)

            # Auto-compute node batch size if not specified
            if self.node_batch_size is None:
                if self.n_edges_train > 0 and self.n_nodes_train > 0:
                    self.node_batch_size = int(
                        self.edge_batch_size /
                        math.floor(self.n_edges_train / self.n_nodes_train)
                    )
                else:
                    self.node_batch_size = 256  # Default fallback

            print(f"Number of training nodes: {self.n_nodes_train}")
            print(f"Number of validation nodes: {self.n_nodes_val}")
            print(f"Number of training edges: {self.n_edges_train}")
            print(f"Number of validation edges: {self.n_edges_val}")
            print(f"Edge batch size: {self.edge_batch_size}")
            print(f"Node batch size: {self.node_batch_size}")

    def train_dataloader(self):
        """
        Create training dataloaders.

        Returns DualGraphDataLoader combining edge and node loaders.
        Each process gets a rank-specific random generator for different sampling.

        Returns
        -------
        DualGraphDataLoader
            Combined edge + node dataloader.
        """
        # Create rank-specific random generator for DDP
        generator = None
        if self.seed is not None:
            # Each rank gets a different seed to sample different subgraphs
            rank = 0
            if self.trainer is not None and hasattr(self.trainer, 'global_rank'):
                rank = self.trainer.global_rank
            worker_seed = self.seed + rank
            generator = torch.Generator()
            generator.manual_seed(worker_seed)

        # Edge-level dataloader (graph reconstruction)
        edge_loader = LinkNeighborLoader(
            self.edge_train_data,
            num_neighbors=[self.num_neighbors] * self.loaders_n_hops,
            batch_size=self.edge_batch_size,
            edge_label_index=self.edge_train_data.edge_label_index[
                :, self.edge_train_data.edge_label.bool()
            ],
            directed=False,
            shuffle=True,
            neg_sampling_ratio=1.0,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            generator=generator,
        )

        # Node-level dataloader (gene expression reconstruction)
        node_loader = NeighborLoader(
            self.node_masked_data,
            num_neighbors=[self.num_neighbors] * self.loaders_n_hops,
            batch_size=self.node_batch_size,
            directed=False,
            shuffle=True,
            input_nodes=self.node_masked_data.train_mask,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            generator=generator,
        )

        return DualGraphDataLoader(edge_loader, node_loader)

    def val_dataloader(self):
        """
        Create validation dataloaders.

        Returns
        -------
        DualGraphDataLoader or None
            Combined edge + node validation dataloader, or None if no validation set.
        """
        # Check if validation set exists
        if (self.edge_val_data is None or
            self.edge_val_data.edge_label.sum() == 0 or
            self.node_masked_data.val_mask.sum() == 0):
            return None

        # No shuffling or random sampling for validation
        edge_val_loader = LinkNeighborLoader(
            self.edge_val_data,
            num_neighbors=[self.num_neighbors] * self.loaders_n_hops,
            batch_size=self.edge_batch_size,
            edge_label_index=self.edge_val_data.edge_label_index[
                :, self.edge_val_data.edge_label.bool()
            ],
            directed=False,
            shuffle=False,  # No shuffling for validation
            neg_sampling_ratio=1.0,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

        node_val_loader = NeighborLoader(
            self.node_masked_data,
            num_neighbors=[self.num_neighbors] * self.loaders_n_hops,
            batch_size=self.node_batch_size,
            directed=False,
            shuffle=False,  # No shuffling for validation
            input_nodes=self.node_masked_data.val_mask,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

        return DualGraphDataLoader(edge_val_loader, node_val_loader)
