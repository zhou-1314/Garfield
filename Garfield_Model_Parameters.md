# Garfield Model Parameters Reference

This document provides a comprehensive reference for all configuration parameters in Garfield. Parameters are organized by category for easy navigation.

---

## Table of Contents

1. [Data Preprocessing Parameters](#data-preprocessing-parameters)
2. [Data Split Parameters](#data-split-parameters)
3. [Model Architecture Parameters](#model-architecture-parameters)
4. [Graph Construction Parameters](#graph-construction-parameters)
5. [Data Loader Parameters](#data-loader-parameters)
6. [Loss Function Parameters](#loss-function-parameters)
7. [Training Parameters](#training-parameters)
8. [Distributed Training Parameters](#distributed-training-parameters)
9. [Logging and Checkpointing Parameters](#logging-and-checkpointing-parameters)
10. [Advanced Parameters](#advanced-parameters)

---

## Data Preprocessing Parameters

### Core Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata_list` | `List[AnnData]` or `AnnData` | Required | Input data. Can be a single AnnData object or list of AnnData objects for batch integration. |
| `profile` | `str` | `'RNA'` | Data modality type. Options: `'RNA'`, `'ATAC'`, `'ADT'`, `'multi-modal'`, `'spatial'`. |
| `data_type` | `str` | `None` | Multi-omics data type. Options: `'Paired'`, `'UnPaired'`, or `None` for single-modality data. |
| `sub_data_type` | `List[str]` | `None` | Modalities in multi-modal data. Examples: `['rna', 'atac']`, `['rna', 'adt']`. |
| `sample_col` | `str` | `None` | Column name in `adata.obs` indicating batch/sample identifiers for batch correction. |

### Feature Selection and Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `used_hvg` | `bool` | `True` | Whether to select highly variable genes (HVGs) for dimensionality reduction. |
| `rna_n_top_features` | `int` | `2000` | Number of top HVGs to retain for RNA-seq data. |
| `atac_n_top_features` | `int` | `10000` | Number of top variable features for ATAC-seq data. |
| `min_features` | `int` | `100` | Minimum number of detected features required per cell (quality control). |
| `min_cells` | `int` | `3` | Minimum number of cells expressing a feature for it to be retained. |
| `keep_mt` | `bool` | `False` | Whether to retain mitochondrial genes (genes starting with 'MT-' or 'mt-'). |
| `target_sum` | `float` | `1e4` | Target library size for count normalization (total counts per cell after normalization). |

### Dimensionality Reduction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | `int` | `50` | Number of principal components (PCs) or latent semantic indexing (LSI) components to compute. |
| `used_pca_feat` | `bool` | `False` | Whether to use PCA/LSI features as encoder input. If `False`, uses raw normalized features. |
| `svd_solver` | `str` | `'arpack'` | SVD solver for PCA. Options: `'arpack'`, `'randomized'`, `'auto'`. |
| `use_top_pcs` | `bool` | `False` | Whether to use only top PCs for downstream analysis. |

### Graph Construction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_const_method` | `str` | `'mu_std'` | Graph construction method. Options: <br>• `'mu_std'`: Statistical distance-based (recommended for spatial)<br>• `'Radius'`: Radius-based neighborhood<br>• `'KNN'`: K-nearest neighbors<br>• `'Squidpy'`: Squidpy spatial graph<br>• `'Scanpy'`: Scanpy's neighbor graph (for non-spatial) |
| `n_neighbors` | `int` | `15` | Number of nearest neighbors for graph construction. |
| `metric` | `str` | `'euclidean'` | Distance metric for neighbor search. Options: `'euclidean'`, `'cosine'`, `'manhattan'`, etc. |
| `weight` | `float` | `0.8` | Graph weighting factor (0-1). <br>• For **spatial data**: Weight of spatial graph vs. molecular similarity graph (1-weight).<br>• For **multi-modal data**: Weight of primary modality graph vs. secondary modality graph. |
| `adj_key` | `str` | `'spatial_connectivities'` | Key in `adata.obsp` or `adata.uns` storing the adjacency matrix. |

### Additional Preprocessing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genome` | `str` | `None` | Reference genome annotation (e.g., `'hg38'`, `'mm10'`). Used for ATAC-seq peak annotation. |
| `use_gene_weight` | `bool` | `True` | Whether to apply gene-specific weights based on expression variability. |

---

## Data Split Parameters

Control how data is split into training, validation, and test sets for model evaluation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edge_val_ratio` | `float` | `0.1` | Proportion of edges (0-1) reserved for validation in edge reconstruction task. |
| `edge_test_ratio` | `float` | `0.0` | Proportion of edges reserved for testing. Typically set to 0 for transfer learning. |
| `node_val_ratio` | `float` | `0.1` | Proportion of nodes (cells) reserved for validation in gene expression reconstruction. |
| `node_test_ratio` | `float` | `0.0` | Proportion of nodes reserved for testing. |

**Note**: Validation sets are required for early stopping and model evaluation metrics (AUROC, AUPRC, etc.). Set both `edge_val_ratio` and `node_val_ratio` to > 0 to enable evaluation.

---

## Model Architecture Parameters

### Encoder Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conv_type` | `str` | `'GAT'` | Type of graph convolutional layer. Options:<br>• `'GAT'`: Graph Attention Network (recommended)<br>• `'GATv2Conv'`: GATv2 with improved attention<br>• `'GCN'`: Graph Convolutional Network |
| `gnn_layer` | `int` | `2` | Number of stacked graph convolutional layers in the encoder. |
| `use_FCencoder` | `bool` | `True` | Whether to add a fully connected layer before graph layers for initial feature transformation. |
| `hidden_dims` | `List[int]` | `[128, 128]` | Hidden layer dimensions. Length should match `gnn_layer`. Example: `[256, 128]` for 2 layers. |
| `bottle_neck_neurons` | `int` | `20` | Dimensionality of the latent embedding space (bottleneck layer). |

### Attention Mechanism (for GAT)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | `3` | Number of attention heads in each GAT layer for multi-head attention. |
| `concat` | `bool` | `True` | Whether to concatenate (`True`) or average (`False`) attention head outputs. |
| `used_edge_weight` | `bool` | `True` | Whether to incorporate edge weights into graph convolution operations. |

### Regularization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dropout` | `float` | `0.2` | Dropout rate (0-1) applied to hidden layers during training to prevent overfitting. |
| `drop_feature_rate` | `float` | `0.2` | Feature dropout rate for data augmentation (drops node features randomly). |
| `drop_edge_rate` | `float` | `0.2` | Edge dropout rate for graph augmentation (drops edges randomly). |

### Advanced Architecture Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `augment_type` | `str` | `'svd'` | Data augmentation method for contrastive learning:<br>• `'svd'`: Low-rank SVD approximation (stable)<br>• `'dropout'`: Feature dropout augmentation<br>• `None`: No augmentation |
| `svd_q` | `int` | `5` | Rank parameter for SVD augmentation (only used if `augment_type='svd'`). |
| `cluster_num` | `int` | `20` | Number of clusters for cluster-level contrastive learning in latent space. |
| `used_DSBN` | `bool` | `False` | Whether to use Domain-Specific Batch Normalization for batch effect correction. |

---

## Graph Construction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_neighbors` | `int` | `3` | Number of neighbor nodes to sample in graph-based mini-batch loaders. |
| `loaders_n_hops` | `int` | `3` | Number of hops (neighborhood expansion) for subgraph sampling during training. |

**Explanation**: Garfield uses mini-batch graph sampling to handle large datasets. These parameters control the receptive field size.

---

## Data Loader Parameters

Configure mini-batch data loading for efficient training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edge_batch_size` | `int` | `256` | Number of edges sampled per batch for edge reconstruction task. |
| `node_batch_size` | `int` | `None` | Number of nodes (cells) per batch for node-level task. If `None`, uses all nodes. |
| `num_workers` | `int` | `0` | Number of parallel workers for data loading. Set to 0 for single-process loading, or 2-8 for multi-process (faster on multi-GPU). |
| `persistent_workers` | `bool` | `False` | Whether to keep data loading workers alive between epochs. Set to `True` for faster multi-GPU training when `num_workers > 0`. |

**Performance Tip**: For multi-GPU training, set `num_workers=4` and `persistent_workers=True` for optimal performance.

---

## Loss Function Parameters

Garfield uses a composite loss function with multiple weighted components. These hyperparameters control the contribution of each loss term.

### Loss Components

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_edge_recon_loss` | `bool` | `True` | Whether to include edge reconstruction loss (graph structure preservation). |
| `include_gene_expr_recon_loss` | `bool` | `True` | Whether to include gene expression reconstruction loss (feature reconstruction). |
| `used_mmd` | `bool` | `False` | Whether to use Maximum Mean Discrepancy (MMD) loss for batch effect removal. |

### Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_edge_recon` | `float` | `500.0` | Weight for edge (graph structure) reconstruction loss. |
| `lambda_gene_expr_recon` | `float` | `300.0` | Weight for gene expression (node feature) reconstruction loss. |
| `lambda_latent_adj_recon_loss` | `float` | `1.0` | Weight for latent adjacency matrix reconstruction loss. |
| `lambda_latent_contrastive_instanceloss` | `float` | `1.0` | Weight for instance-level contrastive loss (cell-cell similarity). |
| `lambda_latent_contrastive_clusterloss` | `float` | `0.5` | Weight for cluster-level contrastive loss (cluster separation). |
| `lambda_omics_recon_mmd_loss` | `float` | `0.2` | Weight for MMD loss in omics reconstruction (batch correction). |

**Note**: Loss weights are task-dependent and may require tuning. Higher weights emphasize that particular objective.

---

## Training Parameters

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_epochs` | `int` | `100` | Maximum number of training epochs. |
| `n_epochs_no_edge_recon` | `int` | `0` | Number of initial epochs to train without edge reconstruction loss (warm-up phase). |
| `learning_rate` | `float` | `0.001` | Initial learning rate for AdamW optimizer. |
| `weight_decay` | `float` | `1e-5` | L2 regularization weight decay parameter for AdamW optimizer. |
| `gradient_clipping` | `float` | `5.0` | Maximum gradient norm for gradient clipping. Set to 0 to disable. |

### Early Stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_early_stopping` | `bool` | `True` | Whether to enable early stopping based on validation loss plateau. |
| `early_stopping_kwargs` | `dict` | `None` | Dictionary of early stopping parameters. Example:<br>`{'patience': 10, 'min_delta': 0.001, 'metric': 'val_loss'}` |
| `reload_best_model` | `bool` | `True` | Whether to reload the best model (based on validation loss) after training completes. |

### Reproducibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `2024` | Random seed for reproducibility across runs. Sets seeds for NumPy, PyTorch, and Python random. |

---

## Distributed Training Parameters

Garfield supports both single-GPU and multi-GPU distributed training via PyTorch Lightning.

### Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accelerator` | `str` | `'auto'` | Hardware accelerator type. Options:<br>• `'auto'`: Auto-detect available hardware<br>• `'gpu'`: Force GPU usage<br>• `'cpu'`: Force CPU usage<br>• `'tpu'`: Use TPU (if available) |
| `devices` | `int` or `List[int]` or `str` | `1` | Number or list of devices to use. Examples:<br>• `1`: Single device<br>• `4`: 4 devices<br>• `[0, 1, 2, 3]`: Specific GPU IDs<br>• `'auto'`: Use all available devices |
| `num_nodes` | `int` | `1` | Number of compute nodes for multi-node distributed training (advanced). |

### Training Strategy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `'auto'` | Distributed training strategy. Options:<br>• `'auto'`: Auto-select based on hardware<br>• `'ddp'`: DistributedDataParallel (recommended for multi-GPU)<br>• `'ddp_find_unused_parameters_true'`: **Required for Garfield** (dual-decoder architecture)<br>• `'ddp_spawn'`: DDP with process spawning |
| `precision` | `str` | `'32'` | Numerical precision. Options:<br>• `'32'`: Full precision (FP32)<br>• `'16-mixed'`: Mixed precision (faster, recommended for modern GPUs)<br>• `'bf16-mixed'`: BFloat16 mixed precision<br>• `'64'`: Double precision |

### Training Backend Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lightning` | `str` or `bool` | `'auto'` | Training backend selection:<br>• `'auto'`: Use original trainer for single-GPU, Lightning for multi-GPU (recommended)<br>• `True`: Force PyTorch Lightning trainer<br>• `False`: Force original trainer (fastest for single-GPU) |
| `lightning_sampling_mode` | `str` | `'auto'` | Random sampling mode for Lightning dataloaders:<br>• `'auto'`: Optimized mode (single-GPU: global RNG, multi-GPU: per-rank generators)<br>• `'legacy'`: Original Lightning behavior<br>• `'optimized'`: Explicit fast mode |

### Performance Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accumulate_grad_batches` | `int` | `1` | Number of batches for gradient accumulation. Effective batch size = `batch_size × accumulate_grad_batches × num_gpus`. Useful for large models with limited GPU memory. |

**Important Note for Multi-GPU**: Due to Garfield's dual-decoder architecture (edge + node decoders), you **must** use `strategy='ddp_find_unused_parameters_true'` for multi-GPU training to avoid DDP errors.

---

## Logging and Checkpointing Parameters

### Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logger` | `str` | `'tensorboard'` | Logging backend. Options:<br>• `'tensorboard'`: TensorBoard logging (recommended)<br>• `'wandb'`: Weights & Biases<br>• `'csv'`: CSV file logging (lightweight)<br>• `None`: Disable logging |
| `log_every_n_steps` | `int` | `50` | Frequency of logging (every N training steps). |
| `log_style` | `str` | `'auto'` | Progress bar style:<br>• `'auto'`: Auto-detect (notebook vs. terminal)<br>• `'notebook'`: Jupyter notebook-friendly progress bar<br>• `'lightning'`: Standard PyTorch Lightning progress bar |
| `monitor` | `bool` | `True` | Whether to display training progress and metrics. |
| `verbose` | `bool` | `False` | Whether to display detailed debug information during training. |

### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | `str` | `None` | Directory for saving model checkpoints. If `None`, defaults to `<workdir>/checkpoints`. |
| `save_top_k` | `int` | `1` | Number of best models to keep (based on validation loss). Set to 0 to disable. |
| `save_last` | `bool` | `True` | Whether to always save the last checkpoint (regardless of performance). **Performance tip**: Set to `False` for faster training. |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_key` | `str` | `'garfield_latent'` | Key name for storing latent embeddings in `adata.obsm`. |

---

## Advanced Parameters

### Debugging and Development

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_dev_run` | `bool` | `False` | If `True`, runs only 1 batch for training and validation (for debugging). |
| `limit_train_batches` | `float` | `1.0` | Fraction of training data to use (0-1). Useful for debugging or quick experiments. |
| `limit_val_batches` | `float` | `1.0` | Fraction of validation data to use (0-1). |

### Legacy Parameters (Deprecated)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_id` | `int` | `None` | **DEPRECATED**. Use `accelerator='gpu'` and `devices=[device_id]` instead. |

---

## Example Configurations

### Single-GPU Training (Default)

```python
import Garfield as gf

config = {
    'adata_list': adata,
    'profile': 'RNA',
    'n_epochs': 100,
    'use_lightning': False,  # Use faster original trainer
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()
```

### Multi-GPU Training (8 GPUs)

```python
config = {
    'adata_list': adata,
    'profile': 'spatial',

    # Multi-GPU configuration
    'accelerator': 'gpu',
    'devices': 8,  # Use all 8 GPUs
    'strategy': 'ddp_find_unused_parameters_true',  # Required!
    'num_workers': 4,
    'persistent_workers': True,

    # Training parameters
    'n_epochs': 100,
    'precision': '16-mixed',  # Faster training
    'use_lightning': True,
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()
```

### Spatial Transcriptomics

```python
config = {
    'adata_list': adata,
    'profile': 'spatial',
    'graph_const_method': 'mu_std',
    'weight': 0.5,  # Balance spatial vs. molecular graphs
    'n_neighbors': 5,
    'used_hvg': True,
    'rna_n_top_features': 3000,
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()
```

---

## References

For more information, see:
- [Garfield Documentation](https://garfield-bio.readthedocs.io/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

---

**Last Updated**: 2025-12-08
**Version**: Compatible with Garfield v1.0.0+
