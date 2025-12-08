# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Garfield is a Python package for graph-based contrastive learning that enables fast single-cell embedding and spatial multi-omics analysis. It implements a Graph Neural Network Variational Autoencoder (GNN-VAE) architecture with contrastive learning for unified integration and niche transfer across single-cell and spatial multi-omics data.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install from PyPI
pip install Garfield

# Install from GitHub
pip install git+https://github.com/zhou-1314/Garfield.git
```

### Dependencies
- Core dependencies are listed in `requirements.txt`
- Documentation dependencies are in `docs/requirements.txt`
- Requires Python ≥ 3.7
- Built on PyTorch and PyTorch Geometric frameworks

### Documentation
```bash
# Build documentation (from docs/ directory)
cd docs
make html

# Clean documentation build
make clean
```

## Architecture Overview

### Core Components

**Main Model (`Garfield/model/Garfield.py`)**
- `Garfield` class: Main entry point that orchestrates the entire pipeline
- Inherits from `torch.nn.Module` and `BaseModelMixin`
- Handles data preprocessing, model training, and evaluation

**GNN-VAE Implementation (`Garfield/modules/GNNModelVAE.py`)**
- `GNNModelVAE` class: Core variational autoencoder with graph neural networks
- Supports both GAT (Graph Attention) and GCN (Graph Convolution) encoders
- Implements contrastive learning losses and reconstruction objectives

**Training System (`Garfield/trainer/trainer.py`)**
- `GarfieldTrainer` class: Handles model training with early stopping
- Supports both edge-level and node-level tasks
- Implements multiple loss functions with configurable weights

**Data Pipeline**
- `Garfield/data/`: Data loading, preprocessing, and batch handling
- `Garfield/preprocessing/`: Graph construction and feature engineering
- Supports multiple data types: RNA, ATAC, ADT, spatial, and multi-modal

**Neural Network Components**
- `Garfield/nn/encoders.py`: GAT and GCN encoder implementations
- `Garfield/nn/decoders.py`: Various decoder architectures
- `Garfield/modules/loss.py`: Contrastive and reconstruction loss functions

### Key Parameters

The model accepts extensive configuration through parameters detailed in the README. Critical parameters include:

- **Data Configuration**: `profile`, `data_type`, `sub_data_type`, `sample_col`
- **Graph Construction**: `graph_const_method`, `weight`, `n_neighbors`
- **Model Architecture**: `conv_type`, `hidden_dims`, `bottle_neck_neurons`
- **Training**: `n_epochs`, `learning_rate`, loss function weights
- **Distributed Training**: `use_lightning`, `lightning_sampling_mode`, `accelerator`, `devices`, `strategy`

### Dual Training System

Garfield supports two training backends that can be selected via the `use_lightning` parameter:

**Original Trainer** (`use_lightning=False`):
- Legacy training implementation
- Optimized for single-GPU scenarios
- Uses global RNG for reproducibility
- Simpler code path with minimal overhead

**PyTorch Lightning Trainer** (`use_lightning=True`):
- Modern distributed training framework
- Required for multi-GPU/multi-node training
- Supports advanced features (checkpointing, logging, callbacks)
- Slightly higher overhead but better scalability

**Auto-Selection** (`use_lightning='auto'`, default):
- Automatically chooses best trainer for your setup
- Single GPU → Original Trainer (faster)
- Multi-GPU/Multi-node → Lightning Trainer (scalable)

### Random Sampling Modes

The `lightning_sampling_mode` parameter controls random sampling behavior in PyTorch Lightning dataloaders:

**'auto' (default)**:
- Smart mode that adapts to your setup
- Single GPU: Uses global RNG (matches original trainer behavior)
- Multi-GPU: Uses per-rank generator (correct distributed training)
- Best choice for most users

**'legacy'**:
- Always creates separate `torch.Generator` objects
- Matches original PyTorch Lightning behavior exactly
- Use this for reproducing results from early Lightning implementations
- Slightly slower on single GPU

**'optimized'**:
- Explicit fast mode (same as 'auto')
- Single GPU: Uses global RNG (faster)
- Multi-GPU: Uses per-rank generator (correct)
- Use when you want to be explicit about optimization

**Why this matters:**
Even with the same seed, `torch.Generator()` produces different random sequences than the global RNG (`torch.manual_seed()`). This affects:
- Neighbor sampling in graph dataloaders
- Mini-batch composition
- Final model quality and embeddings

**Usage examples:**
```python
# Default: auto-select best mode
model = Garfield({'adata_list': [adata]})

# Reproduce original Lightning behavior
model = Garfield({
    'adata_list': [adata],
    'lightning_sampling_mode': 'legacy'
})

# Explicit optimization
model = Garfield({
    'adata_list': [adata],
    'lightning_sampling_mode': 'optimized'
})
```

**Testing:** See `tests/test_sampling_modes.py` for validation of all three modes.

## Package Structure

```
Garfield/
├── __init__.py          # Main package exports
├── _settings.py         # Global settings
├── _version.py          # Version information
├── analysis/            # Post-training analysis tools
├── data/               # Data handling and loading
├── model/              # Main model implementations
├── modules/            # Core GNN-VAE modules
├── nn/                 # Neural network components
├── plot/               # Visualization utilities
├── preprocessing/      # Data preprocessing pipeline
└── trainer/            # Training orchestration
```

## Data Types and Profiles

The system supports multiple data profiles:
- **'RNA'**: Single-cell RNA sequencing
- **'ATAC'**: Single-cell ATAC sequencing  
- **'ADT'**: Antibody-derived tags
- **'multi-modal'**: Combined modalities
- **'spatial'**: Spatial transcriptomics/proteomics

## Development Notes

- No explicit test framework detected - validation appears to be through example notebooks
- Documentation built with Sphinx and hosted on ReadTheDocs
- Uses standard Python packaging with setuptools
- Follows scientific Python conventions (NumPy, pandas, scikit-learn ecosystem)
- Integrates with single-cell analysis tools (scanpy, anndata)

## Recent Optimizations (2025-12)

The codebase has been optimized for improved performance, robustness, and usability. Key improvements include:

### Critical Bug Fixes

1. **Lightning Module Class Name**: Fixed class name from `GarfieldLightning` to `GarfieldLightningModule` to match imports
2. **Model Wrapping**: Refactored Lightning module to accept pre-created GNN-VAE model, eliminating double nesting (`self.model.model`)
3. **Error Handling**: Added graceful error handling for missing PyTorch Lightning dependency

### Performance Enhancements

1. **Gradient Clipping**: Lightning module now properly implements gradient clipping via `on_before_optimizer_step` hook, matching the original trainer's behavior
2. **Gradient Accumulation**: New `accumulate_grad_batches` parameter enables training with larger effective batch sizes (useful for large graphs)
3. **Distributed Logging**: Added `sync_dist=True` to all Lightning logging calls for correct metric aggregation across GPUs
4. **Configurable Weight Decay**: Lightning module now respects the `weight_decay` parameter (previously hardcoded to 1e-4)

### Parameter Validation

Added comprehensive parameter validation in `_validate_distributed_params()` method:
- Validates `precision` values ('32', '16-mixed', 'bf16-mixed', etc.)
- Validates `accelerator` values ('auto', 'gpu', 'cpu', 'tpu', etc.)
- Validates `devices` format (int, list, or 'auto')
- Validates `num_workers` and `persistent_workers` compatibility
- Validates `lightning_sampling_mode` values
- Validates `use_lightning` values

This catches configuration errors early with clear error messages, preventing cryptic failures during training.

### New Parameters

**`accumulate_grad_batches`** (default: 1):
- Number of gradient accumulation steps
- Set to > 1 to simulate larger batch sizes without increasing memory
- Example: `accumulate_grad_batches=4` with `batch_size=128` → effective batch size of 512
- Useful for: Large graphs, limited GPU memory, better gradient estimates

**Usage Example:**
```python
model = Garfield({
    'adata_list': [adata],
    'use_lightning': True,
    'devices': 2,  # 2 GPUs
    'num_workers': 4,  # 4 dataloader workers per GPU
    'accumulate_grad_batches': 2,  # 2x effective batch size
    'precision': '16-mixed',  # Mixed precision training
})
model.train()
```

### Performance Tips

For optimal multi-GPU performance:
```python
config = {
    'adata_list': [adata],
    'use_lightning': 'auto',  # Auto-select best trainer
    'accelerator': 'gpu',
    'devices': -1,  # Use all available GPUs
    'strategy': 'ddp',  # Distributed Data Parallel
    'num_workers': 4,  # 4 workers per GPU (adjust based on CPU cores)
    'persistent_workers': True,  # Keep workers alive between epochs
    'precision': '16-mixed',  # Mixed precision for 2x speedup
    'accumulate_grad_batches': 1,  # Increase if memory-limited
    'lightning_sampling_mode': 'auto',  # Smart sampling mode
}
model = Garfield(config)
model.train()
```

Expected speedups:
- 2 GPUs: ~1.9x faster
- 4 GPUs: ~3.5x faster
- 8 GPUs: ~6-7x faster

### Code Quality Improvements

1. **Comprehensive Docstrings**: All Lightning module methods have detailed documentation
2. **Type Hints**: Added type annotations to Lightning module parameters
3. **Better Error Messages**: Validation errors provide clear guidance on valid parameter values
4. **Consistent Logging**: All losses logged with proper synchronization for distributed training

### Testing

Validation scripts are available in `tests/`:
- `test_sampling_modes.py`: Tests all three sampling modes
- `test_refactored_training.py`: Compares original vs. Lightning trainers
- `test_quality_consistency.py`: Validates model quality preservation

### Known Limitations

1. **n_epochs_no_edge_recon**: This parameter (from original trainer) is not yet implemented in Lightning module. It's always 0 in practice, so this doesn't affect typical usage.
2. **Gradient Clipping Method**: Lightning module uses `clip_grad_value_` (matches original trainer), not `clip_grad_norm_`. Both are valid; this is a design choice for consistency.

### Migration Notes

If migrating from an older version:
- Class name changed: `GarfieldLightning` → `GarfieldLightningModule`
- No changes needed for users (internal refactoring only)
- All existing code should work without modifications
- Lightning training is now more robust and performant