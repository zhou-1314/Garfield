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