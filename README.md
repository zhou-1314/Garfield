# Garfield: Graph-based Contrastive Learning for Fast Single-Cell Embedding

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/zhou-1314/Garfield/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/garfield)](https://pypi.org/project/garfield)
[![PyPI Downloads](https://static.pepy.tech/badge/garfield)](https://pepy.tech/projects/garfield)
[![Docs](https://readthedocs.org/projects/garfield-bio/badge/?version=latest)](https://garfield-bio.readthedocs.io/en/latest/?badge=latest)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Garfield** is a graph neural network-based variational autoencoder (GNN-VAE) framework that enables unified integration and niche transfer across single-cell and spatial multi-omics data through graph-based contrastive learning.

This repository contains the official PyTorch implementation of **Garfield** described in our paper: [Graph-based Contrastive Learning Enables Unified Integration and Niche Transfer Across Single-Cell and Spatial Multi-Omics](https://www.biorxiv.org/content/10.1101/2025.02.19.638965v1).

<p align="center">
<img src="imgs/Garfield_framework.png" alt="Garfield Framework" width="900"/>
</p>

---

## 🚀 Key Features

- **Unified Framework**: Integrates single-cell RNA-seq, ATAC-seq, ADT, and spatial transcriptomics/proteomics data
- **Graph-Based Learning**: Leverages graph neural networks with contrastive learning for robust cell representations
- **Dual-Task Architecture**: Simultaneously learns from graph structure and molecular features
- **Multi-GPU Support**: Distributed training with PyTorch Lightning for large-scale datasets (NEW in v1.0.0+)
- **Fast Inference**: Optimized for speed with optional graph construction optimizations
- **Flexible**: Supports various data modalities, batch correction, and transfer learning

---

## 📰 News

**January 28, 2025:** We officially released v1.0.0 of Garfield with multi-GPU distributed training support!

---

## 📦 Installation

### Quick Install from PyPI

```bash
pip install Garfield
```

### Install from GitHub (Latest Development Version)

```bash
pip install git+https://github.com/zhou-1314/Garfield.git
```

### Install from Source

```bash
git clone https://github.com/zhou-1314/Garfield.git
cd Garfield
pip install -e .
```

### Requirements

Garfield requires Python ≥ 3.7 and is built on the PyTorch framework. Key dependencies:
- `torch >= 1.10.0`
- `torch-geometric >= 2.0.0`
- `scanpy >= 1.8.0`
- `anndata >= 0.7.0`
- `pytorch-lightning >= 2.0.0` (for multi-GPU training)

For the complete list of dependencies, see `requirements.txt`.

---

## 🚦 Quick Start

### Basic Usage

```python
import scanpy as sc
import Garfield as gf

# Load your data
adata = sc.read_h5ad('your_data.h5ad')

# Configure Garfield
config = {
    'adata_list': adata,
    'profile': 'RNA',  # or 'spatial', 'ATAC', 'ADT', 'multi-modal'
    'n_epochs': 100,
}

# Create and train model
model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()

# Access embeddings
embeddings = adata.obsm['garfield_latent']
```

### Spatial Transcriptomics

```python
import scanpy as sc
import Garfield as gf

# Load spatial data
adata = sc.read_h5ad('spatial_data.h5ad')
adata.X = adata.layers['counts'].copy()

# Configure for spatial data
config = {
    'adata_list': adata,
    'profile': 'spatial',
    'graph_const_method': 'mu_std',  # Spatial graph construction
    'weight': 0.5,  # Balance spatial vs. molecular graphs
    'n_neighbors': 5,
    'rna_n_top_features': 3000,
    'n_epochs': 100,
}

# Train model
model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()

# Downstream analysis
sc.pp.neighbors(adata, use_rep='garfield_latent')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
```

### Multi-GPU Training

For large datasets, leverage multi-GPU distributed training:

```python
import Garfield as gf

config = {
    'adata_list': adata,
    'profile': 'spatial',

    # Multi-GPU configuration
    'accelerator': 'gpu',
    'devices': 8,  # Use all 8 GPUs
    'strategy': 'ddp_find_unused_parameters_true',  # Required for Garfield
    'num_workers': 4,  # Data loading workers per GPU
    'persistent_workers': True,

    # Training parameters
    'n_epochs': 100,
    'precision': '16-mixed',  # Mixed precision for 2x speedup
    'batch_size': 4096,
    'use_lightning': True,
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()  # Automatic distributed training across 8 GPUs
```

**Expected Performance**:
- 2 GPUs: ~1.9x faster
- 4 GPUs: ~3.5x faster
- 8 GPUs: ~6-7x faster

**Important**: Due to Garfield's dual-decoder architecture, multi-GPU training requires `strategy='ddp_find_unused_parameters_true'`.

---

## 📖 Documentation

### Complete Documentation

Visit our [ReadTheDocs page](https://garfield-bio.readthedocs.io/en/latest/) for:
- Detailed tutorials
- API reference
- Use case examples
- Troubleshooting guide

### Model Parameters

For a comprehensive list of all configuration parameters, see:
- [Garfield_Model_Parameters.md](Garfield_Model_Parameters.md) - Detailed parameter reference
- [Online Docs](https://garfield-bio.readthedocs.io/en/latest/) - Interactive documentation

### Key Parameter Categories

| Category | Key Parameters | Description |
|----------|----------------|-------------|
| **Data** | `adata_list`, `profile`, `sample_col` | Input data and modality configuration |
| **Preprocessing** | `used_hvg`, `target_sum`, `n_components` | Feature selection and normalization |
| **Model** | `conv_type`, `hidden_dims`, `bottle_neck_neurons` | Neural network architecture |
| **Training** | `n_epochs`, `learning_rate`, `gradient_clipping` | Optimization parameters |
| **Distributed** | `accelerator`, `devices`, `strategy` | Multi-GPU configuration |
| **Loss** | `lambda_*` parameters | Loss function weights |

See [Garfield_Model_Parameters.md](Garfield_Model_Parameters.md) for complete details.

---

## 🎯 Supported Data Types

Garfield supports multiple single-cell and spatial omics modalities:

| Data Type | Profile | Description |
|-----------|---------|-------------|
| **scRNA-seq** | `'RNA'` | Single-cell RNA sequencing |
| **scATAC-seq** | `'ATAC'` | Single-cell ATAC sequencing |
| **CITE-seq** | `'ADT'` | Antibody-derived tags (protein) |
| **Multi-modal** | `'multi-modal'` | Combined RNA+ATAC or RNA+ADT |
| **Spatial Transcriptomics** | `'spatial'` | Visium, Slide-seq, MERFISH, etc. |
| **Spatial Proteomics** | `'spatial'` | CODEX, IMC, etc. |

---

## 🔬 Example Workflows

### 1. Batch Integration (Single-Cell RNA-seq)

```python
import scanpy as sc
import Garfield as gf

# Load multiple batches
adata1 = sc.read_h5ad('batch1.h5ad')
adata2 = sc.read_h5ad('batch2.h5ad')
adata = adata1.concatenate(adata2, batch_key='batch')

# Configure for batch correction
config = {
    'adata_list': adata,
    'profile': 'RNA',
    'sample_col': 'batch',  # Column indicating batches
    'used_hvg': True,
    'n_epochs': 100,
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()

# Integrated embeddings in adata.obsm['garfield_latent']
```

### 2. Multi-Modal Integration (RNA + ATAC)

```python
config = {
    'adata_list': [adata_rna, adata_atac],
    'profile': 'multi-modal',
    'data_type': 'Paired',  # or 'UnPaired'
    'sub_data_type': ['rna', 'atac'],
    'weight': 0.6,  # Weight of RNA graph (1-weight for ATAC)
    'n_epochs': 100,
}

model = gf.Garfield(gf.settings.set_gf_params(config))
model.train()
```

### 3. Transfer Learning

```python
# Train on reference
config_ref = {
    'adata_list': adata_reference,
    'profile': 'RNA',
    'n_epochs': 100,
}
model = gf.Garfield(gf.settings.set_gf_params(config_ref))
model.train()
model.save('garfield_reference')

# Transfer to query
model_query = gf.Garfield.load('garfield_reference')
embeddings_query = model_query.predict(adata_query)
```

---

## 📊 Performance Benchmarks

### Training Speed (Slide-seq V2 Mouse Hippocampus, ~50K cells)

| Configuration | Training Time | Speedup |
|---------------|---------------|---------|
| Single GPU (A800) | 372s | 1.0x |
| 2 GPUs (DDP) | 196s | 1.9x |
| 4 GPUs (DDP) | 106s | 3.5x |
| 8 GPUs (DDP) | 62s | 6.0x |

### Scalability

Garfield has been tested on datasets ranging from:
- **Small**: ~1,000 cells
- **Medium**: ~50,000 cells
- **Large**: ~500,000 cells
- **Very Large**: >1,000,000 cells (with multi-GPU)

---

## 🔧 Advanced Configuration

### Graph Construction Methods

Choose the appropriate graph construction method for your data:

```python
config = {
    'graph_const_method': 'mu_std',  # Options:
    # 'mu_std': Statistical distance (recommended for spatial)
    # 'KNN': K-nearest neighbors (fast, general purpose)
    # 'Radius': Radius-based (for uniform spatial sampling)
    # 'Squidpy': Squidpy library (spatial transcriptomics)
    # 'Scanpy': Scanpy neighbors (non-spatial)
}
```

### Loss Function Tuning

Fine-tune loss weights for your specific task:

```python
config = {
    # Reconstruction losses
    'lambda_edge_recon': 500.0,  # Graph structure
    'lambda_gene_expr_recon': 300.0,  # Gene expression

    # Contrastive losses
    'lambda_latent_contrastive_instanceloss': 1.0,  # Cell-cell similarity
    'lambda_latent_contrastive_clusterloss': 0.5,  # Cluster separation

    # Batch correction
    'lambda_omics_recon_mmd_loss': 0.2,  # MMD for batch effect
}
```

### Hardware Optimization

```python
config = {
    # For single-GPU (fastest)
    'use_lightning': False,  # Use original optimized trainer

    # For multi-GPU
    'use_lightning': True,
    'devices': 8,
    'num_workers': 4,
    'persistent_workers': True,
    'precision': '16-mixed',  # 2x speedup on modern GPUs
    'accumulate_grad_batches': 1,  # Increase for large models
}
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: RuntimeError with multi-GPU training
```python
# Solution: Use correct strategy for dual-decoder architecture
config = {
    'strategy': 'ddp_find_unused_parameters_true',  # Required!
}
```

**Issue**: Out of memory
```python
# Solution: Reduce batch size or use gradient accumulation
config = {
    'edge_batch_size': 2048,  # Reduce from default 4096
    'node_batch_size': 64,  # Reduce from default 128
    'accumulate_grad_batches': 2,  # Accumulate gradients
}
```

**Issue**: Slow training on single GPU
```python
# Solution: Use original trainer instead of Lightning
config = {
    'use_lightning': False,  # 2-3x faster for single GPU
}
```

---

## 📚 Citation

If you use Garfield in your research, please cite our paper:

```bibtex
@article{zhou2025graph,
  title={Graph-based Contrastive Learning Enables Unified Integration and Niche Transfer Across Single-Cell and Spatial Multi-Omics},
  author={Zhou, Weige and Fan, Xueying and Li, Lanxiang and Zheng, Jianrong and Liu, Xiaodong and Jin, Wenfei and Tian, Luyi},
  journal={bioRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.02.19.638965}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Support

- **Documentation**: [garfield-bio.readthedocs.io](https://garfield-bio.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/zhou-1314/Garfield/issues)
- **Email**: zhouwg1314@gmail.com

---

## 🙏 Acknowledgments

Garfield uses and/or references the following excellent libraries and packages:

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [PyTorch Lightning](https://lightning.ai/) - Distributed training framework
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis toolkit
- [AnnData](https://anndata.readthedocs.io/) - Annotated data structures
- [NicheCompass](https://github.com/Lotfollahi-lab/nichecompass) - Spatial analysis methods
- [scArches](https://github.com/theislab/scarches) - Transfer learning framework
- [SIMBA](https://github.com/pinellolab/simba) - Multi-omics integration
- [MaxFuse](https://github.com/shuxiaoc/maxfuse) - Data fusion methods

Thanks to all their contributors and maintainers!

---

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Paper**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.19.638965v1)
- **Documentation**: [ReadTheDocs](https://garfield-bio.readthedocs.io/)
- **PyPI**: [pypi.org/project/garfield](https://pypi.org/project/garfield)
- **GitHub**: [github.com/zhou-1314/Garfield](https://github.com/zhou-1314/Garfield)

---

<p align="center">
<b>Developed with ❤️ by the Garfield Team</b>
</p>
