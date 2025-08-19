# Garfield Hyperparameter Optimization Framework

This directory contains a comprehensive hyperparameter optimization framework for the Garfield graph-based contrastive learning model. The framework supports multiple optimization strategies, parallel execution, and extensive customization options.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install optuna scikit-optimize
pip install psutil  # for resource monitoring

# Optional for advanced features
pip install ray  # for distributed optimization
pip install wandb  # for experiment tracking
```

### 2. Basic Usage

```python
import Garfield as gf
from garfield_hyperopt import run_quick_optimization

# Load your data
adata_list = [adata1, adata2, ...]  # Your AnnData objects

# Run quick optimization (20 trials)
best_params = run_quick_optimization(
    adata_list=adata_list,
    results_dir="./hyperopt_results",
    n_trials=20
)

# Use optimized parameters
gf_params = gf.settings.gf_params.copy()
gf_params.update(best_params)
gf_params['adata_list'] = adata_list

model = gf.Garfield(gf_params)
model.train_model()
```

## 📁 Framework Components

### Core Files

- **`hyperopt_config.py`** - Configuration classes and parameter spaces
- **`garfield_hyperopt.py`** - Main optimization engine with Optuna/random search
- **`parallel_hyperopt.py`** - Parallel and distributed optimization utilities  
- **`hyperopt_example.py`** - Complete examples and usage demonstrations

## 🎯 Optimization Strategies

### 1. Quick Optimization (Fast Iteration)
```python
from garfield_hyperopt import run_quick_optimization

best_params = run_quick_optimization(
    adata_list=adata_list,
    n_trials=20  # Fast, 5-10 minutes
)
```

**Use case**: Initial exploration, debugging, proof of concept

### 2. Comprehensive Optimization (Thorough Search)
```python  
from garfield_hyperopt import run_comprehensive_optimization

best_params = run_comprehensive_optimization(
    adata_list=adata_list,
    n_trials=200  # Thorough, 2-4 hours
)
```

**Use case**: Final model optimization, production deployment

### 3. Parallel Optimization (Scalable)
```python
from parallel_hyperopt import run_parallel_optimization

best_params = run_parallel_optimization(
    adata_list=adata_list,
    n_trials=100,
    n_parallel_jobs=8,  # Use 8 CPU cores/GPUs
    use_gpu=True
)
```

**Use case**: Large parameter spaces, multiple GPUs/CPUs available

### 4. Custom Optimization (Targeted Search)
```python
from hyperopt_config import HyperparameterSpace, OptimizationConfig
from garfield_hyperopt import GarfieldHyperOptimizer

# Define custom search space
custom_space = HyperparameterSpace(
    conv_type=["GAT", "GCN"],
    hidden_dims_sizes=[[64, 64], [128, 128], [256, 256]],
    learning_rate=(1e-4, 1e-3),
    # ... other parameters
)

# Custom configuration
config = OptimizationConfig(
    method="optuna",
    n_trials=100,
    objective=OptimizationObjective.MAXIMIZE_ARI
)

# Run optimization
optimizer = GarfieldHyperOptimizer(adata_list, config, custom_space)
best_params = optimizer.optimize()
```

## 🔧 Parameter Categories

### Model Architecture Parameters
- **`conv_type`**: Graph convolution type (`GAT`, `GATv2Conv`, `GCN`)
- **`gnn_layer`**: Number of GNN layers (1-4)
- **`hidden_dims`**: Hidden layer dimensions (`[64,64]`, `[128,128]`, etc.)
- **`bottle_neck_neurons`**: Latent space dimensionality (10-50)
- **`num_heads`**: Attention heads for GAT layers (1-8)
- **`dropout`**: Dropout rate (0.0-0.5)

### Training Parameters
- **`learning_rate`**: Adam learning rate (1e-5 to 1e-2)
- **`weight_decay`**: L2 regularization (1e-6 to 1e-3)
- **`n_epochs`**: Training epochs (50-300)
- **`batch_size`**: Batch size for training (64, 128, 256, 512)

### Loss Function Weights
- **`lambda_contrastive_instance`**: Instance-level contrastive loss weight
- **`lambda_contrastive_cluster`**: Cluster-level contrastive loss weight
- **`lambda_gene_expr_recon`**: Gene expression reconstruction loss weight
- **`lambda_edge_recon`**: Edge reconstruction loss weight

### Graph Construction Parameters
- **`n_neighbors`**: Number of neighbors for graph construction (5-30)
- **`graph_const_method`**: Graph construction method (`mu_std`, `KNN`, `Radius`)
- **`weight`**: Modality weighting factor (0.3-0.9)

### Data Preprocessing Parameters
- **`n_components`**: PCA components (20-100)
- **`rna_n_top_features`**: Number of top RNA features (1000, 2000, 3000, 4000)
- **`target_sum`**: Normalization target sum (1e4, 2e4, 5e4)

## 📊 Optimization Objectives

Choose from several optimization objectives:

```python
from hyperopt_config import OptimizationObjective

# Available objectives:
OptimizationObjective.MINIMIZE_LOSS              # Minimize training loss
OptimizationObjective.MAXIMIZE_ACCURACY          # Maximize classification accuracy
OptimizationObjective.MAXIMIZE_ARI              # Maximize Adjusted Rand Index
OptimizationObjective.MAXIMIZE_NMI              # Maximize Normalized Mutual Info
OptimizationObjective.MINIMIZE_RECONSTRUCTION_ERROR  # Minimize reconstruction error
```

## 🎛️ Predefined Search Spaces

The framework includes several predefined search spaces:

### Quick Search Space
- Limited parameter ranges
- Fast convergence  
- Good for initial exploration

### Comprehensive Search Space
- Full parameter ranges
- Thorough exploration
- Best for final optimization

### Architecture-Focused Search Space
- Focuses on model architecture parameters
- Fixed training hyperparameters
- Good for architecture ablation studies

### Training-Focused Search Space  
- Focuses on training hyperparameters
- Fixed architecture parameters
- Good for training optimization

```python
from hyperopt_config import get_default_search_spaces

search_spaces = get_default_search_spaces()
quick_space = search_spaces['quick']
comprehensive_space = search_spaces['comprehensive']
architecture_space = search_spaces['architecture']  
training_space = search_spaces['training']
```

## ⚡ Performance Optimization Tips

### 1. Memory Management
```python
config = OptimizationConfig(
    max_memory_gb=32,         # Adjust based on your system
    gpu_memory_fraction=0.8,  # Use 80% of GPU memory
)
```

### 2. Early Stopping
```python
config = OptimizationConfig(
    use_early_stopping=True,
    early_stopping_kwargs={
        'patience': 10,
        'min_delta': 0.001
    }
)
```

### 3. Cross-Validation
```python
config = OptimizationConfig(
    use_cv=True,
    cv_folds=5,  # 5-fold cross-validation
)
```

### 4. Parallel Execution
```python
# CPU parallelization
config = OptimizationConfig(n_jobs=8)

# GPU parallelization  
run_parallel_optimization(
    adata_list=adata_list,
    n_parallel_jobs=4,  # Number of GPUs
    use_gpu=True
)
```

## 📈 Results Analysis

### Accessing Results
```python
import json

# Load optimization results
with open('results_dir/optimization_results.json', 'r') as f:
    results = json.load(f)

best_params = results['best_params']
best_score = results['best_score']
history = results['optimization_history']
```

### Visualization
```python
from hyperopt_example import analyze_optimization_results

# Analyze and plot results
analyze_optimization_results("./results_dir")
```

## 🔄 Advanced Usage

### Custom Objective Function
```python
def custom_objective(model, data):
    """Custom objective function"""
    embeddings = model.get_latent_representation()
    # Compute your custom metric
    return custom_metric_score

# Use in optimizer
optimizer = GarfieldHyperOptimizer(
    adata_list=adata_list,
    config=config,
    search_space=search_space,
    custom_objective=custom_objective
)
```

### Parameter Constraints
```python
config = OptimizationConfig(
    param_constraints={
        'learning_rate': lambda lr: lr * 1000 < 1,  # Custom constraint
        'hidden_dims': lambda dims: sum(dims) < 512  # Architecture constraint
    }
)
```

### Distributed Optimization (Multi-Node)
```python
# Coming soon: Ray-based distributed optimization
from distributed_hyperopt import RayHyperOptimizer

optimizer = RayHyperOptimizer(
    adata_list=adata_list,
    config=config,
    search_space=search_space,
    n_nodes=4  # Use 4 compute nodes
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   config.max_memory_gb = 16  # Reduce memory limit
   config.gpu_memory_fraction = 0.6  # Reduce GPU memory usage
   ```

2. **Slow Optimization**
   ```python
   # Use smaller search space
   search_space = get_default_search_spaces()['quick']
   
   # Reduce trials
   config.n_trials = 50
   
   # Disable cross-validation for speed
   config.use_cv = False
   ```

3. **GPU Issues**
   ```python
   # Force CPU mode
   run_parallel_optimization(..., use_gpu=False)
   
   # Reduce GPU memory fraction
   config.gpu_memory_fraction = 0.5
   ```

### Debugging Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
config.verbose = True
```

## 📝 Example Workflows

### Workflow 1: Quick Start for New Dataset
```python
# 1. Quick exploration (5-10 minutes)
quick_params = run_quick_optimization(adata_list, n_trials=20)

# 2. Custom optimization based on quick results (30-60 minutes)  
custom_space = HyperparameterSpace(
    conv_type=[quick_params['conv_type']],  # Fix best conv type
    hidden_dims_sizes=[quick_params['hidden_dims']],  # Fix architecture
    # Optimize other parameters around best quick results
    learning_rate=(quick_params['learning_rate'] * 0.5, quick_params['learning_rate'] * 2),
    dropout=(max(0, quick_params['dropout'] - 0.1), min(0.5, quick_params['dropout'] + 0.1))
)

final_params = GarfieldHyperOptimizer(adata_list, config, custom_space).optimize()
```

### Workflow 2: Production Model Development  
```python
# 1. Architecture search
arch_params = run_optimization_with_space('architecture', n_trials=100)

# 2. Training optimization with best architecture
training_space = get_default_search_spaces()['training'] 
training_space.conv_type = [arch_params['conv_type']]
training_space.hidden_dims_sizes = [arch_params['hidden_dims']]

training_params = GarfieldHyperOptimizer(adata_list, config, training_space).optimize()

# 3. Final comprehensive optimization
final_params = run_comprehensive_optimization(adata_list, n_trials=200)
```

### Workflow 3: Multi-GPU Cluster Optimization
```python
# Parallel optimization across multiple GPUs
best_params = run_parallel_optimization(
    adata_list=adata_list,
    search_space_name="comprehensive",
    n_trials=500,
    n_parallel_jobs=8,  # 8 GPUs
    use_gpu=True,
    results_dir="./cluster_hyperopt"
)
```

## 🤝 Contributing

To contribute to the hyperparameter optimization framework:

1. Fork the repository
2. Create a feature branch
3. Add new optimization methods in `garfield_hyperopt.py`
4. Add new search spaces in `hyperopt_config.py`
5. Add examples in `hyperopt_example.py`
6. Submit a pull request

## 📚 References

- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://optuna.org/)
- [Garfield: Graph-based Contrastive Learning Paper](https://www.biorxiv.org/content/10.1101/2025.02.19.638965v1)
- [Hyperparameter Optimization Best Practices](https://distill.pub/2020/bayesian-optimization/)

---

For questions or issues, please create an issue in the repository or contact the maintainers.