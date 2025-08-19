"""
Hyperparameter optimization configuration for Garfield

This module defines the hyperparameter search space and optimization utilities
for the Garfield graph-based contrastive learning model.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from enum import Enum

class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter search"""
    MINIMIZE_LOSS = "minimize_loss"
    MAXIMIZE_ACCURACY = "maximize_accuracy" 
    MAXIMIZE_ARI = "maximize_ari"  # Adjusted Rand Index
    MAXIMIZE_NMI = "maximize_nmi"  # Normalized Mutual Information
    MINIMIZE_RECONSTRUCTION_ERROR = "minimize_reconstruction_error"

@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space for Garfield"""
    
    # Model Architecture Parameters
    conv_type: List[str] = field(default_factory=lambda: ["GAT", "GATv2Conv", "GCN"])
    gnn_layer: Tuple[int, int] = (1, 4)  # min, max layers
    hidden_dims_sizes: List[List[int]] = field(default_factory=lambda: [
        [64, 64], [128, 128], [256, 256], [128, 64], [256, 128]
    ])
    bottle_neck_neurons: Tuple[int, int] = (10, 50)
    num_heads: Tuple[int, int] = (1, 8)  # for GAT layers
    dropout: Tuple[float, float] = (0.0, 0.5)
    drop_feature_rate: Tuple[float, float] = (0.0, 0.5)
    drop_edge_rate: Tuple[float, float] = (0.0, 0.5)
    
    # Training Parameters  
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    weight_decay: Tuple[float, float] = (1e-6, 1e-3)
    batch_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    n_epochs: Tuple[int, int] = (50, 300)
    
    # Loss Function Weights
    lambda_contrastive_instance: Tuple[float, float] = (0.1, 5.0)
    lambda_contrastive_cluster: Tuple[float, float] = (0.1, 2.0) 
    lambda_gene_expr_recon: Tuple[float, float] = (50.0, 1000.0)
    lambda_edge_recon: Tuple[float, float] = (100.0, 1000.0)
    lambda_adj_recon: Tuple[float, float] = (0.5, 5.0)
    
    # Graph Construction
    n_neighbors: Tuple[int, int] = (5, 30)
    graph_const_methods: List[str] = field(default_factory=lambda: ["mu_std", "KNN", "Radius"])
    weight: Tuple[float, float] = (0.3, 0.9)  # modality weighting
    
    # Data Preprocessing
    n_components: Tuple[int, int] = (20, 100)  # PCA components
    rna_n_top_features: List[int] = field(default_factory=lambda: [1000, 2000, 3000, 4000])
    target_sum: List[float] = field(default_factory=lambda: [1e4, 2e4, 5e4])
    
    # Augmentation
    augment_types: List[str] = field(default_factory=lambda: ["dropout", "svd"])
    svd_q: Tuple[int, int] = (3, 15)

@dataclass 
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    
    # Optimization method
    method: str = "optuna"  # "optuna", "random", "grid", "bayesian"
    
    # Search parameters
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1  # parallel jobs
    
    # Objective
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_LOSS
    
    # Early stopping for individual trials
    trial_timeout: int = 3600  # 1 hour per trial
    min_epochs_before_stop: int = 20
    patience: int = 10
    
    # Cross-validation
    cv_folds: int = 3
    use_cv: bool = True
    
    # Resource allocation
    max_memory_gb: int = 16
    gpu_memory_fraction: float = 0.8
    
    # Logging and checkpoints
    log_every_n_trials: int = 5
    save_checkpoint_every: int = 10
    results_dir: str = "./hyperopt_results"
    
    # Fixed parameters (won't be optimized)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    
    # Parameter constraints
    param_constraints: Dict[str, Any] = field(default_factory=dict)

def get_default_search_spaces() -> Dict[str, HyperparameterSpace]:
    """Get predefined search spaces for different use cases"""
    
    # Quick search space (for fast iteration)
    quick_space = HyperparameterSpace(
        conv_type=["GAT", "GCN"],
        gnn_layer=(1, 2),
        hidden_dims_sizes=[[64, 64], [128, 128]],
        bottle_neck_neurons=(15, 25),
        dropout=(0.1, 0.3),
        learning_rate=(1e-4, 1e-3),
        n_epochs=(50, 100)
    )
    
    # Comprehensive search space (for thorough optimization)  
    comprehensive_space = HyperparameterSpace()
    
    # Architecture-focused search space
    architecture_space = HyperparameterSpace(
        # Focus on model architecture
        conv_type=["GAT", "GATv2Conv", "GCN"],
        gnn_layer=(1, 4),
        hidden_dims_sizes=[
            [32, 32], [64, 64], [128, 128], [256, 256],
            [64, 32], [128, 64], [256, 128], [512, 256]
        ],
        bottle_neck_neurons=(5, 50),
        num_heads=(1, 8),
        dropout=(0.0, 0.5),
        # Keep other params relatively fixed
        learning_rate=(5e-4, 2e-3),
        n_epochs=(80, 150)
    )
    
    # Training-focused search space
    training_space = HyperparameterSpace(
        # Keep architecture simple
        conv_type=["GAT"],
        gnn_layer=(2, 2),
        hidden_dims_sizes=[[128, 128]],
        bottle_neck_neurons=(20, 20),
        # Focus on training hyperparameters
        learning_rate=(1e-5, 1e-2),
        weight_decay=(1e-6, 1e-3),
        batch_sizes=[32, 64, 128, 256, 512],
        lambda_contrastive_instance=(0.1, 10.0),
        lambda_contrastive_cluster=(0.1, 5.0),
        lambda_gene_expr_recon=(10.0, 1000.0)
    )
    
    return {
        "quick": quick_space,
        "comprehensive": comprehensive_space, 
        "architecture": architecture_space,
        "training": training_space
    }

def validate_config(config: OptimizationConfig, space: HyperparameterSpace) -> bool:
    """Validate optimization configuration and search space"""
    
    # Check if method is supported
    supported_methods = ["optuna", "random", "grid", "bayesian"]
    if config.method not in supported_methods:
        raise ValueError(f"Method {config.method} not supported. Use one of {supported_methods}")
    
    # Check resource constraints
    if config.max_memory_gb < 4:
        print("Warning: Low memory limit may cause issues with large datasets")
    
    if config.n_jobs > 1 and config.method == "optuna":
        print("Warning: Parallel execution with Optuna may require SQLite storage")
    
    # Validate search space ranges
    if space.learning_rate[0] >= space.learning_rate[1]:
        raise ValueError("Invalid learning rate range")
        
    if space.dropout[0] >= space.dropout[1]:
        raise ValueError("Invalid dropout range")
    
    return True

# Example usage configurations
QUICK_CONFIG = OptimizationConfig(
    method="optuna",
    n_trials=20,
    n_jobs=1,
    objective=OptimizationObjective.MINIMIZE_LOSS,
    cv_folds=2,
    trial_timeout=1800  # 30 minutes
)

COMPREHENSIVE_CONFIG = OptimizationConfig(
    method="optuna", 
    n_trials=200,
    n_jobs=2,
    objective=OptimizationObjective.MAXIMIZE_ARI,
    cv_folds=5,
    trial_timeout=7200  # 2 hours
)