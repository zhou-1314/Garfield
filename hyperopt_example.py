"""
Hyperparameter Optimization Example for Garfield

This script demonstrates how to use the hyperparameter optimization framework
with different optimization strategies and configurations.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# Import Garfield and hyperopt modules
import Garfield as gf
from hyperopt_config import (
    OptimizationConfig, HyperparameterSpace, OptimizationObjective,
    get_default_search_spaces, QUICK_CONFIG, COMPREHENSIVE_CONFIG
)
from garfield_hyperopt import (
    GarfieldHyperOptimizer, run_quick_optimization, run_comprehensive_optimization
)
from parallel_hyperopt import run_parallel_optimization

def prepare_example_data():
    """Prepare example data for hyperparameter optimization"""
    
    print("Loading example dataset...")
    
    # Option 1: Use built-in scanpy dataset
    adata = sc.datasets.pbmc3k()
    
    # Basic preprocessing
    adata.var_names_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Add some metadata for multi-batch scenario
    n_obs = adata.n_obs
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], n_obs)
    adata.obs['cell_type'] = np.random.choice(['TypeA', 'TypeB', 'TypeC', 'TypeD'], n_obs)
    
    # For this example, we'll use the same data as multiple "batches"
    adata_list = [adata.copy() for _ in range(3)]
    
    print(f"Prepared {len(adata_list)} datasets with {adata.n_obs} cells and {adata.n_vars} genes each")
    
    return adata_list

def example_1_quick_optimization():
    """Example 1: Quick hyperparameter optimization for fast iteration"""
    
    print("\n" + "="*60)
    print("Example 1: Quick Hyperparameter Optimization")
    print("="*60)
    
    # Prepare data
    adata_list = prepare_example_data()
    
    # Run quick optimization
    print("Starting quick optimization (20 trials)...")
    
    best_params = run_quick_optimization(
        adata_list=adata_list,
        results_dir="./results_quick_hyperopt",
        n_trials=20
    )
    
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params

def example_2_custom_search_space():
    """Example 2: Custom search space optimization"""
    
    print("\n" + "="*60)  
    print("Example 2: Custom Search Space Optimization")
    print("="*60)
    
    # Prepare data
    adata_list = prepare_example_data()
    
    # Define custom search space focused on architecture
    custom_search_space = HyperparameterSpace(
        conv_type=["GAT", "GCN"],
        gnn_layer=(1, 3),
        hidden_dims_sizes=[[64, 64], [128, 128], [256, 256], [128, 64]],
        bottle_neck_neurons=(10, 30),
        dropout=(0.1, 0.4),
        learning_rate=(1e-4, 1e-3),
        n_epochs=(50, 150),
        # Focus on these specific loss weights
        lambda_contrastive_instance=(0.5, 2.0),
        lambda_gene_expr_recon=(100.0, 500.0)
    )
    
    # Custom configuration
    custom_config = OptimizationConfig(
        method="optuna",
        n_trials=50,
        objective=OptimizationObjective.MAXIMIZE_ARI,
        cv_folds=3,
        results_dir="./results_custom_hyperopt"
    )
    
    # Run optimization
    optimizer = GarfieldHyperOptimizer(
        adata_list=adata_list,
        config=custom_config,
        search_space=custom_search_space
    )
    
    print("Starting custom optimization (50 trials with Optuna)...")
    best_params = optimizer.optimize()
    
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params

def example_3_parallel_optimization():
    """Example 3: Parallel hyperparameter optimization"""
    
    print("\n" + "="*60)
    print("Example 3: Parallel Hyperparameter Optimization") 
    print("="*60)
    
    # Prepare data
    adata_list = prepare_example_data()
    
    # Run parallel optimization
    print("Starting parallel optimization (100 trials with 4 workers)...")
    
    best_params = run_parallel_optimization(
        adata_list=adata_list,
        search_space_name="architecture",
        n_trials=100,
        n_parallel_jobs=4,
        results_dir="./results_parallel_hyperopt",
        use_gpu=True
    )
    
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params

def example_4_comprehensive_optimization():
    """Example 4: Comprehensive optimization with full search space"""
    
    print("\n" + "="*60)
    print("Example 4: Comprehensive Hyperparameter Optimization")
    print("="*60)
    
    # Prepare data  
    adata_list = prepare_example_data()
    
    # Run comprehensive optimization
    print("Starting comprehensive optimization (100 trials)...")
    print("This will take longer but explore the full parameter space...")
    
    best_params = run_comprehensive_optimization(
        adata_list=adata_list,
        results_dir="./results_comprehensive_hyperopt",
        n_trials=100
    )
    
    print(f"Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params

def example_5_train_with_optimized_parameters():
    """Example 5: Train Garfield model with optimized parameters"""
    
    print("\n" + "="*60)
    print("Example 5: Training with Optimized Parameters")
    print("="*60)
    
    # Get optimized parameters from a previous run
    # In practice, you would use the results from your optimization
    optimized_params = {
        'conv_type': 'GAT',
        'gnn_layer': 2,
        'hidden_dims': [128, 128],
        'bottle_neck_neurons': 20,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'n_epochs': 100,
        'lambda_latent_contrastive_instanceloss': 1.0,
        'lambda_gene_expr_recon': 300.0
    }
    
    # Prepare data
    adata_list = prepare_example_data()
    
    # Create full parameter set by combining with defaults
    gf_params = gf.settings.gf_params.copy()
    gf_params.update(optimized_params)
    gf_params['adata_list'] = adata_list
    
    # Train model
    print("Training Garfield model with optimized parameters...")
    model = gf.Garfield(gf_params)
    model.train_model()
    
    # Get results
    embeddings = model.get_latent_representation()
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return model, embeddings

def analyze_optimization_results(results_dir: str):
    """Analyze and visualize optimization results"""
    
    print(f"\n" + "="*60)
    print(f"Analyzing Results from {results_dir}")
    print("="*60)
    
    import json
    from pathlib import Path
    
    results_path = Path(results_dir) / "optimization_results.json"
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Optimization method: {results['config']['method']}")
    print(f"Number of trials: {len(results.get('optimization_history', []))}")
    print(f"Best score: {results['best_score']}")
    
    # Plot optimization history if available
    if 'optimization_history' in results:
        history = results['optimization_history']
        scores = [trial['score'] for trial in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(scores, 'b-', alpha=0.7, label='Trial scores')
        
        # Plot running best
        running_best = []
        best_so_far = scores[0]
        for score in scores:
            if score < best_so_far:  # Assuming minimization
                best_so_far = score
            running_best.append(best_so_far)
        
        plt.plot(running_best, 'r-', linewidth=2, label='Running best')
        plt.xlabel('Trial')
        plt.ylabel('Score')
        plt.title('Hyperparameter Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(Path(results_dir) / 'optimization_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Optimization plot saved to {results_dir}/optimization_progress.png")

def main():
    """Main function demonstrating different optimization approaches"""
    
    print("Garfield Hyperparameter Optimization Examples")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Example 1: Quick optimization for fast iteration
        best_params_quick = example_1_quick_optimization()
        
        # Example 2: Custom search space
        best_params_custom = example_2_custom_search_space()
        
        # Example 3: Parallel optimization (commented out by default)
        # best_params_parallel = example_3_parallel_optimization()
        
        # Example 4: Comprehensive optimization (commented out - takes longer)
        # best_params_comprehensive = example_4_comprehensive_optimization()
        
        # Example 5: Train with optimized parameters
        model, embeddings = example_5_train_with_optimized_parameters()
        
        # Analyze results
        analyze_optimization_results("./results_quick_hyperopt")
        analyze_optimization_results("./results_custom_hyperopt")
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("Check the results directories for detailed optimization results.")
        print("="*80)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have all required dependencies installed:")
        print("- pip install optuna")
        print("- pip install scikit-optimize") 
        print("- pip install scanpy")

if __name__ == "__main__":
    main()