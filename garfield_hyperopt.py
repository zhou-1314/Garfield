"""
Hyperparameter Optimization Engine for Garfield

This module provides a comprehensive framework for optimizing Garfield hyperparameters
using various optimization strategies including Optuna, random search, and Bayesian optimization.
"""

import os
import json
import time
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Garfield imports
import Garfield as gf
from hyperopt_config import (
    HyperparameterSpace, OptimizationConfig, OptimizationObjective,
    get_default_search_spaces, validate_config
)

# Optional imports for different optimization backends
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Install with: pip install scikit-optimize")

class GarfieldHyperOptimizer:
    """Main hyperparameter optimization class for Garfield"""
    
    def __init__(self, 
                 adata_list: List,
                 config: OptimizationConfig,
                 search_space: HyperparameterSpace,
                 base_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Garfield hyperparameter optimizer
        
        Parameters
        ---------- 
        adata_list : List
            List of AnnData objects for training
        config : OptimizationConfig
            Optimization configuration
        search_space : HyperparameterSpace
            Hyperparameter search space definition
        base_params : Dict, optional
            Base parameters that won't be optimized
        """
        
        self.adata_list = adata_list
        self.config = config
        self.search_space = search_space
        self.base_params = base_params or {}
        
        # Validate configuration
        validate_config(config, search_space)
        
        # Setup results directory first
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Then setup logging
        self.setup_logging()
        
        # Initialize optimization backend
        self.backend = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        # Setup cross-validation
        self.cv_splitter = self._setup_cv_splitter()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.results_dir}/optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_cv_splitter(self):
        """Setup cross-validation splitter"""
        if self.config.use_cv:
            return KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        return None
    
    def _sample_hyperparameters(self, trial=None) -> Dict[str, Any]:
        """Sample hyperparameters from the search space"""
        
        if trial is not None:  # Optuna trial
            params = {}
            
            # Model architecture
            params['conv_type'] = trial.suggest_categorical('conv_type', self.search_space.conv_type)
            params['gnn_layer'] = trial.suggest_int('gnn_layer', *self.search_space.gnn_layer)
            params['hidden_dims'] = trial.suggest_categorical('hidden_dims', 
                                                             [str(dims) for dims in self.search_space.hidden_dims_sizes])
            params['hidden_dims'] = eval(params['hidden_dims'])  # Convert back to list
            
            params['bottle_neck_neurons'] = trial.suggest_int('bottle_neck_neurons', 
                                                            *self.search_space.bottle_neck_neurons)
            
            if params['conv_type'] in ['GAT', 'GATv2Conv']:
                params['num_heads'] = trial.suggest_int('num_heads', *self.search_space.num_heads)
            else:
                params['num_heads'] = 1
                
            params['dropout'] = trial.suggest_float('dropout', *self.search_space.dropout)
            params['drop_feature_rate'] = trial.suggest_float('drop_feature_rate', 
                                                            *self.search_space.drop_feature_rate)
            params['drop_edge_rate'] = trial.suggest_float('drop_edge_rate', 
                                                         *self.search_space.drop_edge_rate)
            
            # Training parameters
            params['learning_rate'] = trial.suggest_float('learning_rate', *self.search_space.learning_rate, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', *self.search_space.weight_decay, log=True)
            params['edge_batch_size'] = trial.suggest_categorical('edge_batch_size', self.search_space.batch_sizes)
            params['n_epochs'] = trial.suggest_int('n_epochs', *self.search_space.n_epochs)
            
            # Loss weights
            params['lambda_latent_contrastive_instanceloss'] = trial.suggest_float(
                'lambda_contrastive_instance', *self.search_space.lambda_contrastive_instance)
            params['lambda_latent_contrastive_clusterloss'] = trial.suggest_float(
                'lambda_contrastive_cluster', *self.search_space.lambda_contrastive_cluster)
            params['lambda_gene_expr_recon'] = trial.suggest_float(
                'lambda_gene_expr_recon', *self.search_space.lambda_gene_expr_recon)
            params['lambda_edge_recon'] = trial.suggest_float(
                'lambda_edge_recon', *self.search_space.lambda_edge_recon)
            params['lambda_latent_adj_recon_loss'] = trial.suggest_float(
                'lambda_adj_recon', *self.search_space.lambda_adj_recon)
            
            # Graph construction
            params['n_neighbors'] = trial.suggest_int('n_neighbors', *self.search_space.n_neighbors)
            params['graph_const_method'] = trial.suggest_categorical('graph_const_method', 
                                                                   self.search_space.graph_const_methods)
            params['weight'] = trial.suggest_float('weight', *self.search_space.weight)
            
            # Preprocessing
            params['n_components'] = trial.suggest_int('n_components', *self.search_space.n_components)
            params['rna_n_top_features'] = trial.suggest_categorical('rna_n_top_features', 
                                                                   self.search_space.rna_n_top_features)
            params['target_sum'] = trial.suggest_categorical('target_sum', self.search_space.target_sum)
            
            # Augmentation
            params['augment_type'] = trial.suggest_categorical('augment_type', self.search_space.augment_types)
            if params['augment_type'] == 'svd':
                params['svd_q'] = trial.suggest_int('svd_q', *self.search_space.svd_q)
                
        else:  # Random sampling
            params = {}
            params['conv_type'] = np.random.choice(self.search_space.conv_type)
            params['gnn_layer'] = np.random.randint(*self.search_space.gnn_layer)
            hidden_dims_idx = np.random.randint(len(self.search_space.hidden_dims_sizes))
            params['hidden_dims'] = self.search_space.hidden_dims_sizes[hidden_dims_idx].copy()
            params['bottle_neck_neurons'] = np.random.randint(*self.search_space.bottle_neck_neurons)
            
            if params['conv_type'] in ['GAT', 'GATv2Conv']:
                params['num_heads'] = np.random.randint(*self.search_space.num_heads)
            else:
                params['num_heads'] = 1
                
            params['dropout'] = np.random.uniform(*self.search_space.dropout)
            params['drop_feature_rate'] = np.random.uniform(*self.search_space.drop_feature_rate)
            params['drop_edge_rate'] = np.random.uniform(*self.search_space.drop_edge_rate)
            
            params['learning_rate'] = np.exp(np.random.uniform(
                np.log(self.search_space.learning_rate[0]),
                np.log(self.search_space.learning_rate[1])
            ))
            params['weight_decay'] = np.exp(np.random.uniform(
                np.log(self.search_space.weight_decay[0]),
                np.log(self.search_space.weight_decay[1])
            ))
            
            params['edge_batch_size'] = np.random.choice(self.search_space.batch_sizes)
            params['n_epochs'] = np.random.randint(*self.search_space.n_epochs)
            
            # Loss weights
            params['lambda_latent_contrastive_instanceloss'] = np.random.uniform(*self.search_space.lambda_contrastive_instance)
            params['lambda_latent_contrastive_clusterloss'] = np.random.uniform(*self.search_space.lambda_contrastive_cluster)
            params['lambda_gene_expr_recon'] = np.random.uniform(*self.search_space.lambda_gene_expr_recon)
            params['lambda_edge_recon'] = np.random.uniform(*self.search_space.lambda_edge_recon)
            params['lambda_latent_adj_recon_loss'] = np.random.uniform(*self.search_space.lambda_adj_recon)
            
            # Graph construction
            params['n_neighbors'] = np.random.randint(*self.search_space.n_neighbors)
            params['graph_const_method'] = np.random.choice(self.search_space.graph_const_methods)
            params['weight'] = np.random.uniform(*self.search_space.weight)
            
            # Preprocessing
            params['n_components'] = np.random.randint(*self.search_space.n_components)
            params['rna_n_top_features'] = np.random.choice(self.search_space.rna_n_top_features)
            params['target_sum'] = np.random.choice(self.search_space.target_sum)
            
            # Augmentation
            params['augment_type'] = np.random.choice(self.search_space.augment_types)
            if params['augment_type'] == 'svd':
                params['svd_q'] = np.random.randint(*self.search_space.svd_q)
            
        return params
    
    def _evaluate_parameters(self, params: Dict[str, Any], trial=None) -> float:
        """Evaluate a set of hyperparameters"""
        
        try:
            # Combine sampled params with base params
            full_params = {**self.base_params, **params}
            full_params['adata_list'] = self.adata_list
            
            if self.config.use_cv:
                scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(range(len(self.adata_list)))):
                    # Create train/val data splits
                    train_data = [self.adata_list[i] for i in train_idx]
                    val_data = [self.adata_list[i] for i in val_idx]
                    
                    fold_params = full_params.copy()
                    fold_params['adata_list'] = train_data
                    
                    # Train model
                    model = gf.Garfield(fold_params)
                    model.train_model()
                    
                    # Evaluate on validation data
                    score = self._compute_score(model, val_data)
                    scores.append(score)
                    
                    # Report intermediate value for pruning
                    if trial is not None:
                        trial.report(np.mean(scores), fold_idx)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                
                final_score = np.mean(scores)
                
            else:
                # Single training run
                model = gf.Garfield(full_params)
                model.train_model()
                score = self._compute_score(model, self.adata_list)
                final_score = score
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return worst possible score
            if self.config.objective == OptimizationObjective.MINIMIZE_LOSS:
                return float('inf')
            else:
                return float('-inf')
    
    def _compute_score(self, model, data) -> float:
        """Compute optimization score based on objective"""
        
        if self.config.objective == OptimizationObjective.MINIMIZE_LOSS:
            # Use final training loss
            return model.trainer.current_loss
            
        elif self.config.objective == OptimizationObjective.MAXIMIZE_ARI:
            # Compute clustering ARI
            embeddings = model.get_latent_representation()
            # Assumes ground truth labels available
            true_labels = data[0].obs['cell_type']  # Adjust based on your data
            pred_labels = self._cluster_embeddings(embeddings)
            return adjusted_rand_score(true_labels, pred_labels)
            
        elif self.config.objective == OptimizationObjective.MAXIMIZE_NMI:
            # Compute clustering NMI
            embeddings = model.get_latent_representation() 
            true_labels = data[0].obs['cell_type']
            pred_labels = self._cluster_embeddings(embeddings)
            return normalized_mutual_info_score(true_labels, pred_labels)
            
        else:
            raise ValueError(f"Unsupported objective: {self.config.objective}")
    
    def _cluster_embeddings(self, embeddings, n_clusters=None):
        """Cluster embeddings for evaluation"""
        from sklearn.cluster import KMeans
        
        if n_clusters is None:
            n_clusters = len(np.unique(self.adata_list[0].obs['cell_type']))
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
    
    def optimize_optuna(self) -> Dict[str, Any]:
        """Run optimization using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for this optimization method")
        
        # Create study
        direction = "minimize" if "minimize" in self.config.objective.value else "maximize"
        
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        def objective(trial):
            params = self._sample_hyperparameters(trial)
            return self._evaluate_parameters(params, trial)
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save results
        self._save_optimization_results(study)
        
        return study.best_params
    
    def optimize_random(self) -> Dict[str, Any]:
        """Run random search optimization"""
        
        best_params = None
        best_score = float('inf') if "minimize" in self.config.objective.value else float('-inf')
        
        for trial_idx in range(self.config.n_trials):
            params = self._sample_hyperparameters()
            score = self._evaluate_parameters(params)
            
            self.optimization_history.append({
                'trial': trial_idx,
                'params': params,
                'score': score
            })
            
            # Update best
            is_better = (score < best_score if "minimize" in self.config.objective.value 
                        else score > best_score)
            
            if is_better:
                best_score = score
                best_params = params.copy()
                
            if trial_idx % self.config.log_every_n_trials == 0:
                self.logger.info(f"Trial {trial_idx}: Score = {score:.4f}, Best = {best_score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization using specified method"""
        
        self.logger.info(f"Starting hyperparameter optimization with {self.config.method}")
        self.logger.info(f"Search space: {asdict(self.search_space)}")
        self.logger.info(f"Configuration: {asdict(self.config)}")
        
        start_time = time.time()
        
        if self.config.method == "optuna":
            best_params = self.optimize_optuna()
        elif self.config.method == "random":
            best_params = self.optimize_random()
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.method}")
        
        end_time = time.time()
        
        self.logger.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {self.best_score}")
        
        return best_params
    
    def _save_optimization_results(self, study=None):
        """Save optimization results to disk"""
        
        results = {
            'config': asdict(self.config),
            'search_space': asdict(self.search_space),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
        
        if study is not None:
            results['optuna_study'] = {
                'best_trial': study.best_trial._asdict() if hasattr(study.best_trial, '_asdict') else str(study.best_trial),
                'trials_dataframe': study.trials_dataframe().to_dict()
            }
        
        # Save to JSON
        with open(self.results_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.results_dir}")


def run_quick_optimization(adata_list: List, 
                          results_dir: str = "./quick_hyperopt",
                          n_trials: int = 20) -> Dict[str, Any]:
    """
    Quick hyperparameter optimization for fast iteration
    
    Parameters
    ----------
    adata_list : List
        List of AnnData objects
    results_dir : str
        Directory to save results
    n_trials : int
        Number of optimization trials
        
    Returns
    -------
    Dict containing best parameters
    """
    
    from hyperopt_config import QUICK_CONFIG, get_default_search_spaces
    
    config = QUICK_CONFIG
    config.results_dir = results_dir
    config.n_trials = n_trials
    
    search_spaces = get_default_search_spaces()
    
    optimizer = GarfieldHyperOptimizer(
        adata_list=adata_list,
        config=config,
        search_space=search_spaces['quick']
    )
    
    return optimizer.optimize()


def run_comprehensive_optimization(adata_list: List,
                                 results_dir: str = "./comprehensive_hyperopt", 
                                 n_trials: int = 100) -> Dict[str, Any]:
    """
    Comprehensive hyperparameter optimization for thorough search
    
    Parameters
    ----------
    adata_list : List
        List of AnnData objects  
    results_dir : str
        Directory to save results
    n_trials : int
        Number of optimization trials
        
    Returns
    -------
    Dict containing best parameters
    """
    
    from hyperopt_config import COMPREHENSIVE_CONFIG, get_default_search_spaces
    
    config = COMPREHENSIVE_CONFIG
    config.results_dir = results_dir
    config.n_trials = n_trials
    
    search_spaces = get_default_search_spaces()
    
    optimizer = GarfieldHyperOptimizer(
        adata_list=adata_list,
        config=config, 
        search_space=search_spaces['comprehensive']
    )
    
    return optimizer.optimize()


if __name__ == "__main__":
    # Example usage
    import scanpy as sc
    
    # Load example data (replace with your data)
    # adata = sc.datasets.pbmc3k()
    # adata_list = [adata]
    
    # Run quick optimization
    # best_params = run_quick_optimization(adata_list)
    # print(f"Best parameters: {best_params}")
    
    print("Hyperparameter optimization framework ready!")
    print("Usage:")
    print("1. Load your data into adata_list")
    print("2. Run run_quick_optimization() or run_comprehensive_optimization()")
    print("3. Use the returned best parameters for your final model")