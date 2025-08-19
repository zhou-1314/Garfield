"""
Parallel Hyperparameter Optimization for Garfield

This module provides utilities for distributed and parallel hyperparameter optimization
including GPU management, memory optimization, and distributed training capabilities.
"""

import os
import time
import json
import pickle
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
import logging

import numpy as np
import torch
import psutil
from queue import Queue
import threading

from hyperopt_config import OptimizationConfig, HyperparameterSpace
from garfield_hyperopt import GarfieldHyperOptimizer

class ResourceManager:
    """Manages computational resources for parallel optimization"""
    
    def __init__(self, max_memory_gb: int = 16, gpu_memory_fraction: float = 0.8):
        self.max_memory_gb = max_memory_gb
        self.gpu_memory_fraction = gpu_memory_fraction
        self.available_gpus = self._detect_gpus()
        self.gpu_queue = Queue()
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        
        # Initialize GPU queue
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    
    def acquire_gpu(self, timeout: int = 300) -> Optional[int]:
        """Acquire a GPU for computation"""
        try:
            gpu_id = self.gpu_queue.get(timeout=timeout)
            self._set_gpu_memory_limit(gpu_id)
            return gpu_id
        except:
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU back to the pool"""
        if gpu_id is not None:
            torch.cuda.empty_cache()
            self.gpu_queue.put(gpu_id)
    
    def _set_gpu_memory_limit(self, gpu_id: int):
        """Set memory limit for GPU"""
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            memory_limit = int(total_memory * self.gpu_memory_fraction)
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, gpu_id)

class MemoryMonitor:
    """Monitors system memory usage"""
    
    def __init__(self, max_memory_gb: int):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.lock = threading.Lock()
        
    def check_memory_available(self) -> bool:
        """Check if enough memory is available"""
        with self.lock:
            available_memory = psutil.virtual_memory().available
            return available_memory > self.max_memory_bytes
    
    def wait_for_memory(self, timeout: int = 300):
        """Wait for sufficient memory to become available"""
        start_time = time.time()
        while not self.check_memory_available():
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout waiting for memory")
            time.sleep(5)

class DistributedHyperOptimizer:
    """Distributed hyperparameter optimization for Garfield"""
    
    def __init__(self,
                 adata_list: List,
                 config: OptimizationConfig,
                 search_space: HyperparameterSpace,
                 n_parallel_jobs: int = None,
                 use_gpu: bool = True):
        """
        Initialize distributed hyperparameter optimizer
        
        Parameters
        ----------
        adata_list : List
            List of AnnData objects
        config : OptimizationConfig  
            Optimization configuration
        search_space : HyperparameterSpace
            Parameter search space
        n_parallel_jobs : int, optional
            Number of parallel jobs (default: number of CPUs)
        use_gpu : bool
            Whether to use GPU acceleration
        """
        
        self.adata_list = adata_list
        self.config = config
        self.search_space = search_space
        self.use_gpu = use_gpu
        
        # Setup parallel execution
        self.n_parallel_jobs = n_parallel_jobs or min(mp.cpu_count(), config.n_jobs)
        self.resource_manager = ResourceManager(config.max_memory_gb, config.gpu_memory_fraction)
        
        # Results tracking
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup distributed logging"""
        log_format = '%(asctime)s - %(process)d - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.results_dir / 'distributed_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _worker_function(self, 
                        worker_id: int, 
                        param_sets: List[Dict[str, Any]], 
                        shared_results: Dict) -> List[Dict[str, Any]]:
        """Worker function for parallel execution"""
        
        worker_results = []
        gpu_id = None
        
        try:
            # Acquire resources
            if self.use_gpu and self.resource_manager.available_gpus:
                gpu_id = self.resource_manager.acquire_gpu()
                
            self.resource_manager.memory_monitor.wait_for_memory()
            
            self.logger.info(f"Worker {worker_id} starting with GPU {gpu_id}")
            
            for param_idx, params in enumerate(param_sets):
                try:
                    start_time = time.time()
                    
                    # Set device for this worker
                    if gpu_id is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                        params['device_id'] = gpu_id
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        params['device_id'] = -1  # CPU mode
                    
                    # Create optimizer for this parameter set
                    optimizer = GarfieldHyperOptimizer(
                        adata_list=self.adata_list,
                        config=self.config,
                        search_space=self.search_space,
                        base_params=params
                    )
                    
                    # Evaluate parameters
                    score = optimizer._evaluate_parameters(params)
                    
                    execution_time = time.time() - start_time
                    
                    result = {
                        'worker_id': worker_id,
                        'param_idx': param_idx,
                        'params': params,
                        'score': score,
                        'execution_time': execution_time,
                        'gpu_id': gpu_id
                    }
                    
                    worker_results.append(result)
                    
                    # Save checkpoint
                    self._save_worker_checkpoint(worker_id, result)
                    
                    self.logger.info(f"Worker {worker_id} completed param {param_idx}: "
                                   f"score={score:.4f}, time={execution_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"Worker {worker_id} error on param {param_idx}: {e}")
                    result = {
                        'worker_id': worker_id,
                        'param_idx': param_idx,
                        'params': params,
                        'score': float('inf'),
                        'execution_time': -1,
                        'error': str(e),
                        'gpu_id': gpu_id
                    }
                    worker_results.append(result)
        
        finally:
            # Release resources
            if gpu_id is not None:
                self.resource_manager.release_gpu(gpu_id)
                
        return worker_results
    
    def _save_worker_checkpoint(self, worker_id: int, result: Dict[str, Any]):
        """Save checkpoint for individual worker"""
        checkpoint_file = self.checkpoint_dir / f"worker_{worker_id}_checkpoint.json"
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        else:
            checkpoint_data = {'results': []}
        
        checkpoint_data['results'].append(result)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
    
    def _distribute_parameters(self, all_params: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Distribute parameters across workers"""
        n_params_per_worker = len(all_params) // self.n_parallel_jobs
        remainder = len(all_params) % self.n_parallel_jobs
        
        param_sets = []
        start_idx = 0
        
        for worker_id in range(self.n_parallel_jobs):
            n_params = n_params_per_worker + (1 if worker_id < remainder else 0)
            end_idx = start_idx + n_params
            param_sets.append(all_params[start_idx:end_idx])
            start_idx = end_idx
        
        return param_sets
    
    def optimize_parallel(self) -> Dict[str, Any]:
        """Run parallel hyperparameter optimization"""
        
        self.logger.info(f"Starting distributed optimization with {self.n_parallel_jobs} workers")
        self.logger.info(f"Available GPUs: {self.resource_manager.available_gpus}")
        
        start_time = time.time()
        
        # Generate all parameter combinations
        if self.config.method == "random":
            all_params = []
            for _ in range(self.config.n_trials):
                optimizer = GarfieldHyperOptimizer(
                    adata_list=self.adata_list,
                    config=self.config,
                    search_space=self.search_space
                )
                params = optimizer._sample_hyperparameters()
                all_params.append(params)
        else:
            raise ValueError(f"Parallel execution not yet implemented for {self.config.method}")
        
        # Distribute parameters across workers
        param_sets = self._distribute_parameters(all_params)
        
        # Execute in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_parallel_jobs) as executor:
            
            future_to_worker = {
                executor.submit(self._worker_function, worker_id, param_set, {}): worker_id
                for worker_id, param_set in enumerate(param_sets)
            }
            
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                    self.logger.info(f"Worker {worker_id} completed with {len(worker_results)} results")
                except Exception as e:
                    self.logger.error(f"Worker {worker_id} generated exception: {e}")
        
        # Find best result
        if self.config.objective.value.startswith("minimize"):
            best_result = min(all_results, key=lambda x: x['score'] if x['score'] != float('inf') else float('inf'))
        else:
            best_result = max(all_results, key=lambda x: x['score'] if x['score'] != float('-inf') else float('-inf'))
        
        end_time = time.time()
        
        # Save final results
        final_results = {
            'config': asdict(self.config),
            'search_space': asdict(self.search_space),
            'best_result': best_result,
            'all_results': all_results,
            'execution_time': end_time - start_time,
            'n_workers': self.n_parallel_jobs,
            'n_trials': len(all_results)
        }
        
        with open(self.results_dir / 'parallel_optimization_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"Distributed optimization completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Best score: {best_result['score']:.4f}")
        self.logger.info(f"Best parameters: {best_result['params']}")
        
        return best_result['params']

class AsyncHyperOptimizer:
    """Asynchronous hyperparameter optimization with early stopping"""
    
    def __init__(self,
                 adata_list: List,
                 config: OptimizationConfig,
                 search_space: HyperparameterSpace):
        
        self.adata_list = adata_list
        self.config = config
        self.search_space = search_space
        
        self.active_trials = {}
        self.completed_trials = []
        self.best_score = float('inf') if config.objective.value.startswith("minimize") else float('-inf')
        self.best_params = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def should_stop_early(self, current_score: float, trial_progress: float) -> bool:
        """Determine if a trial should be stopped early"""
        
        if trial_progress < 0.2:  # Don't stop too early
            return False
            
        if self.config.objective.value.startswith("minimize"):
            return current_score > self.best_score * 1.5  # Stop if 50% worse than best
        else:
            return current_score < self.best_score * 0.5  # Stop if 50% worse than best
    
    def optimize_async(self) -> Dict[str, Any]:
        """Run asynchronous optimization with early stopping"""
        
        # Implementation would use async/await patterns
        # This is a simplified version showing the concept
        pass

def run_parallel_optimization(adata_list: List,
                            search_space_name: str = "comprehensive",
                            n_trials: int = 100,
                            n_parallel_jobs: int = None,
                            results_dir: str = "./parallel_hyperopt",
                            use_gpu: bool = True) -> Dict[str, Any]:
    """
    Run parallel hyperparameter optimization
    
    Parameters
    ----------
    adata_list : List
        List of AnnData objects
    search_space_name : str
        Name of predefined search space ('quick', 'comprehensive', 'architecture', 'training')
    n_trials : int
        Total number of trials
    n_parallel_jobs : int, optional
        Number of parallel workers
    results_dir : str
        Directory to save results
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    Dict containing best parameters
    """
    
    from hyperopt_config import OptimizationConfig, get_default_search_spaces
    
    # Setup configuration
    config = OptimizationConfig(
        method="random",  # Only random search supported for parallel currently
        n_trials=n_trials,
        results_dir=results_dir,
        n_jobs=n_parallel_jobs or min(mp.cpu_count(), 8)
    )
    
    search_spaces = get_default_search_spaces()
    search_space = search_spaces[search_space_name]
    
    # Run distributed optimization
    optimizer = DistributedHyperOptimizer(
        adata_list=adata_list,
        config=config,
        search_space=search_space,
        n_parallel_jobs=n_parallel_jobs,
        use_gpu=use_gpu
    )
    
    return optimizer.optimize_parallel()


if __name__ == "__main__":
    # Example usage
    print("Parallel hyperparameter optimization framework ready!")
    print("Usage:")
    print("1. Load your data: adata_list = [adata1, adata2, ...]")
    print("2. Run: run_parallel_optimization(adata_list, n_trials=100, n_parallel_jobs=4)")
    print("3. Check results in the specified results directory")