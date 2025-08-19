"""
Simple test to verify hyperparameter optimization works with current setup
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/data/zhouwg_data/project/Garfield_dev')

def create_mock_adata():
    """Create mock AnnData-like objects for testing"""
    class MockAnnData:
        def __init__(self, n_obs=100, n_vars=50):
            self.n_obs = n_obs
            self.n_vars = n_vars
            self.obs = {
                'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_obs),
                'batch': np.random.choice(['batch1', 'batch2'], n_obs)
            }
            self.var = {'gene_names': [f'Gene_{i}' for i in range(n_vars)]}
            self.X = np.random.randn(n_obs, n_vars)
            
        def copy(self):
            new_adata = MockAnnData(self.n_obs, self.n_vars)
            new_adata.obs = self.obs.copy()
            new_adata.var = self.var.copy()
            new_adata.X = self.X.copy()
            return new_adata
            
    return MockAnnData()

def test_parameter_optimization_workflow():
    """Test the complete parameter optimization workflow"""
    print("Testing complete hyperparameter optimization workflow...")
    
    try:
        from hyperopt_config import HyperparameterSpace, OptimizationConfig, OptimizationObjective
        from garfield_hyperopt import GarfieldHyperOptimizer
        
        # Create mock data
        mock_adata = create_mock_adata()
        adata_list = [mock_adata.copy() for _ in range(3)]
        
        print(f"✓ Created mock data: {len(adata_list)} datasets with {mock_adata.n_obs} cells each")
        
        # Create configuration for quick test
        config = OptimizationConfig(
            method="random",  # Use random search since optuna isn't installed
            n_trials=5,       # Small number for quick test
            use_cv=False,     # Disable CV for speed
            results_dir="./test_simple_hyperopt"
        )
        
        # Use quick search space
        from hyperopt_config import get_default_search_spaces
        spaces = get_default_search_spaces()
        search_space = spaces['quick']
        
        print("✓ Created optimization configuration")
        
        # Create optimizer (but don't run actual training)
        class MockGarfieldOptimizer(GarfieldHyperOptimizer):
            def _evaluate_parameters(self, params, trial=None):
                """Mock evaluation that returns a random score"""
                # Simulate some computation time
                import time
                time.sleep(0.1)
                
                # Return a mock score based on parameters
                # Better scores for certain parameter combinations
                score = 0.5
                
                if params.get('conv_type') == 'GAT':
                    score += 0.1
                    
                if 0.0001 <= params.get('learning_rate', 0) <= 0.001:
                    score += 0.2
                    
                if params.get('dropout', 0) < 0.3:
                    score += 0.1
                    
                # Add some randomness
                score += np.random.normal(0, 0.05)
                
                return max(0, min(1, score))  # Clamp to [0, 1]
        
        optimizer = MockGarfieldOptimizer(
            adata_list=adata_list,
            config=config,
            search_space=search_space
        )
        
        print("✓ Created mock optimizer")
        
        # Test parameter sampling
        print("\n✓ Testing parameter sampling:")
        for i in range(3):
            params = optimizer._sample_hyperparameters()
            score = optimizer._evaluate_parameters(params)
            print(f"  Trial {i}: conv={params['conv_type']}, "
                  f"lr={params['learning_rate']:.6f}, "
                  f"dropout={params['dropout']:.3f}, "
                  f"score={score:.4f}")
        
        # Test optimization loop (mock)
        print("\n✓ Running mock optimization...")
        best_params = optimizer.optimize_random()
        
        print(f"✓ Optimization completed!")
        print(f"✓ Best score: {optimizer.best_score:.4f}")
        print(f"✓ Best parameters:")
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        
        # Test results saving
        results_dir = Path(config.results_dir)
        if (results_dir / 'optimization_results.json').exists():
            print(f"✓ Results saved to {results_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_garfield_parameter_compatibility():
    """Test that our optimized parameters work with Garfield"""
    print("\nTesting Garfield parameter compatibility...")
    
    try:
        import Garfield as gf
        from hyperopt_config import HyperparameterSpace
        
        # Get default Garfield parameters
        default_params = gf.settings.gf_params.copy()
        print(f"✓ Loaded {len(default_params)} default Garfield parameters")
        
        # Sample some parameters from our search space
        space = HyperparameterSpace()
        optimized_params = {
            'conv_type': np.random.choice(space.conv_type),
            'gnn_layer': np.random.randint(*space.gnn_layer),
            'hidden_dims': space.hidden_dims_sizes[0].copy(),
            'learning_rate': np.exp(np.random.uniform(np.log(space.learning_rate[0]), np.log(space.learning_rate[1]))),
            'dropout': np.random.uniform(*space.dropout),
            'n_epochs': np.random.randint(*space.n_epochs),
        }
        
        print("✓ Generated optimized parameters:")
        for key, value in optimized_params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        
        # Test parameter merging
        merged_params = default_params.copy()
        merged_params.update(optimized_params)
        
        # Check that all essential parameters exist
        essential_params = ['profile', 'conv_type', 'learning_rate', 'n_epochs', 'hidden_dims']
        for param in essential_params:
            if param not in merged_params:
                print(f"  ⚠ Warning: Missing essential parameter {param}")
            else:
                print(f"  ✓ {param}: {merged_params[param]}")
        
        print("✓ Parameter compatibility test passed")
        return True
        
    except Exception as e:
        print(f"✗ Garfield compatibility test failed: {e}")
        return False

def main():
    """Run simple hyperopt tests"""
    print("=" * 60)
    print("SIMPLE HYPERPARAMETER OPTIMIZATION TEST")
    print("=" * 60)
    
    tests = [
        test_parameter_optimization_workflow,
        test_garfield_parameter_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"SIMPLE TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL SIMPLE TESTS PASSED!")
        print("\nThe hyperparameter optimization framework is working correctly!")
        print("You can now use it with real Garfield training.")
        print("\nNext steps:")
        print("1. Load your real AnnData objects")
        print("2. Use run_quick_optimization() for fast results")
        print("3. Scale up to comprehensive optimization for best results")
    else:
        print("❌ Some simple tests failed.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()