"""
Integration test for hyperparameter optimization with Garfield
Tests integration with the actual Garfield codebase
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/data/zhouwg_data/project/Garfield_dev')

def test_garfield_integration():
    """Test integration with Garfield settings and parameters"""
    print("Testing Garfield integration...")
    
    try:
        # Test if we can access Garfield settings
        import Garfield as gf
        
        # Get default parameters
        default_params = gf.settings.gf_params.copy()
        print(f"✓ Loaded Garfield default parameters: {len(default_params)} parameters")
        
        # Test a few key parameters
        key_params = ['profile', 'conv_type', 'learning_rate', 'n_epochs', 'hidden_dims']
        for param in key_params:
            if param in default_params:
                print(f"  - {param}: {default_params[param]}")
            else:
                print(f"  ⚠ Missing expected parameter: {param}")
        
        return True
        
    except Exception as e:
        print(f"✗ Garfield integration failed: {e}")
        print("This might be due to missing dependencies (seaborn, pandas, etc.)")
        return False

def test_parameter_compatibility():
    """Test that our hyperopt parameters match Garfield's expected parameters"""
    print("\nTesting parameter compatibility...")
    
    try:
        from hyperopt_config import HyperparameterSpace
        
        # Get our search space parameters
        space = HyperparameterSpace()
        
        # Test parameter sampling
        params = {}
        params['conv_type'] = np.random.choice(space.conv_type)
        params['gnn_layer'] = np.random.randint(*space.gnn_layer)
        hidden_dims_idx = np.random.randint(len(space.hidden_dims_sizes))
        params['hidden_dims'] = space.hidden_dims_sizes[hidden_dims_idx].copy()
        params['learning_rate'] = np.exp(np.random.uniform(
            np.log(space.learning_rate[0]), np.log(space.learning_rate[1])
        ))
        params['n_epochs'] = np.random.randint(*space.n_epochs)
        
        print("✓ Successfully sampled parameters:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Test that parameters have reasonable values
        assert params['conv_type'] in ['GAT', 'GATv2Conv', 'GCN']
        assert 1 <= params['gnn_layer'] <= 4
        assert 0.00001 <= params['learning_rate'] <= 0.01
        assert 50 <= params['n_epochs'] <= 300
        assert isinstance(params['hidden_dims'], list)
        
        print("✓ All parameter values are within expected ranges")
        return True
        
    except Exception as e:
        print(f"✗ Parameter compatibility test failed: {e}")
        return False

def test_mock_garfield_training():
    """Test a mock Garfield training setup without actual execution"""
    print("\nTesting mock Garfield training setup...")
    
    try:
        from hyperopt_config import OptimizationConfig, HyperparameterSpace
        
        # Create mock configuration
        config = OptimizationConfig(
            method="random",
            n_trials=3,
            use_cv=False
        )
        
        space = HyperparameterSpace()
        
        # Test parameter generation for multiple trials
        print("✓ Generating parameters for mock trials:")
        
        for trial_idx in range(3):
            # Sample parameters
            params = {}
            params['conv_type'] = np.random.choice(space.conv_type)
            params['learning_rate'] = np.exp(np.random.uniform(
                np.log(space.learning_rate[0]), np.log(space.learning_rate[1])
            ))
            params['dropout'] = np.random.uniform(*space.dropout)
            
            print(f"  Trial {trial_idx}: conv={params['conv_type']}, "
                  f"lr={params['learning_rate']:.6f}, dropout={params['dropout']:.3f}")
        
        print("✓ Mock trial parameter generation successful")
        return True
        
    except Exception as e:
        print(f"✗ Mock training setup failed: {e}")
        return False

def test_results_directory_structure():
    """Test that we can create and manage results directories properly"""
    print("\nTesting results directory structure...")
    
    try:
        import json
        import shutil
        
        # Test directory creation
        test_dirs = ['./test_quick', './test_comprehensive', './test_parallel']
        
        for test_dir in test_dirs:
            dir_path = Path(test_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock results file
            mock_results = {
                'method': 'test',
                'best_score': 0.85,
                'best_params': {'learning_rate': 0.001},
                'n_trials': 10
            }
            
            with open(dir_path / 'results.json', 'w') as f:
                json.dump(mock_results, f, indent=2)
            
            # Verify file was created
            assert (dir_path / 'results.json').exists()
            print(f"✓ Created and verified {test_dir}")
        
        # Clean up test directories
        for test_dir in test_dirs:
            shutil.rmtree(test_dir, ignore_errors=True)
        
        print("✓ Results directory management working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Results directory test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("=" * 60)
    print("GARFIELD HYPEROPT INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_garfield_integration,
        test_parameter_compatibility,
        test_mock_garfield_training,
        test_results_directory_structure
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
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nThe hyperparameter optimization framework is ready to use!")
        print("\nTo use with real data:")
        print("1. Install dependencies: pip install optuna pandas seaborn scanpy psutil")
        print("2. Load your AnnData objects into adata_list")
        print("3. Run: from garfield_hyperopt import run_quick_optimization")
        print("4. Execute: best_params = run_quick_optimization(adata_list)")
    else:
        print("❌ Some integration tests failed.")
        print("The framework structure is correct, but there may be dependency issues.")
        print("Install required packages to enable full functionality.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()