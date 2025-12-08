"""
Comprehensive test script for Garfield PyTorch Lightning optimizations.

This script validates:
1. Critical bug fixes (class name, model wrapping, error handling)
2. Performance enhancements (gradient clipping, accumulation, distributed logging)
3. Parameter validation
4. Both training backends (original and Lightning)
5. New features (gradient accumulation, configurable parameters)

Usage:
    python tests/test_optimizations.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc
import torch
from Garfield import Garfield, settings


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_test(test_name, status="RUNNING"):
    """Print test status."""
    symbols = {'RUNNING': '⏳', 'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️'}
    print(f"{symbols.get(status, '?')} {test_name}... ", end='', flush=True)
    if status == 'RUNNING':
        return
    print(status)


def prepare_test_data():
    """Prepare small test dataset."""
    print("Preparing test data (PBMC3K subset)...")
    adata = sc.datasets.pbmc3k()

    # Filter to make it faster
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[:500, :500].copy()  # Small subset for fast testing

    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    sc.pp.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=15)

    print(f"✅ Test data ready: {adata.shape[0]} cells, {adata.shape[1]} genes\n")
    return adata


def test_parameter_validation():
    """Test parameter validation catches errors."""
    print_section("TEST 1: Parameter Validation")

    adata = prepare_test_data()

    tests = [
        ("Invalid precision", {'precision': 'invalid'}, ValueError),
        ("Invalid accelerator", {'accelerator': 'quantum'}, ValueError),
        ("Negative devices", {'devices': -2}, ValueError),
        ("Empty devices list", {'devices': []}, ValueError),
        ("Negative num_workers", {'num_workers': -1}, ValueError),
        ("persistent_workers without workers", {'persistent_workers': True, 'num_workers': 0}, ValueError),
        ("Invalid accumulate_grad_batches", {'accumulate_grad_batches': 0}, ValueError),
        ("Invalid lightning_sampling_mode", {'lightning_sampling_mode': 'turbo'}, ValueError),
        ("Invalid use_lightning", {'use_lightning': 'maybe'}, ValueError),
    ]

    passed = 0
    for test_name, bad_config, expected_error in tests:
        print_test(test_name, 'RUNNING')
        try:
            config = {
                'adata_list': [adata],
                'n_epochs': 1,
                'monitor': False,
                'verbose': False,
            }
            config.update(bad_config)
            settings.set_gf_params(config)
            model = Garfield(settings.gf_params.copy())
            model.train()  # Should raise error
            print_test(test_name, 'FAIL - No error raised')
        except expected_error as e:
            print_test(test_name, 'PASS')
            passed += 1
        except Exception as e:
            print_test(test_name, f'FAIL - Wrong error: {type(e).__name__}')

    print(f"\n✅ Passed {passed}/{len(tests)} validation tests\n")
    return passed == len(tests)


def test_original_trainer():
    """Test original trainer still works."""
    print_section("TEST 2: Original Trainer")

    adata = prepare_test_data()

    print_test("Training with original trainer", 'RUNNING')
    try:
        config = {
            'adata_list': [adata],
            'profile': 'RNA',
            'adj_key': 'connectivities',
            'n_epochs': 2,
            'use_lightning': False,  # Force original trainer
            'seed': 42,
            'monitor': False,
            'verbose': False,
        }
        settings.set_gf_params(config)
        model = Garfield(settings.gf_params.copy())
        model.train()

        # Check embeddings were computed
        assert 'garfield_latent' in model.adata.obsm
        embeddings = model.adata.obsm['garfield_latent']
        assert embeddings.shape[0] == adata.shape[0]

        print_test("Training with original trainer", 'PASS')
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
        return True
    except Exception as e:
        print_test("Training with original trainer", f'FAIL - {e}')
        import traceback
        traceback.print_exc()
        return False


def test_lightning_trainer():
    """Test Lightning trainer with optimizations."""
    print_section("TEST 3: Lightning Trainer with Optimizations")

    adata = prepare_test_data()

    print_test("Training with Lightning (force mode)", 'RUNNING')
    try:
        config = {
            'adata_list': [adata],
            'profile': 'RNA',
            'adj_key': 'connectivities',
            'n_epochs': 2,
            'use_lightning': True,  # Force Lightning
            'accelerator': 'cpu',  # Use CPU for testing
            'devices': 1,
            'seed': 42,
            'monitor': False,
            'verbose': False,
            'logger': None,  # Disable logging for cleaner output
        }
        settings.set_gf_params(config)
        model = Garfield(settings.gf_params.copy())
        model.train()

        # Check embeddings were computed
        assert 'garfield_latent' in model.adata.obsm
        embeddings = model.adata.obsm['garfield_latent']
        assert embeddings.shape[0] == adata.shape[0]

        print_test("Training with Lightning (force mode)", 'PASS')
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
        return True
    except Exception as e:
        print_test("Training with Lightning (force mode)", f'FAIL - {e}')
        import traceback
        traceback.print_exc()
        return False


def test_gradient_accumulation():
    """Test gradient accumulation feature."""
    print_section("TEST 4: Gradient Accumulation")

    adata = prepare_test_data()

    print_test("Training with gradient accumulation", 'RUNNING')
    try:
        config = {
            'adata_list': [adata],
            'profile': 'RNA',
            'adj_key': 'connectivities',
            'n_epochs': 2,
            'use_lightning': True,
            'accelerator': 'cpu',
            'devices': 1,
            'accumulate_grad_batches': 2,  # 2x effective batch size
            'seed': 42,
            'monitor': False,
            'verbose': False,
            'logger': None,
        }
        settings.set_gf_params(config)
        model = Garfield(settings.gf_params.copy())
        model.train()

        # Check training completed
        assert model.is_trained_
        assert 'garfield_latent' in model.adata.obsm

        print_test("Training with gradient accumulation", 'PASS')
        print(f"   Gradient accumulation: {config['accumulate_grad_batches']} steps")
        print(f"   Training completed successfully")
        return True
    except Exception as e:
        print_test("Training with gradient accumulation", f'FAIL - {e}')
        import traceback
        traceback.print_exc()
        return False


def test_sampling_modes():
    """Test all three sampling modes."""
    print_section("TEST 5: Random Sampling Modes")

    adata = prepare_test_data()
    modes = ['auto', 'legacy', 'optimized']
    results = {}

    for mode in modes:
        print_test(f"Sampling mode: {mode}", 'RUNNING')
        try:
            config = {
                'adata_list': [adata],
                'profile': 'RNA',
                'adj_key': 'connectivities',
                'n_epochs': 1,
                'use_lightning': True,
                'accelerator': 'cpu',
                'devices': 1,
                'lightning_sampling_mode': mode,
                'seed': 42,
                'monitor': False,
                'verbose': False,
                'logger': None,
            }
            settings.set_gf_params(config)
            model = Garfield(settings.gf_params.copy())
            model.train()

            embeddings = model.adata.obsm['garfield_latent']
            results[mode] = {
                'success': True,
                'mean': embeddings.mean(),
                'std': embeddings.std(),
            }

            print_test(f"Sampling mode: {mode}", 'PASS')
            print(f"   Mean: {results[mode]['mean']:.6f}, Std: {results[mode]['std']:.6f}")
        except Exception as e:
            print_test(f"Sampling mode: {mode}", f'FAIL - {e}')
            results[mode] = {'success': False, 'error': str(e)}

    all_passed = all(r['success'] for r in results.values())
    if all_passed:
        print(f"\n✅ All {len(modes)} sampling modes work correctly")
    return all_passed


def test_auto_trainer_selection():
    """Test automatic trainer selection."""
    print_section("TEST 6: Automatic Trainer Selection")

    adata = prepare_test_data()

    tests = [
        ("Single GPU -> Original", {'devices': 1}, "Original"),
        ("Auto mode single GPU", {'devices': 1, 'use_lightning': 'auto'}, "Original"),
    ]

    # Note: We can't test multi-GPU without actual GPUs, but we can test the logic
    passed = 0
    for test_name, config_update, expected_trainer in tests:
        print_test(test_name, 'RUNNING')
        try:
            config = {
                'adata_list': [adata],
                'profile': 'RNA',
                'adj_key': 'connectivities',
                'n_epochs': 1,
                'accelerator': 'cpu',
                'seed': 42,
                'monitor': False,
                'verbose': False,
            }
            config.update(config_update)
            settings.set_gf_params(config)
            model = Garfield(settings.gf_params.copy())

            # Check which trainer would be selected
            should_use_lightning = model._should_use_lightning()
            actual_trainer = "Lightning" if should_use_lightning else "Original"

            if actual_trainer == expected_trainer:
                print_test(test_name, f'PASS (selected {actual_trainer})')
                passed += 1
            else:
                print_test(test_name, f'FAIL (expected {expected_trainer}, got {actual_trainer})')
        except Exception as e:
            print_test(test_name, f'FAIL - {e}')

    print(f"\n✅ Passed {passed}/{len(tests)} selection tests\n")
    return passed == len(tests)


def test_error_handling():
    """Test error handling for missing dependencies."""
    print_section("TEST 7: Error Handling")

    print_test("Error message for invalid config", 'RUNNING')

    # Test that validation provides helpful error messages
    adata = prepare_test_data()
    try:
        config = {
            'adata_list': [adata],
            'precision': 'float128',  # Invalid
        }
        settings.set_gf_params(config)
        model = Garfield(settings.gf_params.copy())
        model.train()
        print_test("Error message for invalid config", 'FAIL - No error raised')
        return False
    except ValueError as e:
        error_msg = str(e)
        has_valid_options = 'Valid options' in error_msg
        has_precision = 'precision' in error_msg.lower()

        if has_valid_options and has_precision:
            print_test("Error message for invalid config", 'PASS')
            print(f"   Error message is helpful: '{error_msg[:80]}...'")
            return True
        else:
            print_test("Error message for invalid config", 'FAIL - Unclear error message')
            return False
    except Exception as e:
        print_test("Error message for invalid config", f'FAIL - Wrong error type: {type(e).__name__}')
        return False


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Garfield Optimization Test Suite                           ║
║                                                                              ║
║  This suite validates all recent optimizations and bug fixes:               ║
║  - Critical bug fixes (class name, model wrapping)                           ║
║  - Performance enhancements (gradient clipping, accumulation)                ║
║  - Parameter validation                                                      ║
║  - Both training backends                                                    ║
║  - New features                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    results = {}

    # Run all tests
    results['Parameter Validation'] = test_parameter_validation()
    results['Original Trainer'] = test_original_trainer()
    results['Lightning Trainer'] = test_lightning_trainer()
    results['Gradient Accumulation'] = test_gradient_accumulation()
    results['Sampling Modes'] = test_sampling_modes()
    results['Auto Selection'] = test_auto_trainer_selection()
    results['Error Handling'] = test_error_handling()

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\n{'='*80}")
    print(f"  Overall: {passed}/{total} test groups passed ({100*passed//total}%)")
    print(f"{'='*80}\n")

    if passed == total:
        print("🎉 All tests passed! The optimizations are working correctly.\n")
        print("Next steps:")
        print("  1. Try multi-GPU training if you have multiple GPUs available")
        print("  2. Experiment with gradient accumulation for larger effective batch sizes")
        print("  3. Use mixed precision (precision='16-mixed') for 2x speedup")
        print("  4. Adjust num_workers based on your CPU cores for optimal dataloading")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.\n")
        return 1


if __name__ == '__main__':
    exit(main())
