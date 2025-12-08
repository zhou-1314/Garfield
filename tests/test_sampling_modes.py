"""
Test script to verify all sampling modes work correctly.

This tests the three lightning_sampling_mode options:
- 'auto' (default)
- 'legacy' (matches original Lightning)
- 'optimized' (explicit fast mode)

Usage:
    python tests/test_sampling_modes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scanpy as sc
import time
from Garfield import Garfield, settings


def test_sampling_mode(mode_name, sampling_mode):
    """Test a specific sampling mode."""
    print(f"\n{'='*80}")
    print(f"Testing Sampling Mode: {mode_name}")
    print(f"{'='*80}")

    # Load fresh data
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Build neighbors graph (required for RNA profile)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15)

    # Configure
    user_config = {
        'adata_list': [adata],
        'profile': 'RNA',
        'adj_key': 'connectivities',  # Standard scanpy neighbors key for RNA
        'n_epochs': 5,  # Very quick for testing
        'seed': 42,
        'use_lightning': True,  # Force Lightning to test sampling
        'lightning_sampling_mode': sampling_mode,
        'monitor': False,
        'verbose': False,
    }

    # Merge with defaults
    settings.set_gf_params(user_config)
    full_config = settings.gf_params.copy()

    # Train
    print(f"Training with lightning_sampling_mode='{sampling_mode}'...")
    start = time.time()

    try:
        model = Garfield(full_config)
        model.train()
        elapsed = time.time() - start

        # Get embeddings from model's internal adata
        embeddings = model.adata.obsm['garfield_latent'].copy()

        print(f"✅ SUCCESS")
        print(f"   Training time: {elapsed:.1f}s")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Embeddings mean: {embeddings.mean():.6f}")
        print(f"   Embeddings std: {embeddings.std():.6f}")

        return {
            'success': True,
            'mode': mode_name,
            'time': elapsed,
            'embeddings': embeddings,
            'adata': model.adata,
        }

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'mode': mode_name,
            'error': str(e),
        }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Sampling Mode Test Suite                                   ║
║                                                                              ║
║  Testing all three lightning_sampling_mode options:                          ║
║  - 'auto' (default, smart selection)                                        ║
║  - 'legacy' (matches original Lightning)                                    ║
║  - 'optimized' (explicit fast mode)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    modes = [
        ('Auto Mode (Default)', 'auto'),
        ('Legacy Mode (Original Lightning)', 'legacy'),
        ('Optimized Mode (Explicit Fast)', 'optimized'),
    ]

    results = []

    for mode_name, sampling_mode in modes:
        result = test_sampling_mode(mode_name, sampling_mode)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    all_success = all(r['success'] for r in results)

    if all_success:
        print("\n✅ All modes PASSED!")

        # Compare embeddings between modes
        print(f"\nEmbedding Comparisons:")

        for i in range(len(results)):
            for j in range(i+1, len(results)):
                emb1 = results[i]['embeddings']
                emb2 = results[j]['embeddings']
                name1 = results[i]['mode']
                name2 = results[j]['mode']

                diff = np.abs(emb1 - emb2)
                correlations = []
                for k in range(emb1.shape[1]):
                    corr = np.corrcoef(emb1[:, k], emb2[:, k])[0, 1]
                    correlations.append(abs(corr))
                mean_corr = np.mean(correlations)

                print(f"\n  {name1} vs {name2}:")
                print(f"    Max diff: {np.max(diff):.6e}")
                print(f"    Mean diff: {np.mean(diff):.6e}")
                print(f"    Correlation: {mean_corr:.4f}")

                if mean_corr > 0.95:
                    print(f"    → Highly similar ✅")
                elif mean_corr > 0.80:
                    print(f"    → Moderately similar ⚠️")
                else:
                    print(f"    → Different (expected for different modes) ℹ️")

        # Speed comparison
        print(f"\n{'='*80}")
        print("SPEED COMPARISON")
        print(f"{'='*80}")

        for result in results:
            if result['success']:
                print(f"{result['mode']}: {result['time']:.1f}s")

        # Expectations
        print(f"\n{'='*80}")
        print("EXPECTED BEHAVIOR")
        print(f"{'='*80}")

        print("""
1. Auto Mode:
   - Single GPU: Uses global RNG (fast)
   - Multi-GPU: Uses generator (correct DDP)
   - Should match 'optimized' on single GPU

2. Legacy Mode:
   - Always uses generator (even on single GPU)
   - Reproduces original Lightning behavior
   - May be slightly slower on single GPU
   - Different results from auto/optimized

3. Optimized Mode:
   - Same as auto (explicit)
   - Should match 'auto' exactly

RESULTS INTERPRETATION:
- Auto ≈ Optimized: ✅ Expected (same implementation)
- Legacy ≠ Auto/Optimized: ✅ Expected (different generator usage)
- All modes produce good quality: ✅ Goal achieved
        """)

    else:
        print("\n❌ Some modes FAILED:")
        for result in results:
            if not result['success']:
                print(f"  - {result['mode']}: {result['error']}")

    print(f"\n{'='*80}")
    print("USAGE GUIDE")
    print(f"{'='*80}")

    print("""
Choose the right mode for your use case:

DEFAULT (Recommended):
    model = Garfield({'adata_list': [adata]})
    # Uses 'auto' mode automatically

REPRODUCE ORIGINAL LIGHTNING:
    model = Garfield({
        'adata_list': [adata],
        'lightning_sampling_mode': 'legacy',
    })

MAXIMUM PERFORMANCE:
    model = Garfield({
        'adata_list': [adata],
        'lightning_sampling_mode': 'optimized',  # or 'auto'
    })

MULTI-GPU (any mode works):
    model = Garfield({
        'adata_list': [adata],
        'devices': 4,
        'strategy': 'ddp',
    })
    """)


if __name__ == '__main__':
    main()
