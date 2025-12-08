"""
Simple direct comparison test for different training methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scanpy as sc
import time
from Garfield import Garfield, settings
from sklearn.metrics import adjusted_rand_score


def test_config(name, use_lightning_value):
    """Test a single configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    # Load fresh data
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Configure - merge with defaults first
    user_config = {
        'adata_list': [adata],
        'profile': 'RNA',
        'n_epochs': 10,  # Quick test
        'seed': 42,
        'use_lightning': use_lightning_value,
        'monitor': False,
        'verbose': False,
    }

    # Merge with defaults using settings
    settings.set_gf_params(user_config)
    full_config = settings.gf_params.copy()

    # Train
    print(f"Training with use_lightning={use_lightning_value}...")
    start = time.time()
    model = Garfield(full_config)
    model.train()
    elapsed = time.time() - start

    # Get embeddings
    embeddings = adata.obsm['garfield_latent'].copy()

    print(f"Training time: {elapsed:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings mean: {embeddings.mean():.6f}")
    print(f"Embeddings std: {embeddings.std():.6f}")

    return {
        'name': name,
        'time': elapsed,
        'embeddings': embeddings,
        'adata': adata,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Simple Configuration Comparison                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Test 1: Original Trainer
    result1 = test_config("Original Trainer", use_lightning_value=False)

    # Test 2: Optimized Lightning
    result2 = test_config("Optimized Lightning", use_lightning_value=True)

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    emb1 = result1['embeddings']
    emb2 = result2['embeddings']

    # Speed
    print(f"\nSpeed:")
    print(f"  Original Trainer: {result1['time']:.1f}s")
    print(f"  Optimized Lightning: {result2['time']:.1f}s")
    print(f"  Speedup: {result1['time']/result2['time']:.2f}x")

    # Embedding differences
    diff = np.abs(emb1 - emb2)
    print(f"\nEmbedding Differences:")
    print(f"  Max difference: {np.max(diff):.6e}")
    print(f"  Mean difference: {np.mean(diff):.6e}")

    # Correlation
    correlations = []
    for i in range(emb1.shape[1]):
        corr = np.corrcoef(emb1[:, i], emb2[:, i])[0, 1]
        correlations.append(abs(corr))
    mean_corr = np.mean(correlations)
    print(f"  Mean correlation: {mean_corr:.4f}")

    # Exact match
    exact = np.allclose(emb1, emb2, atol=1e-6)
    print(f"  Exact match: {exact}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    if exact:
        print("✅ Results are IDENTICAL")
        print("   Both trainers produce exactly the same embeddings.")
    elif mean_corr > 0.95:
        print("✅ Results are HIGHLY SIMILAR")
        print(f"   Correlation: {mean_corr:.4f} (>0.95)")
        print("   Small differences likely due to:")
        print("   - Different training loop implementations")
        print("   - Callback/logger overhead")
    elif mean_corr > 0.80:
        print("⚠️  Results are MODERATELY SIMILAR")
        print(f"   Correlation: {mean_corr:.4f} (0.80-0.95)")
        print("   Differences may be due to:")
        print("   - Different random sampling behavior")
        print("   - Different optimization trajectories")
    else:
        print("❌ Results are DIFFERENT")
        print(f"   Correlation: {mean_corr:.4f} (<0.80)")
        print("   Significant differences indicate:")
        print("   - Generator usage difference")
        print("   - Different random seeds")
        print("   - Bug in one implementation")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")

    if not exact and mean_corr < 0.95:
        print("""
The optimized Lightning version produces different results.

To test if this is due to the generator fix:
1. Temporarily revert the generator optimization in:
   Garfield/data/lightning_datamodule.py (lines 183-196)

2. Change from:
   if is_distributed and self.seed is not None:

   Back to:
   if self.seed is not None:

3. Re-run this test to see if Lightning now matches

This will tell us if the generator change is causing the difference.
        """)
    else:
        print("✅ Results are consistent. Optimization successful!")


if __name__ == '__main__':
    main()
