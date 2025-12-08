"""
Simple script to compare different configurations side-by-side.

This allows you to test and compare:
1. Original Trainer
2. Original Lightning (need manual revert)
3. Optimized Lightning

Usage:
    python tests/compare_configurations.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scanpy as sc
import time
from Garfield import Garfield
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def train_and_evaluate(adata, config_name, config_params):
    """Train model and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")

    # Copy data
    adata_test = adata.copy()

    # Train
    start_time = time.time()
    model = Garfield(config_params)
    model.train()
    train_time = time.time() - start_time

    # Get embeddings
    embeddings = adata_test.obsm['garfield_latent'].copy()

    # Compute clustering if labels available
    sc.pp.neighbors(adata_test, use_rep='garfield_latent', n_neighbors=15)
    sc.tl.leiden(adata_test, key_added='clusters', resolution=0.5)

    # Metrics
    results = {
        'embeddings': embeddings,
        'train_time': train_time,
        'config_name': config_name,
    }

    # If ground truth exists
    if 'cell_type' in adata_test.obs.columns:
        ari = adjusted_rand_score(adata_test.obs['cell_type'], adata_test.obs['clusters'])
        nmi = normalized_mutual_info_score(adata_test.obs['cell_type'], adata_test.obs['clusters'])
        results['ari'] = ari
        results['nmi'] = nmi

    print(f"Training time: {train_time:.1f} seconds")
    if 'ari' in results:
        print(f"ARI: {results['ari']:.4f}")
        print(f"NMI: {results['nmi']:.4f}")

    return results


def compare_embeddings(results1, results2):
    """Compare two result sets."""
    emb1 = results1['embeddings']
    emb2 = results2['embeddings']
    name1 = results1['config_name']
    name2 = results2['config_name']

    print(f"\n{'='*80}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*80}")

    # Speed comparison
    time1 = results1['train_time']
    time2 = results2['train_time']
    speedup = time1 / time2
    print(f"\nSpeed Comparison:")
    print(f"  {name1}: {time1:.1f}s")
    print(f"  {name2}: {time2:.1f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # Quality comparison
    if 'ari' in results1 and 'ari' in results2:
        print(f"\nQuality Comparison:")
        print(f"  {name1} ARI: {results1['ari']:.4f}")
        print(f"  {name2} ARI: {results2['ari']:.4f}")
        print(f"  Difference: {abs(results1['ari'] - results2['ari']):.4f}")

    # Embedding similarity
    exact_match = np.allclose(emb1, emb2, atol=1e-6)
    if exact_match:
        print(f"\n✅ Embeddings are IDENTICAL")
    else:
        diff = np.abs(emb1 - emb2)
        print(f"\n❌ Embeddings DIFFER:")
        print(f"  Max difference: {np.max(diff):.6e}")
        print(f"  Mean difference: {np.mean(diff):.6e}")

        # Correlation
        correlations = []
        for i in range(emb1.shape[1]):
            corr = np.corrcoef(emb1[:, i], emb2[:, i])[0, 1]
            correlations.append(abs(corr))
        mean_corr = np.mean(correlations)
        print(f"  Mean correlation: {mean_corr:.4f}")

        if mean_corr > 0.95:
            print(f"  → Highly correlated (similar results)")
        elif mean_corr > 0.80:
            print(f"  → Moderately correlated")
        else:
            print(f"  → Low correlation (very different results)")


def main():
    """Run comparison."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Configuration Comparison Tool                             ║
║                                                                              ║
║  Compare different training configurations to understand performance         ║
║  and quality tradeoffs.                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Load data
    print("\nLoading dataset...")
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Add cell type if exists
    if 'louvain' in adata.obs:
        adata.obs['cell_type'] = adata.obs['louvain'].copy()

    SEED = 42
    N_EPOCHS = 30  # Reduced for faster testing

    # Base configuration with all required parameters
    base_config = {
        'adata_list': [adata.copy()],
        'profile': 'RNA',
        'n_epochs': N_EPOCHS,
        'seed': SEED,
        'monitor': False,
        'verbose': False,
    }

    # Configurations to test
    configs = [
        {
            'name': 'Original Trainer',
            'params': {
                **base_config,
                'adata_list': [adata.copy()],
                'use_lightning': False,
            }
        },
        {
            'name': 'Optimized Lightning',
            'params': {
                **base_config,
                'adata_list': [adata.copy()],
                'use_lightning': True,
                'devices': 1,
            }
        },
    ]

    # Note about original Lightning
    print(f"\n{'='*80}")
    print("NOTE: Original Lightning Configuration")
    print(f"{'='*80}")
    print("""
To test ORIGINAL Lightning (pre-optimization), you need to manually revert:

File: Garfield/data/lightning_datamodule.py
Lines: 183-196

Change from (current):
    is_distributed = (self.trainer and self.trainer.world_size > 1)
    if is_distributed and self.seed is not None:
        generator = torch.Generator()
        ...

Back to (original):
    if self.seed is not None:
        rank = 0
        if self.trainer is not None:
            rank = self.trainer.global_rank
        generator = torch.Generator()
        generator.manual_seed(self.seed + rank)

Then add this configuration to test:
    {
        'name': 'Original Lightning',
        'params': {
            'use_lightning': True,
            'devices': 1,
            ...
        }
    }
    """)

    # Run tests
    results = {}
    for config in configs:
        result = train_and_evaluate(
            adata,
            config['name'],
            config['params']
        )
        results[config['name']] = result

    # Comparisons
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if len(results) >= 2:
        config_names = list(results.keys())
        compare_embeddings(results[config_names[0]], results[config_names[1]])

    # Summary
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")

    print("""
What we learned:

1. SPEED COMPARISON:
   - Original Trainer vs Optimized Lightning
   - If similar → optimization successful!
   - If Lightning slower → still has overhead

2. QUALITY COMPARISON:
   - Check ARI/NMI scores
   - Check embedding correlation
   - If similar → optimization preserves quality
   - If different → need to investigate

3. EXACT MATCH:
   - Different implementations won't match exactly
   - But should be highly correlated (>0.95)
   - Quality metrics should be similar (within 5%)

TO TEST ORIGINAL LIGHTNING:
1. Revert generator optimization (see note above)
2. Add configuration to this script
3. Re-run to see all three versions

RECOMMENDATIONS:
- If Optimized Lightning ≈ Original Trainer → GOOD! Keep optimization
- If very different → Investigate which is better for your use case
- Check quality metrics on your actual data, not just PBMC
    """)


if __name__ == '__main__':
    main()
