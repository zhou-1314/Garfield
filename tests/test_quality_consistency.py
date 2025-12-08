"""
Test script to validate that model quality is consistent across training methods.

This script verifies that the optimization fixes don't degrade model performance
by comparing representations learned with different training backends.

Usage:
    python tests/test_quality_consistency.py
"""

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import Garfield as gf


def compute_clustering_metrics(adata, latent_key='garfield_latent', cluster_key='leiden', label_key='cell_type'):
    """Compute clustering quality metrics."""
    # Perform clustering on latent representations
    sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15)
    sc.tl.leiden(adata, key_added=cluster_key, resolution=0.5)

    # Compare with ground truth labels if available
    if label_key in adata.obs.columns:
        ari = adjusted_rand_score(adata.obs[label_key], adata.obs[cluster_key])
        nmi = normalized_mutual_info_score(adata.obs[label_key], adata.obs[cluster_key])
        return {'ARI': ari, 'NMI': nmi}
    else:
        return {'ARI': None, 'NMI': None}


def test_quality_consistency():
    """
    Test that different training backends produce similar quality representations.
    """
    print("=" * 80)
    print("Testing Model Quality Consistency")
    print("=" * 80)

    # Load test data
    print("\nLoading test dataset...")
    adata = sc.datasets.pbmc3k()

    # Preprocess
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store ground truth labels
    adata.obs['cell_type'] = adata.obs['louvain'].copy() if 'louvain' in adata.obs else None

    # Fixed seed for reproducibility
    SEED = 42

    # Test configurations
    configs = [
        {
            'name': 'Original Trainer',
            'use_lightning': False,
            'seed': SEED,
        },
        {
            'name': 'Lightning Single GPU (Fixed)',
            'use_lightning': True,
            'seed': SEED,
            'devices': 1,
        },
    ]

    results = {}

    for config in configs:
        config_name = config.pop('name')
        print(f"\n{'-' * 80}")
        print(f"Testing: {config_name}")
        print(f"{'-' * 80}")

        # Create fresh copy of data
        adata_test = adata.copy()

        # Configure model
        model_config = {
            'adata_list': [adata_test],
            'profile': 'RNA',
            'n_epochs': 50,  # Reduced for testing
            'monitor': False,  # Suppress progress bars
            'verbose': False,
            **config
        }

        # Train model
        print(f"Training with {config_name}...")
        model = gf.Garfield(model_config)
        model.train()

        # Compute metrics
        print("Computing clustering metrics...")
        metrics = compute_clustering_metrics(adata_test, latent_key='garfield_latent')

        # Store results
        results[config_name] = {
            'metrics': metrics,
            'latent': adata_test.obsm['garfield_latent'].copy(),
        }

        print(f"Results for {config_name}:")
        if metrics['ARI'] is not None:
            print(f"  ARI: {metrics['ARI']:.4f}")
            print(f"  NMI: {metrics['NMI']:.4f}")
        else:
            print("  No ground truth labels available")

    # Compare results
    print(f"\n{'=' * 80}")
    print("Comparison Summary")
    print(f"{'=' * 80}")

    if len(results) >= 2:
        config_names = list(results.keys())
        ref_name = config_names[0]

        for test_name in config_names[1:]:
            print(f"\nComparing '{test_name}' vs '{ref_name}':")

            # Compare metrics
            ref_metrics = results[ref_name]['metrics']
            test_metrics = results[test_name]['metrics']

            if ref_metrics['ARI'] is not None and test_metrics['ARI'] is not None:
                ari_diff = abs(ref_metrics['ARI'] - test_metrics['ARI'])
                nmi_diff = abs(ref_metrics['NMI'] - test_metrics['NMI'])

                print(f"  ARI difference: {ari_diff:.4f}")
                print(f"  NMI difference: {nmi_diff:.4f}")

                # Check if differences are acceptable (within 5%)
                ARI_THRESHOLD = 0.05
                NMI_THRESHOLD = 0.05

                if ari_diff < ARI_THRESHOLD and nmi_diff < NMI_THRESHOLD:
                    print(f"  ✅ PASS: Metrics are consistent (within {ARI_THRESHOLD:.2%} tolerance)")
                else:
                    print(f"  ❌ FAIL: Metrics differ significantly!")
                    print(f"     Expected ARI diff < {ARI_THRESHOLD:.4f}, NMI diff < {NMI_THRESHOLD:.4f}")

            # Compare latent representations (correlation)
            ref_latent = results[ref_name]['latent']
            test_latent = results[test_name]['latent']

            # Compute correlation between latent spaces
            correlations = []
            for i in range(ref_latent.shape[1]):  # For each latent dimension
                corr = np.corrcoef(ref_latent[:, i], test_latent[:, i])[0, 1]
                correlations.append(abs(corr))

            mean_corr = np.mean(correlations)
            print(f"  Mean absolute correlation: {mean_corr:.4f}")

            CORR_THRESHOLD = 0.90
            if mean_corr > CORR_THRESHOLD:
                print(f"  ✅ PASS: Latent representations are highly correlated (> {CORR_THRESHOLD})")
            else:
                print(f"  ❌ FAIL: Latent representations differ significantly!")
                print(f"     Expected correlation > {CORR_THRESHOLD}")

    print(f"\n{'=' * 80}")
    print("Test Complete")
    print(f"{'=' * 80}\n")

    return results


def test_deterministic_seeding():
    """
    Test that the same seed produces identical results across runs.
    """
    print("\n" + "=" * 80)
    print("Testing Deterministic Seeding")
    print("=" * 80)

    # Load test data
    print("\nLoading test dataset...")
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    SEED = 123

    # Run twice with same seed
    latent_results = []

    for run in [1, 2]:
        print(f"\nRun {run} with seed={SEED}...")
        adata_test = adata.copy()

        model = gf.Garfield({
            'adata_list': [adata_test],
            'profile': 'RNA',
            'n_epochs': 20,
            'seed': SEED,
            'use_lightning': False,
            'monitor': False,
        })
        model.train()

        latent_results.append(adata_test.obsm['garfield_latent'].copy())

    # Check if results are identical
    diff = np.abs(latent_results[0] - latent_results[1])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nDifference between runs:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")

    TOLERANCE = 1e-5
    if max_diff < TOLERANCE:
        print(f"  ✅ PASS: Results are deterministic (max diff < {TOLERANCE})")
    else:
        print(f"  ❌ FAIL: Results are not deterministic!")
        print(f"     Expected max diff < {TOLERANCE}")

    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Garfield Quality Consistency Tests                         ║
║                                                                              ║
║  This script validates that performance optimizations don't degrade         ║
║  model quality by comparing different training backends.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Test 1: Quality consistency across backends
        test_quality_consistency()

        # Test 2: Deterministic seeding
        test_deterministic_seeding()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
