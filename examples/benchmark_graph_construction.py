"""
Benchmark script to compare original vs optimized spatial graph construction.

Usage:
    python examples/benchmark_graph_construction.py
"""
import os
import sys
import time
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# Add Garfield to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import both versions
from Garfield.preprocessing.adj_construction import graph_construction as graph_construction_original
from Garfield.preprocessing.adj_construction_optimized import graph_construction_optimized


def create_synthetic_spatial_data(n_cells=10000, spatial_dim=2):
    """Create synthetic spatial transcriptomics data for benchmarking."""
    np.random.seed(42)

    # Create random gene expression
    adata = sc.AnnData(np.random.randn(n_cells, 2000))

    # Create spatial coordinates (simulate tissue layout)
    # Use a grid-like structure with some noise
    grid_size = int(np.sqrt(n_cells))
    x = np.repeat(np.arange(grid_size), grid_size)[:n_cells]
    y = np.tile(np.arange(grid_size), grid_size)[:n_cells]

    # Add noise
    x = x + np.random.randn(n_cells) * 0.3
    y = y + np.random.randn(n_cells) * 0.3

    adata.obsm['spatial'] = np.column_stack([x, y])

    return adata


def benchmark_single_run(adata, mode, k, method="original", **kwargs):
    """Run single benchmark and return time + result."""
    if method == "original":
        func = graph_construction_original
        start = time.time()
        adj = func(adata, mode=mode, k=k, batch_key=None, verbose=False)
        elapsed = time.time() - start
    elif method == "optimized":
        func = graph_construction_optimized
        start = time.time()
        adj = func(adata, mode=mode, k=k, batch_key=None, verbose=False, **kwargs)
        elapsed = time.time() - start
    else:
        raise ValueError(f"Unknown method: {method}")

    return elapsed, adj


def benchmark_comparison(n_cells_list=[1000, 5000, 10000, 20000, 50000],
                        mode="Radius",
                        k=150):
    """
    Compare original vs optimized implementation across different dataset sizes.
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: Spatial Graph Construction ({mode}, k={k})")
    print(f"{'='*80}\n")

    results = {
        'n_cells': [],
        'original_time': [],
        'optimized_time': [],
        'optimized_parallel_time': [],
        'speedup': [],
        'speedup_parallel': [],
    }

    for n_cells in n_cells_list:
        print(f"\nDataset size: {n_cells:,} cells")
        print("-" * 40)

        # Create synthetic data
        adata = create_synthetic_spatial_data(n_cells)

        # Benchmark original
        print("  Running original implementation...")
        try:
            original_time, adj_original = benchmark_single_run(
                adata, mode, k, method="original"
            )
            print(f"    Time: {original_time:.3f}s")
        except Exception as e:
            print(f"    Failed: {e}")
            original_time = np.nan
            adj_original = None

        # Benchmark optimized (single-threaded)
        print("  Running optimized implementation (1 core)...")
        try:
            optimized_time, adj_optimized = benchmark_single_run(
                adata, mode, k, method="optimized", n_jobs=1
            )
            print(f"    Time: {optimized_time:.3f}s")
            speedup = original_time / optimized_time if not np.isnan(original_time) else np.nan
            print(f"    Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Failed: {e}")
            optimized_time = np.nan
            speedup = np.nan
            adj_optimized = None

        # Benchmark optimized (multi-threaded) - only for larger datasets
        if n_cells >= 10000:
            print("  Running optimized implementation (4 cores)...")
            try:
                optimized_parallel_time, adj_optimized_par = benchmark_single_run(
                    adata, mode, k, method="optimized", n_jobs=4
                )
                print(f"    Time: {optimized_parallel_time:.3f}s")
                speedup_parallel = original_time / optimized_parallel_time if not np.isnan(original_time) else np.nan
                print(f"    Speedup: {speedup_parallel:.2f}x")
            except Exception as e:
                print(f"    Failed: {e}")
                optimized_parallel_time = np.nan
                speedup_parallel = np.nan
        else:
            optimized_parallel_time = np.nan
            speedup_parallel = np.nan

        # Verify correctness (check if adjacency matrices are similar)
        if adj_original is not None and adj_optimized is not None:
            nnz_diff = abs(adj_original.nnz - adj_optimized.nnz)
            print(f"  Non-zero elements: original={adj_original.nnz:,}, "
                  f"optimized={adj_optimized.nnz:,}, diff={nnz_diff}")

        # Store results
        results['n_cells'].append(n_cells)
        results['original_time'].append(original_time)
        results['optimized_time'].append(optimized_time)
        results['optimized_parallel_time'].append(optimized_parallel_time)
        results['speedup'].append(speedup)
        results['speedup_parallel'].append(speedup_parallel)

    return results


def plot_results(results, save_path='benchmark_results.png'):
    """Plot benchmark results."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution time
    ax1.plot(results['n_cells'], results['original_time'],
             'o-', label='Original', linewidth=2, markersize=8)
    ax1.plot(results['n_cells'], results['optimized_time'],
             's-', label='Optimized (1 core)', linewidth=2, markersize=8)
    ax1.plot(results['n_cells'], results['optimized_parallel_time'],
             '^-', label='Optimized (4 cores)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Cells', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Speedup
    ax2.plot(results['n_cells'], results['speedup'],
             's-', label='Optimized (1 core)', linewidth=2, markersize=8)
    ax2.plot(results['n_cells'], results['speedup_parallel'],
             '^-', label='Optimized (4 cores)', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
    ax2.set_xlabel('Number of Cells', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Speedup over Original', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark plot saved to: {save_path}")

    return fig


def print_summary(results):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Cells':>10} {'Original':>12} {'Optimized':>12} {'Opt+Parallel':>14} {'Speedup':>10} {'Speedup+Par':>12}")
    print("-" * 80)

    for i in range(len(results['n_cells'])):
        n = results['n_cells'][i]
        orig = results['original_time'][i]
        opt = results['optimized_time'][i]
        opt_par = results['optimized_parallel_time'][i]
        sp = results['speedup'][i]
        sp_par = results['speedup_parallel'][i]

        orig_str = f"{orig:.2f}s" if not np.isnan(orig) else "N/A"
        opt_str = f"{opt:.2f}s" if not np.isnan(opt) else "N/A"
        opt_par_str = f"{opt_par:.2f}s" if not np.isnan(opt_par) else "N/A"
        sp_str = f"{sp:.2f}x" if not np.isnan(sp) else "N/A"
        sp_par_str = f"{sp_par:.2f}x" if not np.isnan(sp_par) else "N/A"

        print(f"{n:>10,} {orig_str:>12} {opt_str:>12} {opt_par_str:>14} {sp_str:>10} {sp_par_str:>12}")

    # Print average speedup
    valid_speedups = [s for s in results['speedup'] if not np.isnan(s)]
    valid_speedups_par = [s for s in results['speedup_parallel'] if not np.isnan(s)]

    if valid_speedups:
        avg_speedup = np.mean(valid_speedups)
        print(f"\nAverage speedup (1 core): {avg_speedup:.2f}x")

    if valid_speedups_par:
        avg_speedup_par = np.mean(valid_speedups_par)
        print(f"Average speedup (4 cores): {avg_speedup_par:.2f}x")


if __name__ == "__main__":
    # Run benchmark with different models
    for mode in ["Radius", "KNN", "mu_std"]:
        k = 150 if mode == "Radius" else 15

        results = benchmark_comparison(
            n_cells_list=[1000, 5000, 10000, 20000, 50000],
            mode=mode,
            k=k
        )

        print_summary(results)
        plot_results(results, save_path=f'benchmark_{mode}.png')

        print("\n" + "="*80 + "\n")

    print("\nBenchmark complete!")
