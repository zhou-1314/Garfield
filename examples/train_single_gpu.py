"""
Example: Single-GPU Training Comparison - Original Trainer vs Optimized Lightning

This script compares the original trainer with the optimized Lightning trainer
to verify consistency of results on single GPU.

Usage:
    python examples/train_single_gpu.py
"""
import os
import sys
import time
import numpy as np
import scanpy as sc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Garfield as gf

# Enable optimized spatial graph construction (5-20x speedup for large datasets)
os.environ['GARFIELD_USE_OPTIMIZED_GRAPH'] = '1'

# Set working directory
workdir = '/home/zhouwg/data1/project/Garfield_review/Results/result_garfield_single_gpu_comparison'
gf.settings.set_workdir(workdir)

# Load example data
adata_path = '/home/zhouwg/data1/project/Garfield_review/data/slideseqv2_mouse_hippocampus.h5ad'

# Check if data exists
if not os.path.exists(adata_path):
    print(f"Error: Data file not found at {adata_path}")
    print("Please update adata_path in the script to point to your data.")
    exit(1)

print("="*80)
print("SINGLE-GPU TRAINING COMPARISON TEST")
print("="*80)
print(f"Comparing Original Trainer vs Optimized Lightning Trainer")
print(f"Data: {adata_path}")
print(f"Working directory: {workdir}")
print("="*80)

# Load data
print("\nLoading data...")
adata = sc.read_h5ad(adata_path)
adata.X = adata.layers['counts'].copy()
print(f"Data shape: {adata.shape}")
print(f"Max count: {adata.X.max()}")

# Shared configuration parameters (same for both trainers)
shared_config = dict(
    # Input options
    profile='spatial',
    data_type='single-modal',
    weight=0.5,
    sample_col=None,

    # Preprocessing options
    used_hvg=True,
    min_cells=3,
    min_features=0,
    keep_mt=False,
    target_sum=1e4,
    rna_n_top_features=3000,
    n_components=50,
    n_neighbors=5,
    metric='euclidean',
    svd_solver='arpack',

    # Data split parameters
    edge_val_ratio=0.1,
    edge_test_ratio=0.0,
    node_val_ratio=0.1,
    node_test_ratio=0.0,

    # Model parameters
    augment_type='dropout',
    svd_q=5,
    use_FCencoder=True,
    conv_type='GAT',
    gnn_layer=2,
    hidden_dims=[128, 128],
    bottle_neck_neurons=20,
    cluster_num=20,
    drop_feature_rate=0.2,
    drop_edge_rate=0.2,
    num_heads=3,
    dropout=0.2,
    concat=True,
    used_edge_weight=True,
    used_DSBN=False,
    used_mmd=False,

    # Dataloader parameters
    num_neighbors=5,
    loaders_n_hops=2,
    edge_batch_size=4096,
    node_batch_size=128,
    num_workers=4,

    # Loss parameters
    include_edge_recon_loss=False,
    include_gene_expr_recon_loss=True,
    lambda_latent_contrastive_instanceloss=1.0,
    lambda_latent_contrastive_clusterloss=0.5,
    lambda_gene_expr_recon=1.0,
    lambda_edge_recon=1.0,
    lambda_latent_adj_recon_loss=0.0,
    lambda_omics_recon_mmd_loss=0.2,

    # Training parameters
    n_epochs=100,
    n_epochs_no_edge_recon=0,
    learning_rate=0.001,
    weight_decay=1e-05,
    gradient_clipping=5,

    # Device configuration
    accelerator='gpu',
    devices=[1],
    strategy='auto',
    precision='32',

    # Logging and checkpointing
    logger='tensorboard',
    log_every_n_steps=50,
    save_top_k=1,
    save_last=True,

    # Other parameters
    latent_key='garfield_latent',
    reload_best_model=True,
    use_early_stopping=True,
    early_stopping_kwargs=None,
    monitor=True,
    seed=42,  # IMPORTANT: Same seed for reproducibility
    log_style='auto',
    verbose=True,
)

# Store results
results = {}

# ============================================================================
# TEST 1: Original Trainer (use_lightning=False)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: ORIGINAL TRAINER (use_lightning=False)")
print("="*80)

# Reload data for fresh test
adata_original = sc.read_h5ad(adata_path)
adata_original.X = adata_original.layers['counts'].copy()

config_original = shared_config.copy()
config_original.update({
    'adata_list': adata_original,
    'use_lightning': False,  # Use original trainer
    'checkpoint_dir': f'{workdir}/checkpoints_original',
})

print("\nInitializing model with original trainer...")
dict_config_original = gf.settings.set_gf_params(config_original)
model_original = gf.Garfield(dict_config_original)

print("Starting training with original trainer...")
start_time = time.time()
model_original.train()  # Now properly overridden to call train_legacy()
train_time_original = time.time() - start_time

# Get embeddings
embeddings_original = model_original.adata.obsm['garfield_latent'].copy()

results['original'] = {
    'embeddings': embeddings_original,
    'train_time': train_time_original,
    'shape': embeddings_original.shape,
    'mean': embeddings_original.mean(),
    'std': embeddings_original.std(),
}

print(f"\n✅ Original Trainer Complete")
print(f"   Training time: {train_time_original:.1f}s")
print(f"   Embeddings shape: {embeddings_original.shape}")
print(f"   Embeddings mean: {embeddings_original.mean():.6f}")
print(f"   Embeddings std: {embeddings_original.std():.6f}")

# Save original results
save_dir_original = f"{workdir}/results_original"
os.makedirs(save_dir_original, exist_ok=True)
model_original.save(
    dir_path=save_dir_original,
    overwrite=True,
    save_adata=True,
    adata_file_name="adata_original.h5ad"
)


# ============================================================================
# TEST 2: Optimized Lightning Trainer (use_lightning=True, lightning_sampling_mode='auto')
# ============================================================================
print("\n" + "="*80)
print("TEST 2: OPTIMIZED LIGHTNING TRAINER (use_lightning=True, lightning_sampling_mode='auto')")
print("="*80)

# Reload data for fresh test
adata_lightning = sc.read_h5ad(adata_path)
adata_lightning.X = adata_lightning.layers['counts'].copy()

config_lightning = shared_config.copy()
config_lightning.update({
    'adata_list': adata_lightning,
    'use_lightning': True,  # Use Lightning trainer
    'lightning_sampling_mode': 'auto',  # Optimized mode (uses global RNG on single GPU)
    'checkpoint_dir': f'{workdir}/checkpoints_lightning',
})

print("\nInitializing model with optimized Lightning trainer...")
dict_config_lightning = gf.settings.set_gf_params(config_lightning)
model_lightning = gf.Garfield(dict_config_lightning)

print("Starting training with optimized Lightning trainer...")
start_time = time.time()
model_lightning.train()  # Now properly overridden to call train_legacy()
train_time_lightning = time.time() - start_time

# Get embeddings
embeddings_lightning = model_lightning.adata.obsm['garfield_latent'].copy()

results['lightning'] = {
    'embeddings': embeddings_lightning,
    'train_time': train_time_lightning,
    'shape': embeddings_lightning.shape,
    'mean': embeddings_lightning.mean(),
    'std': embeddings_lightning.std(),
}

print(f"\n✅ Optimized Lightning Trainer Complete")
print(f"   Training time: {train_time_lightning:.1f}s")
print(f"   Embeddings shape: {embeddings_lightning.shape}")
print(f"   Embeddings mean: {embeddings_lightning.mean():.6f}")
print(f"   Embeddings std: {embeddings_lightning.std():.6f}")

# Save Lightning results
save_dir_lightning = f"{workdir}/results_lightning"
os.makedirs(save_dir_lightning, exist_ok=True)
model_lightning.save(
    dir_path=save_dir_lightning,
    overwrite=True,
    save_adata=True,
    adata_file_name="adata_lightning.h5ad"
)


# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

# Compare embeddings
emb_orig = results['original']['embeddings']
emb_light = results['lightning']['embeddings']

# Compute differences
max_diff = np.max(np.abs(emb_orig - emb_light))
mean_diff = np.mean(np.abs(emb_orig - emb_light))
mse = np.mean((emb_orig - emb_light) ** 2)
rmse = np.sqrt(mse)

# Compute correlations per dimension
correlations = []
for i in range(emb_orig.shape[1]):
    corr = np.corrcoef(emb_orig[:, i], emb_light[:, i])[0, 1]
    correlations.append(abs(corr))
mean_correlation = np.mean(correlations)
min_correlation = np.min(correlations)

# Compare training times
speedup = results['original']['train_time'] / results['lightning']['train_time']

print("\nEmbedding Comparison:")
print(f"  Max absolute difference: {max_diff:.6e}")
print(f"  Mean absolute difference: {mean_diff:.6e}")
print(f"  Root mean squared error: {rmse:.6e}")
print(f"  Mean correlation across dimensions: {mean_correlation:.4f}")
print(f"  Min correlation across dimensions: {min_correlation:.4f}")

print("\nTraining Time Comparison:")
print(f"  Original trainer: {results['original']['train_time']:.1f}s")
print(f"  Lightning trainer: {results['lightning']['train_time']:.1f}s")
print(f"  Speedup: {speedup:.2f}x")

print("\nStatistics Comparison:")
print(f"  Original - Mean: {results['original']['mean']:.6f}, Std: {results['original']['std']:.6f}")
print(f"  Lightning - Mean: {results['lightning']['mean']:.6f}, Std: {results['lightning']['std']:.6f}")

# Determine if results are consistent
print("\n" + "="*80)
print("CONSISTENCY CHECK")
print("="*80)

# Check if embeddings are highly correlated (>0.95 is considered consistent)
# Check if differences are small (mean diff < 0.01 * std)
is_highly_correlated = mean_correlation > 0.95
is_small_diff = mean_diff < 0.01 * results['original']['std']

if is_highly_correlated and is_small_diff:
    print("✅ CONSISTENT: Original trainer and optimized Lightning trainer produce highly similar results!")
    print(f"   - Mean correlation: {mean_correlation:.4f} (target: >0.95)")
    print(f"   - Mean difference: {mean_diff:.6e} (target: <{0.01 * results['original']['std']:.6e})")
elif is_highly_correlated:
    print("⚠️  MOSTLY CONSISTENT: High correlation but some numerical differences")
    print(f"   - Mean correlation: {mean_correlation:.4f} (target: >0.95)")
    print(f"   - Mean difference: {mean_diff:.6e} (target: <{0.01 * results['original']['std']:.6e})")
else:
    print("❌ INCONSISTENT: Results differ significantly between trainers")
    print(f"   - Mean correlation: {mean_correlation:.4f} (target: >0.95)")
    print(f"   - This may indicate different random sampling behavior")

print("\n" + "="*80)
print("SAVED OUTPUTS")
print("="*80)
print(f"Original trainer results: {save_dir_original}/")
print(f"Lightning trainer results: {save_dir_lightning}/")
print(f"Working directory: {workdir}/")

print("\nDone!")
