"""
Example: Single-GPU Training with Garfield

This script demonstrates how to train Garfield on a single GPU using the
new Lightning-based training infrastructure.

Usage:
    python examples/train_single_gpu.py
"""
import os
import sys
import scanpy as sc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Garfield as gf

# Enable optimized spatial graph construction (5-20x speedup for large datasets)
os.environ['GARFIELD_USE_OPTIMIZED_GRAPH'] = '1'

# Set working directory
workdir = '/home/zhouwg/data1/project/Garfield_review/Results/result_garfield_single_gpu2'
gf.settings.set_workdir(workdir)

# Load example data (replace with your own dataset)
# Example: Load pancreas dataset
# adata = sc.read_h5ad('/path/to/your/data.h5ad')
# For this example, we'll use a placeholder path
adata_path = '/home/zhouwg/data1/project/Garfield_review/data/slideseqv2_mouse_hippocampus.h5ad'

# Check if data exists
if not os.path.exists(adata_path):
    print(f"Error: Data file not found at {adata_path}")
    print("Please update adata_path in the script to point to your data.")
    exit(1)

adata = sc.read_h5ad(adata_path)
# Ensure adata.X is counts.
# adata.layers['counts'] = adata.X.copy()
adata.X = adata.layers['counts'].copy()
adata.X.max()

# Configure Garfield parameters for single-GPU training
user_config = dict(
    # Input options
    adata_list=adata,
    profile='spatial',  # 'RNA', 'ATAC', 'ADT', 'multi-modal', 'spatial'
    data_type='single-modal',
    weight=1.0,
    sample_col=None,  # Column name for batch information

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
    use_FCencoder=False,
    conv_type='GAT',  # 'GAT', 'GATv2Conv', or 'GCN'
    gnn_layer=2,
    hidden_dims=[128, 128],
    bottle_neck_neurons=20,
    cluster_num=20,
    drop_feature_rate=0.2,
    drop_edge_rate=0.2,
    num_heads=3,
    dropout=0.2,
    concat=True,
    used_edge_weight=False,
    used_DSBN=False,
    used_mmd=False,

    # Dataloader parameters
    num_neighbors=5,
    loaders_n_hops=2,
    edge_batch_size=4096,
    node_batch_size=1024,
    num_workers=4,  # Increase for async data loading (e.g., 2-4)

    # Loss parameters
    include_edge_recon_loss=False,
    include_gene_expr_recon_loss=True,
    lambda_latent_contrastive_instanceloss=1.0,
    lambda_latent_contrastive_clusterloss=0.5,
    lambda_gene_expr_recon=10.0,
    lambda_edge_recon=1.0,
    lambda_latent_adj_recon_loss=1.0,
    lambda_omics_recon_mmd_loss=5.0,

    # Training parameters
    n_epochs=100,
    n_epochs_no_edge_recon=0,
    learning_rate=0.001,
    weight_decay=1e-05,
    gradient_clipping=5,

    # Single-GPU configuration (NEW Lightning parameters)
    accelerator='gpu',  # 'gpu', 'cpu', or 'auto'
    devices=[1],          # Use 1 GPU
    strategy='auto',    # Lightning auto-selects strategy
    precision='32',     # '32', '16-mixed', or 'bf16-mixed'

    # Logging and checkpointing
    logger='tensorboard',  # 'tensorboard', 'csv', or None
    log_every_n_steps=50,
    checkpoint_dir=None,  # Defaults to workdir/checkpoints
    save_top_k=1,
    save_last=True,

    # Other parameters
    latent_key='garfield_latent',
    reload_best_model=True,
    use_early_stopping=True,
    early_stopping_kwargs=None,
    monitor=True,
    seed=42,
    verbose=True,
)

# Apply configuration
dict_config = gf.settings.set_gf_params(user_config)

# Initialize Garfield model
print("Initializing Garfield model...")
model = gf.Garfield(dict_config)

# Train model
print("Starting training...")
model.train()

# Save trained model
print("\nSaving trained model...")
model_folder_path = f"{workdir}/model"
os.makedirs(model_folder_path, exist_ok=True)
model.save(
    dir_path=model_folder_path,
    overwrite=True,
    save_adata=True,
    adata_file_name="adata_trained.h5ad"
)

print(f"\nTraining complete! Model saved to {model_folder_path}")
print(f"Embeddings stored in adata.obsm['{user_config['latent_key']}']")

# Optionally: Compute UMAP for visualization
print("\nComputing UMAP...")
sc.pp.neighbors(model.adata, use_rep=user_config['latent_key'], key_added=user_config['latent_key'])
sc.tl.umap(model.adata, neighbors_key=user_config['latent_key'])

# Compute latent Leiden clustering
latent_leiden_resolution = 0.5
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
latent_key = "garfield_latent"

sc.tl.leiden(adata=model.adata,
             resolution=latent_leiden_resolution,
             key_added=latent_cluster_key,
             neighbors_key=latent_key)
len(model.adata.obs[latent_cluster_key].unique())

# Visualize (requires matplotlib)
import matplotlib.pyplot as plt
import squidpy as sq

sq.pl.spatial_scatter(
    model.adata,
    color=latent_cluster_key,
    shape=None,
    figsize=(6, 6),
)

plt.savefig("spatial_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
# sc.pl.umap(model.adata, color=['batch', 'celltype'], wspace=0.35)

print("Done!")
