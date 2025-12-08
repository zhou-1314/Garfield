"""
Example: Multi-GPU Distributed Training with Garfield

This script demonstrates how to train Garfield on multiple GPUs using
PyTorch Lightning's DDP (DistributedDataParallel) strategy.

Key Configuration for Multi-GPU:
    - devices: Number of GPUs to use (e.g., 8 for all A800 GPUs)
    - strategy: 'ddp_find_unused_parameters_true' (REQUIRED for dual-decoder)
    - num_workers: 4 workers per GPU for data loading
    - persistent_workers: True to keep workers alive (faster)
    - use_lightning: True to enable PyTorch Lightning

IMPORTANT: Garfield uses a dual-decoder architecture (edge + node decoders).
Not all parameters are used in every forward pass, so we MUST use
'ddp_find_unused_parameters_true' strategy to avoid DDP errors.

Usage:
    # Simple usage (recommended)
    python examples/train_multi_gpu.py

    # Using torchrun for more control
    torchrun --nproc_per_node=8 examples/train_multi_gpu.py

    # With specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train_multi_gpu.py

Expected Performance (8x A800 GPUs):
    - ~6-7x speedup compared to single GPU
    - Automatic gradient synchronization via DDP
    - Evaluation metrics computed on rank 0 only
"""
import os
import sys
import scanpy as sc
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Garfield as gf

# Enable optimized spatial graph construction (5-20x speedup for large datasets)
os.environ['GARFIELD_USE_OPTIMIZED_GRAPH'] = '1'

# Check GPU availability
if not torch.cuda.is_available():
    print("Error: No CUDA GPUs available. This script requires GPUs for multi-GPU training.")
    exit(1)

num_gpus = torch.cuda.device_count()
# print(f"Found {num_gpus} CUDA GPU(s) available:")
# for i in range(num_gpus):
#     print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Set working directory
workdir = '/home/zhouwg/data1/project/Garfield_review/Results/result_garfield_multi_gpu'
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

print(f"Loading data from {adata_path}...")
adata = sc.read_h5ad(adata_path)
# Ensure adata.X is counts.
# adata.layers['counts'] = adata.X.copy()
adata.X = adata.layers['counts'].copy()
print(f"Data loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"Max count: {adata.X.max()}")

# Configure Garfield parameters for multi-GPU training
user_config = dict(
    # Input options
    adata_list=adata,
    profile='spatial',  # 'RNA', 'ATAC', 'ADT', 'multi-modal', 'spatial'
    data_type='single-modal',
    weight=0.5,
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
    num_workers=4,  # 4 workers per GPU for async data loading
    persistent_workers=True,  # Keep workers alive between epochs (faster)

    # Loss parameters
    include_edge_recon_loss=True,
    include_gene_expr_recon_loss=True,
    lambda_latent_contrastive_instanceloss=1.0,
    lambda_latent_contrastive_clusterloss=0.5,
    lambda_gene_expr_recon=1.0,
    lambda_edge_recon=1.0,
    lambda_latent_adj_recon_loss=0.0,
    lambda_omics_recon_mmd_loss=5.0,

    # Training parameters
    n_epochs=100,
    n_epochs_no_edge_recon=0,
    learning_rate=0.001,
    weight_decay=1e-05,
    gradient_clipping=5,

    # Multi-GPU configuration (NEW Lightning parameters)
    accelerator='gpu',      # Use GPUs
    devices=8,              # Use 8 GPUs (all available A800 GPUs)
    # devices=[0,1,2,3],    # Or specify exact consecutive GPU IDs
    # devices='auto',       # Or auto-detect all available GPUs
    num_nodes=1,            # Single-node training (multi-node possible)
    strategy='ddp_find_unused_parameters_true',  # Required for Garfield's dual-decoder architecture
    precision='32',         # '32', '16-mixed' for faster training

    # Logging and checkpointing
    logger='tensorboard',
    log_every_n_steps=50,
    checkpoint_dir=None,
    save_top_k=1,
    save_last=False,  # Don't save every epoch (performance optimization)

    # Other parameters
    latent_key='garfield_latent',
    reload_best_model=True,
    use_early_stopping=True,
    early_stopping_kwargs=None,
    monitor=True,
    seed=42,
    use_lightning=True,  # Explicitly enable Lightning for multi-GPU
    log_style='auto',
    verbose=True,
)

# Apply configuration
dict_config = gf.settings.set_gf_params(user_config)

# Initialize Garfield model
print("Initializing Garfield model...")
model = gf.Garfield(dict_config)

# Train model (Lightning handles DDP automatically)
print("Starting multi-GPU training...")
devices_info = user_config['devices']
if isinstance(devices_info, int):
    print(f"Using {devices_info} GPUs with strategy '{user_config['strategy']}'")
else:
    print(f"Using GPUs {devices_info} with strategy '{user_config['strategy']}'")
model.train()

# NOTE: In DDP mode, only rank 0 should save the final model
# Lightning handles this automatically in callbacks, but for final save:
import torch
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    # Save trained model (only on rank 0)
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
    # print("\nComputing UMAP...")
    # sc.pp.neighbors(model.adata, use_rep=user_config['latent_key'], key_added=user_config['latent_key'])
    # sc.tl.umap(model.adata, neighbors_key=user_config['latent_key'])

    # # Compute latent Leiden clustering
    # latent_leiden_resolution = 0.5
    # latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
    # latent_key = "garfield_latent"

    # sc.tl.leiden(adata=model.adata,
    #             resolution=latent_leiden_resolution,
    #             key_added=latent_cluster_key,
    #             neighbors_key=latent_key)
    # len(model.adata.obs[latent_cluster_key].unique())


    print("Done!")
