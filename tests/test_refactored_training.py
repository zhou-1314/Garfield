import os
import argparse
import torch
import anndata as ad
import scanpy as sc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Garfield.data.lightning_datamodule import GarfieldDataModule
from Garfield.trainer.lightning_module import GarfieldLightningModule

def main():
    # --- Hardcoded arguments for testing ---
    args = {
        "data_path": "data/pbmc3k_raw.h5ad",
        "input_dim": 1000,
        "gnn_hidden_dim": 128,
        "gnn_output_dim": 64,
        "gnn_num_layers": 2,
        "gnn_dropout": 0.2,
        "num_of_gnn": 2,
        "num_of_gat": 2,
        "gat_num_heads": 2,
        "num_of_s_gat": 1,
        "s_gat_num_heads": 1,
        "decoder_num_layers": 2,
        "graph_decoder_hidden_dim": 128,
        "graph_decoder_output_dim": 1000,
        "instance_cluster_dim": 64,
        "instance_hidden_dim": 128,
        "instance_output_dim": 64,
        "cluster_hidden_dim": 128,
        "cluster_output_dim": 64,
        "num_of_omics": 1,
        "omic_input_dim": [1000],
        "omic_hidden_dim": [128],
        "num_of_omic_decoder": [2],
        "num_of_omic_encoder": [2],
        "kl_weight": 1e-5,
        "recon_weight": 1.0,
        "contrastive_weight": 0.0, # Disabled for now
        "graph_weight": 0.1,
        "use_gnn_cat": True,
        "use_graph_recon": True,
        "use_instance_cluster_loss": False, # Disabled
        "use_cluster_contrastive_loss": False, # Disabled
        "use_instance_contrastive_loss": False, # Disabled
        "recon_loss_type": "mse",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_workers": 0,
        "max_epochs": 2,
        "patience": 10,
        "check_val_every_n_epoch": 1,
        "omic_names": ["rna"],
        "profile": "RNA",
        "data_type": "Paired",
        "sub_data_type": None,
        "sample_col": None,
        "weight": 0.5,
        "graph_const_method": "KNN",
        "genome": None,
        "use_gene_weight": True,
        "user_cache_path": "./",
        "use_top_pcs": False,
        "used_hvg": True,
        "min_features": 100,
        "min_cells": 3,
        "keep_mt": False,
        "target_sum": 1e4,
        "rna_n_top_features": 2000,
        "atac_n_top_features": 10000,
        "n_components": 50,
        "n_neighbors": 15,
        "metric": "correlation",
        "svd_solver": "arpack",
        "svd_q": 5,
        "use_FCencoder": False,
        "hidden_dims": [64, 32],
        "bottle_neck_neurons": 32,
        "num_heads": 2,
        "dropout": 0.2,
        "concat": True,
        "drop_feature_rate": 0.2,
        "drop_edge_rate": 0.2,
        "used_edge_weight": True,
        "used_DSBN": False,
        "conv_type": "GAT",
        "gnn_layer": 2,
        "cluster_num": 20,
        "include_edge_recon_loss": True,
        "include_gene_expr_recon_loss": True,
        "used_mmd": False,
        "lambda_latent_contrastive_instanceloss": 0.1,
        "lambda_latent_contrastive_clusterloss": 0.1,
        "lambda_gene_expr_recon": 1.0,
        "lambda_latent_adj_recon_loss": 0.1,
        "lambda_edge_recon": 0.1,
        "lambda_omics_recon_mmd_loss": 0.1,
        "n_epochs": 2,
        "n_epochs_no_edge_recon": 0,
        "weight_decay": 1e-4,
        "gradient_clipping": 5.0,
        "latent_key": "garfield_latent",
        "reload_best_model": False,
        "use_early_stopping": True,
        "early_stopping_kwargs": {"patience": 10, "metric_improvement_threshold": 0.0},
        "monitor": True,
        "device_id": 0,
        "verbose": True,
        "log_style": "auto",
        "use_lightning": True,
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        "strategy": "auto",
        "precision": 32,
        "logger": "tensorboard",
        "log_every_n_steps": 1,
        "fast_dev_run": False,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "checkpoint_dir": "./test_checkpoints",
        "save_top_k": 1,
        "save_last": True,
        "persistent_workers": False,

        "hvg_num": 2000,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "work_dir": "result_Garfield_test",
        # New args for GarfieldDataModule
        "label_name": None, # No labels in this dataset
        "used_pca_feat": True,
        "adj_key": "connectivities",
        "edge_val_ratio": 0.1,
        "edge_test_ratio": 0.0,
        "node_val_ratio": 0.1,
        "node_test_ratio": 0.0,
        "augment_type": "svd",
        "num_neighbors": 5,
        "loaders_n_hops": 1,
        "edge_batch_size": 32,
        "node_batch_size": 32,
        "persistent_workers": False,
        "seed": 42,
        "lightning_sampling_mode": 'auto',
    }
    args = argparse.Namespace(**args)

    # --- Data Loading ---
    adata = ad.read_h5ad(args.data_path)
    
    # --- Basic Preprocessing ---
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.input_dim)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=args.num_neighbors, use_rep='X_pca')
    adata.obsm['feat'] = adata.obsm['X_pca']


    args.adata_list = [adata]
    # --- Data Module ---
    data_module = GarfieldDataModule(
        adata=adata,
        label_name=args.label_name,
        used_pca_feat=args.used_pca_feat,
        adj_key=args.adj_key,
        edge_val_ratio=args.edge_val_ratio,
        edge_test_ratio=args.edge_test_ratio,
        node_val_ratio=args.node_val_ratio,
        node_test_ratio=args.node_test_ratio,
        augment_type=args.augment_type,
        num_neighbors=args.num_neighbors,
        loaders_n_hops=args.loaders_n_hops,
        edge_batch_size=args.edge_batch_size,
        node_batch_size=args.node_batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
        lightning_sampling_mode=args.lightning_sampling_mode,
    )

    # --- Model ---
    model = GarfieldLightningModule(args)

    # --- Trainer ---
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.work_dir, "checkpoints"),
        filename='garfield-{epoch:02d}-{val_global_loss:.4f}',
        save_top_k=3,
        monitor='val_global_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_global_loss',
        patience=args.patience,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=args.work_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    # --- Training ---
    trainer.fit(model, data_module)
    print("Test script finished successfully!")

if __name__ == "__main__":
    main()