"""
PyTorch Lightning Module for Garfield training.

This module wraps the GNN-VAE model for distributed training with PyTorch Lightning.
"""
import torch
import pytorch_lightning as pl
from typing import Dict, Any


class GarfieldLightningModule(pl.LightningModule):
    """
    Lightning wrapper for Garfield GNN-VAE model.

    This module handles the training and validation steps for the dual-task
    training (edge-level graph reconstruction + node-level gene expression reconstruction).

    Parameters
    ----------
    model : GNNModelVAE
        The pre-instantiated Garfield neural network model.
    learning_rate : float
        Learning rate for AdamW optimizer.
    weight_decay : float
        Weight decay for AdamW optimizer.
    gradient_clipping : float
        Maximum gradient value for clipping (0 = no clipping).
    augment_type : str
        Augmentation type for node-level task ('svd', 'dropout', None).
    lambda_edge_recon : float
        Weight for edge reconstruction loss.
    lambda_gene_expr_recon : float
        Weight for gene expression reconstruction loss.
    lambda_latent_adj_recon_loss : float
        Weight for latent adjacency reconstruction loss.
    lambda_latent_contrastive_instanceloss : float
        Weight for instance-level contrastive loss.
    lambda_latent_contrastive_clusterloss : float
        Weight for cluster-level contrastive loss.
    lambda_omics_recon_mmd_loss : float
        Weight for MMD loss.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clipping: float = 0.0,
        augment_type: str = None,
        lambda_edge_recon: float = 1.0,
        lambda_gene_expr_recon: float = 10.0,
        lambda_latent_adj_recon_loss: float = 1.0,
        lambda_latent_contrastive_instanceloss: float = 1.0,
        lambda_latent_contrastive_clusterloss: float = 1.0,
        lambda_omics_recon_mmd_loss: float = 1.0,
    ):
        super().__init__()

        # Store the neural network model
        self.model = model

        # Store hyperparameters (exclude model to avoid serialization issues)
        self.save_hyperparameters(ignore=['model'])

        # Store training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.augment_type = augment_type

        # Store loss weights
        self.lambda_edge_recon = lambda_edge_recon
        self.lambda_gene_expr_recon = lambda_gene_expr_recon
        self.lambda_latent_adj_recon_loss = lambda_latent_adj_recon_loss
        self.lambda_latent_contrastive_instanceloss = lambda_latent_contrastive_instanceloss
        self.lambda_latent_contrastive_clusterloss = lambda_latent_contrastive_clusterloss
        self.lambda_omics_recon_mmd_loss = lambda_omics_recon_mmd_loss

        # Storage for validation outputs
        self.validation_step_outputs = []

    def forward(self, data_batch, decoder_type, augment_type=None):
        """Forward pass through the model."""
        return self.model(data_batch, decoder_type, augment_type)

    def training_step(self, batch, batch_idx):
        """
        Training step for dual-task learning.

        Parameters
        ----------
        batch : tuple
            (edge_batch, node_batch) from DualGraphDataLoader.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Optimization loss (scalar).
        """
        edge_batch, node_batch = batch

        # Forward pass: node-level task (gene expression reconstruction)
        node_output = self.model(
            data_batch=node_batch,
            decoder_type='omics',
            augment_type=self.augment_type,
        )

        # Forward pass: edge-level task (graph reconstruction)
        edge_output = self.model(
            data_batch=edge_batch,
            decoder_type='graph',
            augment_type=None,
        )

        # Compute all losses
        loss_dict = self.model.loss(
            edge_model_output=edge_output,
            node_model_output=node_output,
            lambda_edge_recon=self.lambda_edge_recon,
            lambda_gene_expr_recon=self.lambda_gene_expr_recon,
            lambda_latent_adj_recon_loss=self.lambda_latent_adj_recon_loss,
            lambda_latent_contrastive_instanceloss=self.lambda_latent_contrastive_instanceloss,
            lambda_latent_contrastive_clusterloss=self.lambda_latent_contrastive_clusterloss,
            lambda_omics_recon_mmd_loss=self.lambda_omics_recon_mmd_loss,
        )

        # Log all losses
        # Only sync across distributed processes if using multiple GPUs (performance optimization)
        sync_dist = self.trainer.world_size > 1 if self.trainer else False

        # Extract batch size from node_batch (number of nodes in this batch)
        # Use node_batch.num_nodes for accurate batch size tracking
        if hasattr(node_batch, 'num_nodes'):
            batch_size = node_batch.num_nodes
        elif hasattr(node_batch, 'batch_size'):
            batch_size = node_batch.batch_size
        else:
            # Fallback: count unique batch indices
            batch_size = len(node_batch.batch.unique()) if hasattr(node_batch, 'batch') else node_batch.x.size(0)

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f"train_{loss_name}",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,  # Only sync on multi-GPU
                batch_size=batch_size,  # Explicit batch size for correct aggregation
            )

        return loss_dict['optim_loss']

    def validation_step(self, batch, batch_idx):
        """
        Validation step for dual-task learning.

        Parameters
        ----------
        batch : tuple
            (edge_batch, node_batch) from DualGraphDataLoader.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Global loss (scalar).
        """
        edge_batch, node_batch = batch

        # Forward pass: node-level task (gene expression reconstruction)
        node_output = self.model(
            data_batch=node_batch,
            decoder_type='omics',
            augment_type=self.augment_type,
        )

        # Forward pass: edge-level task (graph reconstruction)
        edge_output = self.model(
            data_batch=edge_batch,
            decoder_type='graph',
            augment_type=None,
        )

        # Compute all losses
        loss_dict = self.model.loss(
            edge_model_output=edge_output,
            node_model_output=node_output,
            lambda_edge_recon=self.lambda_edge_recon,
            lambda_gene_expr_recon=self.lambda_gene_expr_recon,
            lambda_latent_adj_recon_loss=self.lambda_latent_adj_recon_loss,
            lambda_latent_contrastive_instanceloss=self.lambda_latent_contrastive_instanceloss,
            lambda_latent_contrastive_clusterloss=self.lambda_latent_contrastive_clusterloss,
            lambda_omics_recon_mmd_loss=self.lambda_omics_recon_mmd_loss,
        )

        # Log all losses
        # Only sync across distributed processes if using multiple GPUs (performance optimization)
        sync_dist = self.trainer.world_size > 1 if self.trainer else False

        # Extract batch size from node_batch (number of nodes in this batch)
        if hasattr(node_batch, 'num_nodes'):
            batch_size = node_batch.num_nodes
        elif hasattr(node_batch, 'batch_size'):
            batch_size = node_batch.batch_size
        else:
            # Fallback: count unique batch indices
            batch_size = len(node_batch.batch.unique()) if hasattr(node_batch, 'batch') else node_batch.x.size(0)

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f"val_{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,  # Only sync on multi-GPU
                batch_size=batch_size,  # Explicit batch size for correct aggregation
            )

        # Store for epoch-level aggregation
        self.validation_step_outputs.append(loss_dict['global_loss'])

        return loss_dict['global_loss']

    def on_validation_epoch_end(self):
        """
        Aggregate validation metrics at the end of each epoch.
        """
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('val_global_loss_epoch', avg_loss, sync_dist=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            AdamW optimizer with specified learning rate and weight decay.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        """
        Apply gradient clipping before optimizer step (if configured).

        This matches the behavior of the original trainer's gradient clipping.
        """
        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_value_(
                self.parameters(),
                self.gradient_clipping,
            )
