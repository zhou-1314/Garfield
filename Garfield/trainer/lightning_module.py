"""
PyTorch Lightning module for Garfield model, enabling distributed training.
"""
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor

# Ignore UserWarnings from PyTorch Geometric
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class GarfieldLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Garfield's GNNModelVAE.

    This module handles distributed training (DDP), automatic device placement,
    rank-aware logging, and integrates with Lightning's callback system.

    Parameters
    ----------
    model : nn.Module
        The Garfield GNNModelVAE model instance.
    augment_type : str
        Type of augmentation ('svd', 'dropout', None).
    learning_rate : float
        Learning rate for optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for optimizer.
    gradient_clipping : float
        Maximum value for gradient clipping (0 means no clipping).
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
    n_epochs_no_edge_recon : int, optional
        Number of epochs without edge reconstruction (default: 0).
    verbose : bool, optional
        Whether to log all loss components (default: False).
    """

    def __init__(
        self,
        model: nn.Module,
        augment_type: str,
        learning_rate: float,
        weight_decay: float,
        gradient_clipping: float,
        lambda_edge_recon: float,
        lambda_gene_expr_recon: float,
        lambda_latent_adj_recon_loss: float,
        lambda_latent_contrastive_instanceloss: float,
        lambda_latent_contrastive_clusterloss: float,
        lambda_omics_recon_mmd_loss: float,
        n_epochs_no_edge_recon: int = 0,
        verbose: bool = False,
    ):
        super().__init__()

        # Save hyperparameters (excluding model which is too large)
        self.save_hyperparameters(ignore=['model'])

        # Store model and hyperparameters
        self.model = model
        self.augment_type = augment_type
        self.n_epochs_no_edge_recon = n_epochs_no_edge_recon
        self.verbose = verbose

        # Track whether edge reconstruction is active
        self.edge_recon_active = (self.current_epoch >= n_epochs_no_edge_recon)

    def forward(
        self,
        data_batch,
        decoder_type: str,
        augment_type: Optional[str]
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        data_batch : Data
            PyG Data batch.
        decoder_type : str
            Type of decoder ('omics' or 'graph').
        augment_type : str or None
            Type of augmentation.

        Returns
        -------
        Dict[str, Tensor]
            Model outputs.
        """
        return self.model(data_batch, decoder_type, augment_type)

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """
        Single training step.

        Lightning automatically handles:
        - Device placement (no .to(device) needed)
        - Gradient synchronization in DDP
        - Optimizer step (after this returns)

        Parameters
        ----------
        batch : Tuple[Data, Data]
            Tuple of (edge_batch, node_batch) from DualGraphDataLoader.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tensor
            Loss value for optimization.
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

        # Compute multi-task loss
        loss_dict = self.model.loss(
            edge_model_output=edge_output,
            node_model_output=node_output,
            lambda_edge_recon=self.hparams.lambda_edge_recon,
            lambda_gene_expr_recon=self.hparams.lambda_gene_expr_recon,
            lambda_latent_adj_recon_loss=self.hparams.lambda_latent_adj_recon_loss,
            lambda_latent_contrastive_instanceloss=self.hparams.lambda_latent_contrastive_instanceloss,
            lambda_latent_contrastive_clusterloss=self.hparams.lambda_latent_contrastive_clusterloss,
            lambda_omics_recon_mmd_loss=self.hparams.lambda_omics_recon_mmd_loss,
        )

        # Get batch size for logging (use node batch size as reference)
        batch_size = node_batch.num_nodes

        # Log main losses (synchronized across ranks)
        self.log('train_global_loss', loss_dict['global_loss'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)
        self.log('train_optim_loss', loss_dict['optim_loss'],
                 on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=batch_size)

        # Optionally log all loss components (if verbose)
        if self.verbose:
            for key, value in loss_dict.items():
                if key not in ['global_loss', 'optim_loss']:
                    self.log(f'train_{key}', value,
                            on_step=False, on_epoch=True, sync_dist=True,
                            batch_size=batch_size)

        return loss_dict['optim_loss']

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """
        Single validation step.

        Parameters
        ----------
        batch : Tuple[Data, Data]
            Tuple of (edge_batch, node_batch).
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing validation outputs.
        """
        edge_batch, node_batch = batch

        # Forward pass
        node_output = self.model(
            data_batch=node_batch,
            decoder_type='omics',
            augment_type=self.augment_type,
        )
        edge_output = self.model(
            data_batch=edge_batch,
            decoder_type='graph',
            augment_type=None,
        )

        # Compute loss
        loss_dict = self.model.loss(
            edge_model_output=edge_output,
            node_model_output=node_output,
            lambda_edge_recon=self.hparams.lambda_edge_recon,
            lambda_gene_expr_recon=self.hparams.lambda_gene_expr_recon,
            lambda_latent_adj_recon_loss=self.hparams.lambda_latent_adj_recon_loss,
            lambda_latent_contrastive_instanceloss=self.hparams.lambda_latent_contrastive_instanceloss,
            lambda_latent_contrastive_clusterloss=self.hparams.lambda_latent_contrastive_clusterloss,
            lambda_omics_recon_mmd_loss=self.hparams.lambda_omics_recon_mmd_loss,
        )

        # Get batch size for logging (use node batch size as reference)
        batch_size = node_batch.num_nodes

        # Log validation losses
        self.log('val_global_loss', loss_dict['global_loss'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)
        self.log('val_optim_loss', loss_dict['optim_loss'],
                 on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=batch_size)

        if self.verbose:
            for key, value in loss_dict.items():
                if key not in ['global_loss', 'optim_loss']:
                    self.log(f'val_{key}', value,
                            on_step=False, on_epoch=True, sync_dist=True,
                            batch_size=batch_size)

        # Extract predictions for potential metric computation
        edge_recon_probs = torch.sigmoid(edge_output['edge_recon_logits'])
        edge_recon_labels = edge_output['edge_recon_labels']

        return {
            'val_loss': loss_dict['global_loss'],
            'edge_recon_probs': edge_recon_probs.detach(),
            'edge_recon_labels': edge_recon_labels.detach(),
        }

    def on_train_epoch_start(self) -> None:
        """
        Called at the start of each training epoch.
        Updates edge reconstruction active flag based on epoch number.
        """
        self.edge_recon_active = (self.current_epoch >= self.n_epochs_no_edge_recon)

    def configure_optimizers(self):
        """
        Configure optimizer.

        Lightning handles the optimizer step and lr scheduling automatically.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        """
        Called before optimizer step. Implements gradient clipping.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer instance.
        """
        if self.hparams.gradient_clipping > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.hparams.gradient_clipping
            )
