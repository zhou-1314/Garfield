###### Trainer ##########
import warnings
from typing import Optional, Union
from collections import defaultdict
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import copy
import math
import numpy as np
import random
import scanpy as sc
from anndata import concat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.sampler import NeighborSampler
from torch_geometric.loader import NeighborLoader, GraphSAINTEdgeSampler, NodeLoader, DataLoader
import torch_geometric.transforms as T

import matplotlib
from matplotlib import pyplot as plt
from ..data.dataprocessors import prepare_data
from ..data.dataloaders import initialize_dataloaders
from .utils import _cycle_iterable, print_progress, EarlyStopping
from .metrics import eval_metrics
from .basetrainermixin import BaseMixin
# from .transfer_anno import weighted_knn_trainer, weighted_knn_transfer

# from torch.optim.lr_scheduler import StepLR
# from scheduler import CosineAnnealingWarmRestarts
# GAMMA=0.9
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

class GarfieldTrainer(BaseMixin):
    """
    Initializes the GarfieldTrainer class, which handles data preparation, model initialization,
    and training of the Garfield model.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : nn.Module
        The Garfield model to be trained.
    label_name : str
        Column name for labels in the annotated data.
    used_pca_feat : Bool
        Whether used pca features or not for node feature.
    adj_key : str
        Key for the adjacency matrix (e.g., spatial connectivity) in the data.
    edge_val_ratio : float
        Proportion of edges to use for validation in edge-level tasks.
    edge_test_ratio : float
        Proportion of edges to use for testing in edge-level tasks.
    node_val_ratio : float
        Proportion of nodes to use for validation in node-level tasks.
    node_test_ratio : float
        Proportion of nodes to use for testing in node-level tasks.
    augment_type : str
        Type of data augmentation to apply (e.g., 'dropout', 'svd').
    num_neighbors : int
        Number of neighbors to sample for each node in graph-based tasks.
    loaders_n_hops : int
        Number of hops to consider for neighbors in graph-based tasks.
    edge_batch_size : int
        Batch size for edge-level tasks.
    node_batch_size : int or None
        Batch size for node-level tasks. If None, it will be determined automatically.
    reload_best_model : bool
        Whether to reload the best model after training.
    use_early_stopping : bool
        Whether to apply early stopping during training.
    early_stopping_kwargs : dict
        Additional arguments for early stopping (e.g., patience, delta).
    monitor : bool
        Whether to print monitoring logs during training.
    verbose : bool
        Whether to print detailed logs during training.
    seed : int
        Seed for random number generation to ensure reproducibility.
    kwargs : dict
        Additional arguments for training configuration.
    """
    def __init__(self,
                 adata,
                 model: nn.Module,
                 label_name,
                 used_pca_feat,
                 adj_key,
                 edge_val_ratio,
                 edge_test_ratio,
                 node_val_ratio,
                 node_test_ratio,
                 augment_type,
                 num_neighbors,
                 loaders_n_hops,
                 edge_batch_size,
                 node_batch_size,
                 reload_best_model,
                 use_early_stopping,
                 early_stopping_kwargs,
                 monitor,
                 verbose,
                 device_id,
                 seed,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.label_name_ = label_name
        self.used_pca_feat_ = used_pca_feat
        self.adj_key_ = adj_key # spatial_connectivities

        ## data split
        self.edge_val_ratio_ = edge_val_ratio # 0.1
        self.edge_train_ratio_ = 1 - edge_val_ratio
        self.edge_test_ratio_ = edge_test_ratio # 0.
        self.node_val_ratio_ = node_val_ratio
        self.node_train_ratio_ = 1 - node_val_ratio
        self.node_test_ratio_ = node_test_ratio

        ## data process
        self.augment_type_ = augment_type

        ## data loader
        self.n_sampled_neighbors_ = num_neighbors
        self.loaders_n_hops_ = loaders_n_hops
        self.edge_batch_size_ = edge_batch_size # 512
        self.node_batch_size_ = node_batch_size # None,

        ## other parameters
        self.reload_best_model_ = reload_best_model
        self.use_early_stopping_ = use_early_stopping
        self.early_stopping_kwargs_ = early_stopping_kwargs
        self.monitor_ = monitor
        self.verbose_ = verbose
        self.loaders_n_hops_ = kwargs.pop("loaders_n_hops", 1)
        self.gradient_clipping_ = kwargs.pop("gradient_clipping", 0.)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None
        self.early_stopping_kwargs_ = (self.early_stopping_kwargs_ if
                                       self.early_stopping_kwargs_ else {})
        if not "early_stopping_metric" in self.early_stopping_kwargs_:
            if self.edge_val_ratio_ > 0 and self.node_val_ratio_ > 0:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "val_global_loss")
            else:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "train_global_loss")
        self.early_stopping = EarlyStopping(**self.early_stopping_kwargs_)

        print("\n--- INITIALIZING TRAINER ---")
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # seed = 2024
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model.to(self.device)

        # Load dataset
        # Prepare data and get node-level and edge-level training and validation
        # splits
        data_dict = prepare_data(
            adata=self.adata,
            label_name=self.label_name_,
            used_pca_feat=self.used_pca_feat_,
            adj_key=self.adj_key_,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=self.edge_test_ratio_,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=self.node_test_ratio_,)
        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)
        print(f"Number of training nodes: {self.n_nodes_train}")
        print(f"Number of validation nodes: {self.n_nodes_val}")
        print(f"Number of training edges: {self.n_edges_train}")
        print(f"Number of validation edges: {self.n_edges_val}")

        # Determine node batch size automatically if not specified
        if self.node_batch_size_ is None:
            self.node_batch_size_ = int(self.edge_batch_size_ / math.floor(
                self.n_edges_train / self.n_nodes_train))
        print(f"Edge batch size: {self.edge_batch_size_}")
        print(f"Node batch size: {self.node_batch_size_}")

        # Initialize node-level and edge-level dataloaders
        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.n_sampled_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

    def train(self,
              n_epochs,
              n_epochs_no_edge_recon, # : int=0
              learning_rate,
              weight_decay,
              gradient_clipping,
              lambda_edge_recon,
              lambda_gene_expr_recon,
              lambda_latent_adj_recon_loss,
              lambda_latent_contrastive_instanceloss,
              lambda_latent_contrastive_clusterloss,
              lambda_omics_recon_mmd_loss
              ):
        """
        Trains the Garfield model for a specified number of epochs with joint edge-level and node-level tasks.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train the model.
        n_epochs_no_edge_recon : int
            Number of epochs to train without edge reconstruction.
        learning_rate : float
            Learning rate for the optimizer.
        weight_decay : float
            Weight decay (L2 regularization) for the optimizer.
        gradient_clipping : float
            Maximum value for gradient clipping to prevent gradient explosion.
        lambda_edge_recon : float
            Weight for edge reconstruction loss.
        lambda_gene_expr_recon : float
            Weight for gene expression reconstruction loss.
        lambda_latent_adj_recon_loss : float
            Weight for latent adjacency reconstruction loss.
        lambda_latent_contrastive_instanceloss : float
            Weight for instance-level latent contrastive loss.
        lambda_latent_contrastive_clusterloss : float
            Weight for cluster-level latent contrastive loss.
        lambda_omics_recon_mmd_loss : float
            Weight for omics reconstruction MMD loss.

        Returns
        -------
        None
        """
        self.n_epochs_ = n_epochs
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.lr_ = learning_rate
        self.weight_decay_ = weight_decay
        self.gradient_clipping_ = gradient_clipping
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_latent_contrastive_instanceloss = lambda_latent_contrastive_instanceloss
        self.lambda_latent_contrastive_clusterloss = lambda_latent_contrastive_clusterloss
        self.lambda_omics_recon_mmd_loss_ = lambda_omics_recon_mmd_loss

        print("\n--- MODEL TRAINING ---")
        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            if self.epoch < self.n_epochs_no_edge_recon_:
                self.edge_recon_active = False
            else:
                self.edge_recon_active = True

            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0

            # Jointly loop through edge- and node-level batches, repeating node-
            # level batches until edge-level batches are complete
            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader,
                    _cycle_iterable(self.node_train_loader)):  # itertools.cycle
                # resulted in memory leak
                # Forward pass node-level batch
                node_train_data_batch = node_train_data_batch.to(self.device)
                node_train_model_output = self.model(
                    data_batch=node_train_data_batch,
                    decoder_type="omics",
                    augment_type=self.augment_type_)

                # Forward pass edge-level batch
                edge_train_data_batch = edge_train_data_batch.to(self.device)
                edge_train_model_output = self.model(
                    data_batch=edge_train_data_batch,
                    decoder_type="graph",
                    augment_type=None)

                # Calculate training loss
                train_loss_dict = self.model.loss(
                    edge_model_output=edge_train_model_output,
                    node_model_output=node_train_model_output,
                    lambda_edge_recon=lambda_edge_recon,
                    lambda_gene_expr_recon=lambda_gene_expr_recon,
                    lambda_latent_adj_recon_loss=lambda_latent_adj_recon_loss,
                    lambda_latent_contrastive_instanceloss=lambda_latent_contrastive_instanceloss,
                    lambda_latent_contrastive_clusterloss=lambda_latent_contrastive_clusterloss,
                    lambda_omics_recon_mmd_loss=lambda_omics_recon_mmd_loss
                )

                train_global_loss = train_loss_dict["global_loss"]
                train_optim_loss = train_loss_dict["optim_loss"]

                if self.verbose_:
                    for key, value in train_loss_dict.items():
                        self.iter_logs[f"train_{key}"].append(value.item())
                else:
                    self.iter_logs["train_global_loss"].append(
                        train_global_loss.item())
                    self.iter_logs["train_optim_loss"].append(
                        train_optim_loss.item())
                self.iter_logs["n_train_iter"] += 1
                # Optimize for training loss
                self.optimizer.zero_grad()

                train_optim_loss.backward()
                # Clip gradients
                if self.gradient_clipping_ > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clipping_)
                self.optimizer.step()

            # Validate model
            if (self.edge_val_loader is not None and
                    self.node_val_loader is not None):
                self.test_metrics_epoch(
                    lambda_edge_recon=lambda_edge_recon,
                    lambda_gene_expr_recon=lambda_gene_expr_recon,
                    lambda_latent_adj_recon_loss=lambda_latent_adj_recon_loss,
                    lambda_latent_contrastive_instanceloss=lambda_latent_contrastive_instanceloss,
                    lambda_latent_contrastive_clusterloss=lambda_latent_contrastive_clusterloss,
                    lambda_omics_recon_mmd_loss=lambda_omics_recon_mmd_loss)
            elif (self.edge_val_loader is None and
                  self.node_val_loader is not None):
                warnings.warn("You have specified a node validation set but no "
                              "edge validation set. Skipping validation...")
            elif (self.edge_val_loader is not None and
                  self.node_val_loader is None):
                warnings.warn("You have specified an edge validation set but no"
                              " node validation set. Skipping validation...")

            # Convert iteration level logs into epoch level logs
            for key in self.iter_logs:
                if key.startswith("train"):
                    self.epoch_logs[key].append(
                        np.array(self.iter_logs[key]).sum() /
                        self.iter_logs["n_train_iter"])
                if key.startswith("val"):
                    self.epoch_logs[key].append(
                        np.array(self.iter_logs[key]).sum() /
                        self.iter_logs["n_val_iter"])

            # Monitor epoch level logs
            if self.monitor_:
                print_progress(self.epoch, self.epoch_logs, self.n_epochs_)

            # Check early stopping
            if self.use_early_stopping_:
                if self.is_early_stopping():
                    break

        # Track training time and load best model
        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        print(f"Model training finished after {int(minutes)} min {int(seconds)}"
              " sec.")
        if self.best_model_state_dict is not None and self.reload_best_model_:
            print("Using best model state, which was in epoch "
                  f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()
        # Calculate after training validation metrics
        if self.edge_val_loader is not None:
            self.validate_end()

    @torch.no_grad()
    def test_metrics_epoch(self,
                           lambda_edge_recon,
                           lambda_gene_expr_recon,
                           lambda_latent_adj_recon_loss,
                           lambda_latent_contrastive_instanceloss,
                           lambda_latent_contrastive_clusterloss,
                           lambda_omics_recon_mmd_loss
                           ):
        """
        Evaluates the Garfield model at the end of each epoch on validation data.

        Parameters
        ----------
        lambda_edge_recon : float
            Weight for edge reconstruction loss.
        lambda_gene_expr_recon : float
            Weight for gene expression reconstruction loss.
        lambda_latent_contrastive_instanceloss : float
            Weight for instance-level latent contrastive loss.
        lambda_latent_contrastive_clusterloss : float
            Weight for cluster-level latent contrastive loss.
        lambda_omics_recon_mmd_loss : float
            Weight for omics reconstruction MMD loss.

        Returns
        -------
        None
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])

        # Jointly loop through edge- and node-level batches, repeating node-
        # level batches until edge-level batches are complete
        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, _cycle_iterable(self.node_val_loader)):
            # Forward pass node level batch
            node_val_data_batch = node_val_data_batch.to(self.device)
            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder_type="omics",
                augment_type=self.augment_type_)

            # Forward pass edge level batch
            edge_val_data_batch = edge_val_data_batch.to(self.device)
            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder_type="graph",
                augment_type=None)

            # Calculate validation loss
            val_loss_dict = self.model.loss(
                edge_model_output=edge_val_model_output,
                node_model_output=node_val_model_output,
                lambda_edge_recon=lambda_edge_recon,
                lambda_gene_expr_recon=lambda_gene_expr_recon,
                lambda_latent_adj_recon_loss=lambda_latent_adj_recon_loss,
                lambda_latent_contrastive_instanceloss=lambda_latent_contrastive_instanceloss,
                lambda_latent_contrastive_clusterloss=lambda_latent_contrastive_clusterloss,
                lambda_omics_recon_mmd_loss=lambda_omics_recon_mmd_loss
            )

            val_global_loss = val_loss_dict["global_loss"]
            val_optim_loss = val_loss_dict["optim_loss"]
            if self.verbose_:
                for key, value in val_loss_dict.items():
                    self.iter_logs[f"val_{key}"].append(value.item())
            else:
                self.iter_logs["val_global_loss"].append(val_global_loss.item())
                self.iter_logs["val_optim_loss"].append(val_optim_loss.item())
            self.iter_logs["n_val_iter"] += 1

            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated)
        if self.verbose_:
            self.epoch_logs["val_auroc_score"].append(
                val_eval_dict["auroc_score"])
            self.epoch_logs["val_auprc_score"].append(
                val_eval_dict["auprc_score"])
            self.epoch_logs["val_best_acc_score"].append(
                val_eval_dict["best_acc_score"])
            self.epoch_logs["val_best_f1_score"].append(
                val_eval_dict["best_f1_score"])

        self.model.train()

    @torch.no_grad()
    def validate_end(self):
        """
        Evaluates the Garfield model after training on the validation dataset.

        Returns
        -------
        None
        """
        self.model.eval()

        # Get edge-level ground truth and predictions
        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        for edge_val_data_batch in self.edge_val_loader:
            edge_val_data_batch = edge_val_data_batch.to(self.device)

            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder_type="graph",
                augment_type=None)

            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())

        # Get node-level ground truth and predictions
        omics_pred_dict_val_accumulated = np.array([])
        omics_truth_dict_val_accumulated = np.array([])
        for node_val_data_batch in self.node_val_loader:
            node_val_data_batch = node_val_data_batch.to(self.device)

            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder_type="omics",
                augment_type=self.augment_type_)

            omics_truth_dict_val = node_val_model_output["truth_x"]
            omics_pred_dict_val = node_val_model_output["recon_features"]
            omics_pred_dict_val_accumulated = np.append(
                omics_pred_dict_val_accumulated,
                omics_pred_dict_val.detach().cpu().numpy())
            omics_truth_dict_val_accumulated = np.append(
                omics_truth_dict_val_accumulated,
                omics_truth_dict_val.detach().cpu().numpy())

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            omics_recon_pred=omics_pred_dict_val_accumulated,
            omics_recon_truth=omics_truth_dict_val_accumulated)
        print("\n--- MODEL EVALUATION ---")
        print(f"val AUROC score: {val_eval_dict['auroc_score']:.4f}")
        print(f"val AUPRC score: {val_eval_dict['auprc_score']:.4f}")
        print(f"val best accuracy score: {val_eval_dict['best_acc_score']:.4f}")
        print(f"val best F1 score: {val_eval_dict['best_f1_score']:.4f}")
        print(f"val MSE score: {val_eval_dict['gene_expr_mse_score']:.4f}")

    def is_early_stopping(self) -> bool:
        """
        Check whether to apply early stopping, update learning rate and save
        best model state.

        Returns
        ----------
        stop_training:
            If `True`, stop NicheCompass model training.
        """
        early_stopping_metric = self.early_stopping.early_stopping_metric
        current_metric = self.epoch_logs[early_stopping_metric][-1]
        if self.early_stopping.update_state(current_metric):
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(current_metric)
        if reduce_lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor
            print(f"New learning rate is {param_group['lr']}.\n")
        stop_training = not continue_training
        return stop_training

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            # Input options
            'data_dir': dct['data_dir_'],
            'project_name': dct['project_name_'],
            'adata_list': dct['adata_list_'],
            'profile': dct['profile_'],
            'data_type': dct['data_type_'],
            'sub_data_type': dct['sub_data_type_'],
            'weight': dct['weight_'],
            'graph_const_method': dct['graph_const_method_'],
            'genome': dct['genome_'],
            'sample_col': dct['sample_col_'],

            ## whether to use metacell mode
            'metacell': dct['metacell_'],
            'metacell_size': dct['metacell_size_'],
            'single_n_top_genes': dct['single_n_top_genes_'],
            'n_pcs': dct['n_pcs_'],

            # Preprocessing options
            'filter_cells_rna': dct['filter_cells_rna_'],
            'min_features': dct['min_features_'],
            'min_cells': dct['min_cells_'],
            'keep_mt': dct['keep_mt_'],
            'normalize': dct['normalize_'],
            'target_sum': dct['target_sum_'],
            'used_hvg': dct['used_hvg_'],
            'used_scale': dct['used_scale_'],
            'used_feat': dct['used_feat_'],
            'rna_n_top_features': dct['rna_n_top_features_'],
            'atac_n_top_features': dct['atac_n_top_features_'],
            'n_components': dct['n_components_'],
            'n_neighbors': dct['n_neighbors_'],
            'svd_solver': dct['svd_solver_'],
            'method': dct['method_'],
            'metric': dct['metric_'],
            'resolution_tol': dct['resolution_tol_'],
            'leiden_runs': dct['leiden_runs_'],
            'leiden_seed': dct['leiden_seed_'],
            'verbose': dct['verbose_'],

            # Model options
            'augment_type': dct['augment_type_'],
            'svd_q': dct['svd_q_'],
            'gnn_layer': dct['gnn_layer_'],
            'conv_type': dct['conv_type_'],
            'hidden_dims': dct['hidden_dims_'],
            'bottle_neck_neurons': dct['bottle_neck_neurons_'],
            'cluster_num': dct['cluster_num_'],
            'num_heads': dct['num_heads_'],
            'concat': dct['concat_'],
            'used_edge_weight': dct['used_edge_weight_'],
            'used_recon_exp': dct['used_recon_exp_'],
            'used_DSBN': dct['used_DSBN_'],
            'used_mmd': dct['used_mmd_'],
            'test_split': dct['test_split_'],
            'val_split': dct['val_split_'],
            'batch_size': dct['batch_size_'],
            'loader_type': dct['loader_type_'],
            'num_neighbors': dct['num_neighbors_'],
            'epochs': dct['epochs_'],
            'dropout': dct['dropout_'],
            'mmd_temperature': dct['mmd_temperature_'],
            'instance_temperature': dct['instance_temperature_'],
            'cluster_temperature': dct['cluster_temperature_'],
            'l2_reg': dct['l2_reg_'],
            'patience': dct['patience_'],
            'monitor_only_val_losses': dct['monitor_only_val_losses_'],
            'gradient_clipping': dct['gradient_clipping_'],
            'learning_rate': dct['learning_rate_'],
            'weight_decay': dct['weight_decay_'],

            # Other options
            'projection': dct['projection_'],
            'impute': dct['impute_'],
            'outdir': dct['outdir_'],
            'load': False
        }

        return init_params

    ## Loss curve
    def plot_loss_curves(self,
                         title="Losses Curve"
                         ) -> plt.figure:
        """
        Plot loss curves.

        Parameters
        ----------
        loss_dict:
            Dictionary containing the training and validation losses.

        Returns
        ----------
        fig:
            Matplotlib figure of loss curves.
        """
        # Plot epochs as integers
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot loss
        train_global_loss = self.iter_logs["train_global_loss"]
        train_optim_loss = self.iter_logs["train_optim_loss"]
        val_global_loss = self.iter_logs["val_global_loss"]
        val_optim_loss = self.iter_logs["val_optim_loss"]
        # x = np.linspace(0, len(train_global_loss), num=len(train_global_loss))
        plt.plot(train_global_loss, label="Train_global_loss")
        plt.plot(train_optim_loss, label="Train_optim_loss")
        plt.plot(val_global_loss, label="Val_global_loss")
        plt.plot(val_optim_loss, label="Val_optim_loss")
        # for loss_key, loss in self.iter_logs.items():
        #     plt.plot(loss, label=loss_key)
        plt.title(title)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend() # loc="upper right"

        # Retrieve figure
        fig = plt.gcf()
        plt.close()
        return fig



