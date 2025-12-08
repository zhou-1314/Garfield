import warnings
from types import SimpleNamespace
from typing import Literal, List, Optional, Tuple, Union

import torch
import numpy as np
from anndata import AnnData

from .._settings import settings
from .basemodelmixin import BaseModelMixin
from ..data.dataprocessors import prepare_data
from ..preprocessing.preprocess import DataProcess
from ..data.dataloaders import initialize_dataloaders
from .utils import weighted_knn_trainer, weighted_knn_transfer
from ..nn.encoders import GATEncoder, GCNEncoder
from ..modules.GNNModelVAE import GNNModelVAE


class Garfield(torch.nn.Module, BaseModelMixin):
    """
    Garfield: Graph-based Contrastive Learning enable Fast Single-Cell Embedding
    """

    def __init__(self, gf_params):
        super(Garfield, self).__init__()
        if gf_params is None:
            gf_params = settings.gf_params.copy()
        else:
            assert isinstance(gf_params, dict), "`gf_params` must be dict"

        self.args = SimpleNamespace(**gf_params)
        # data preprocessing parameters
        self.adata_list_ = self.args.adata_list
        self.profile_ = self.args.profile
        self.data_type_ = self.args.data_type
        self.sub_data_type_ = self.args.sub_data_type
        self.sample_col_ = self.args.sample_col
        self.weight_ = self.args.weight
        self.graph_const_method_ = self.args.graph_const_method
        self.genome_ = self.args.genome
        self.use_gene_weight_ = self.args.use_gene_weight
        self.user_cache_path_ = self.args.user_cache_path
        self.use_top_pcs_ = self.args.use_top_pcs
        self.used_hvg_ = self.args.used_hvg
        self.min_features_ = self.args.min_features
        self.min_cells_ = self.args.min_cells
        self.keep_mt_ = self.args.keep_mt
        self.target_sum_ = self.args.target_sum
        self.rna_n_top_features_ = self.args.rna_n_top_features
        self.atac_n_top_features_ = self.args.atac_n_top_features
        self.n_components_ = self.args.n_components
        self.n_neighbors_ = self.args.n_neighbors
        self.metric_ = self.args.metric
        self.svd_solver_ = self.args.svd_solver
        # datasets
        self.used_pca_feat_ = self.args.used_pca_feat
        self.adj_key_ = self.args.adj_key
        # data split parameters
        self.edge_val_ratio_ = self.args.edge_val_ratio
        self.edge_test_ratio_ = self.args.edge_test_ratio
        self.node_val_ratio_ = self.args.node_val_ratio
        self.node_test_ratio_ = self.args.node_test_ratio
        # model parameters
        self.augment_type_ = self.args.augment_type
        self.svd_q_ = self.args.svd_q  # if augment_type == 'svd'
        self.use_FCencoder_ = self.args.use_FCencoder
        self.hidden_dims_ = self.args.hidden_dims
        self.bottle_neck_neurons_ = self.args.bottle_neck_neurons
        self.num_heads_ = self.args.num_heads
        self.dropout_ = self.args.dropout
        self.concat_ = self.args.concat
        self.drop_feature_rate_ = self.args.drop_feature_rate
        self.drop_edge_rate_ = self.args.drop_edge_rate
        self.used_edge_weight_ = self.args.used_edge_weight
        self.used_DSBN_ = self.args.used_DSBN
        # self.n_domain_ = self.args.num_classes
        self.conv_type_ = self.args.conv_type
        self.gnn_layer_ = self.args.gnn_layer
        self.cluster_num_ = self.args.cluster_num
        # data loader parameters
        self.num_neighbors_ = self.args.num_neighbors
        self.loaders_n_hops_ = self.args.loaders_n_hops
        self.edge_batch_size_ = self.args.edge_batch_size
        self.node_batch_size_ = self.args.node_batch_size
        # loss parameters
        self.include_edge_recon_loss_ = self.args.include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = self.args.include_gene_expr_recon_loss
        self.used_mmd_ = self.args.used_mmd
        self.lambda_latent_contrastive_instanceloss_ = (
            self.args.lambda_latent_contrastive_instanceloss
        )
        self.lambda_latent_contrastive_clusterloss_ = (
            self.args.lambda_latent_contrastive_clusterloss
        )
        self.lambda_gene_expr_recon_ = self.args.lambda_gene_expr_recon
        self.lambda_latent_adj_recon_loss_ = self.args.lambda_latent_adj_recon_loss
        self.lambda_edge_recon_ = self.args.lambda_edge_recon
        self.lambda_omics_recon_mmd_loss_ = self.args.lambda_omics_recon_mmd_loss
        # train parameters
        self.n_epochs_ = self.args.n_epochs
        self.n_epochs_no_edge_recon_ = self.args.n_epochs_no_edge_recon
        self.learning_rate_ = self.args.learning_rate
        self.weight_decay_ = self.args.weight_decay
        self.gradient_clipping_ = self.args.gradient_clipping
        # other parameters
        self.latent_key_ = self.args.latent_key
        self.reload_best_model_ = self.args.reload_best_model
        self.use_early_stopping_ = self.args.use_early_stopping
        self.early_stopping_kwargs_ = self.args.early_stopping_kwargs
        self.monitor_ = self.args.monitor
        self.seed_ = self.args.seed
        self.device_id_ = self.args.device_id
        self.verbose_ = self.args.verbose
        self.log_style_ = self.args.log_style
        self.use_lightning_ = self.args.use_lightning
        self.lightning_sampling_mode_ = self.args.lightning_sampling_mode

        # Set seed for reproducibility
        np.random.seed(self.seed_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
        else:
            torch.manual_seed(self.seed_)

        # Data load and preprocessing
        print("--- DATA LOADING AND PREPROCESSING ---")
        self.adata = DataProcess(
            adata_list=self.adata_list_,
            profile=self.profile_,
            data_type=self.data_type_,
            sub_data_type=self.sub_data_type_,
            sample_col=self.sample_col_,
            genome=self.genome_,
            weight=self.weight_,
            graph_const_method=self.graph_const_method_,
            use_gene_weight=self.use_gene_weight_,
            user_cache_path=self.user_cache_path_,
            use_top_pcs=self.use_top_pcs_,
            used_hvg=self.used_hvg_,
            min_features=self.min_features_,
            min_cells=self.min_cells_,
            keep_mt=self.keep_mt_,
            target_sum=self.target_sum_,
            rna_n_top_features=self.rna_n_top_features_,
            atac_n_top_features=self.atac_n_top_features_,
            n_components=self.n_components_,
            n_neighbors=self.n_neighbors_,
            metric=self.metric_,
            svd_solver=self.svd_solver_,
        )
        # set up model
        if not self.used_pca_feat_:
            self.num_features_ = self.adata.n_vars
        else:
            self.num_features_ = self.adata.obsm["feat"].shape[1]

        if self.sample_col_ is not None:
            try:
                self.n_domain_ = len(self.adata.obs[self.sample_col_].unique())
            except KeyError:
                self.n_domain_ = len(self.adata.obs["rna:" + self.sample_col_].unique())
        else:
            self.n_domain_ = None
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        ## 选择 encoder
        ## 设定参数
        if self.conv_type_ in ["GAT", "GATv2Conv"]:
            encoder = GATEncoder(
                in_channels=self.num_features_,
                hidden_dims=self.hidden_dims_,
                latent_dim=self.bottle_neck_neurons_,
                conv_type=self.conv_type_,
                use_FCencoder=self.use_FCencoder_,
                num_heads=self.num_heads_,
                dropout=self.dropout_,
                concat=self.concat_,
                num_domains=1,  # batch normalization
                drop_feature_rate=self.drop_feature_rate_,
                drop_edge_rate=self.drop_edge_rate_,
                used_edge_weight=self.used_edge_weight_,
                svd_q=self.svd_q_,
                used_DSBN=self.used_DSBN_,
            )
        else:
            encoder = GCNEncoder(
                in_channels=self.num_features_,
                hidden_dims=self.hidden_dims_,
                latent_dim=self.bottle_neck_neurons_,
                use_FCencoder=self.use_FCencoder_,
                dropout=self.dropout_,
                num_domains=1,  # batch normalization
                drop_feature_rate=self.drop_feature_rate_,
                drop_edge_rate=self.drop_edge_rate_,
                used_edge_weight=self.used_edge_weight_,
                svd_q=self.svd_q_,
                used_DSBN=self.used_DSBN_,
            )

        ## GCNModelVAE
        self.model = GNNModelVAE(
            encoder=encoder,
            bottle_neck_neurons=self.bottle_neck_neurons_,
            hidden_dims=self.hidden_dims_,
            feature_dim=self.num_features_,
            num_heads=self.num_heads_,
            dropout=self.dropout_,
            concat=self.concat_,
            n_domain=self.n_domain_,
            used_edge_weight=self.used_edge_weight_,
            used_DSBN=self.used_DSBN_,
            conv_type=self.conv_type_,
            gnn_layer=self.gnn_layer_,
            cluster_num=self.cluster_num_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            used_mmd=self.used_mmd_,
        )
        self.is_trained_ = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _should_use_lightning(self):
        """
        Determine whether to use PyTorch Lightning trainer or original trainer.

        Returns
        -------
        bool
            True if should use Lightning, False for original trainer.
        """
        use_lightning = self.use_lightning_

        if use_lightning == 'auto':
            # Auto-detect: use Lightning only for multi-GPU/multi-node scenarios
            devices = self.args.devices
            num_nodes = self.args.num_nodes

            # Check if multi-GPU or multi-node
            if num_nodes > 1:
                return True
            if isinstance(devices, list) and len(devices) > 1:
                return True
            if isinstance(devices, int) and devices > 1:
                return True

            # Single device: use original trainer for better performance
            return False
        else:
            # Explicit True/False
            return bool(use_lightning)

    def _validate_distributed_params(self):
        """
        Validate distributed training parameters to catch configuration errors early.

        Raises
        ------
        ValueError
            If invalid parameter values are detected.
        """
        # Validate precision
        valid_precisions = ['32', '16-mixed', 'bf16-mixed', '64', 'bf16', '16', 32, 16, 64]
        if self.args.precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{self.args.precision}'. "
                f"Valid options: {valid_precisions}"
            )

        # Validate accelerator
        valid_accelerators = ['auto', 'cpu', 'gpu', 'tpu', 'cuda', 'mps']
        if self.args.accelerator not in valid_accelerators:
            raise ValueError(
                f"Invalid accelerator '{self.args.accelerator}'. "
                f"Valid options: {valid_accelerators}"
            )

        # Validate devices
        if isinstance(self.args.devices, int):
            if self.args.devices < 0:
                raise ValueError(f"devices must be >= 0, got {self.args.devices}")
        elif isinstance(self.args.devices, list):
            if len(self.args.devices) == 0:
                raise ValueError("devices list cannot be empty")
        elif self.args.devices != 'auto':
            raise ValueError(
                f"devices must be int, list, or 'auto', got {type(self.args.devices)}"
            )

        # Validate num_nodes
        if self.args.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.args.num_nodes}")

        # Validate num_workers
        if self.args.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.args.num_workers}")

        # Validate persistent_workers
        if self.args.persistent_workers and self.args.num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0. "
                "Set num_workers >= 1 or persistent_workers=False."
            )

        # Validate accumulate_grad_batches
        if self.args.accumulate_grad_batches < 1:
            raise ValueError(
                f"accumulate_grad_batches must be >= 1, got {self.args.accumulate_grad_batches}"
            )

        # Validate lightning_sampling_mode
        valid_sampling_modes = ['auto', 'legacy', 'optimized']
        if self.lightning_sampling_mode_ not in valid_sampling_modes:
            raise ValueError(
                f"Invalid lightning_sampling_mode '{self.lightning_sampling_mode_}'. "
                f"Valid options: {valid_sampling_modes}"
            )

        # Validate use_lightning
        if self.use_lightning_ not in ['auto', True, False]:
            raise ValueError(
                f"Invalid use_lightning '{self.use_lightning_}'. "
                f"Valid options: 'auto', True, False"
            )

    def train_legacy(self, **trainer_kwargs):
        """
        Train the Garfield model.

        Automatically selects between PyTorch Lightning trainer (for distributed
        training) and original trainer (for single GPU, better performance).

        For distributed training (multi-GPU, multi-node), PyTorch Lightning is used.
        For single GPU/CPU, the original faster trainer is used by default.

        You can override this behavior with the `use_lightning` parameter:
        - 'auto' (default): Automatic selection based on hardware configuration
        - True: Force PyTorch Lightning trainer
        - False: Force original trainer

        Parameters
        ----------
        trainer_kwargs : dict
            Additional keyword arguments passed to pl.Trainer (Lightning only).
        """
        # Validate distributed training parameters
        self._validate_distributed_params()

        # Determine which trainer to use
        if self._should_use_lightning():
            print("\n--- Using PyTorch Lightning Trainer (Distributed Training) ---")
            self._train_with_lightning(**trainer_kwargs)
        else:
            print("\n--- Using Original Trainer (Single GPU) ---")
            if trainer_kwargs:
                import warnings
                warnings.warn(
                    "trainer_kwargs are ignored when using the original trainer. "
                    "Set use_lightning=True to use PyTorch Lightning with custom kwargs."
                )
            self._train_with_original_trainer()

    def _train_with_lightning(self, **trainer_kwargs):
        """
        Train the Garfield model using PyTorch Lightning.

        Supports both single-GPU and multi-GPU distributed training (DDP).

        Parameters
        ----------
        trainer_kwargs : dict
            Additional keyword arguments passed to pl.Trainer.
        """
        import os

        # Error handling for PyTorch Lightning import
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
            from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
        except ImportError as e:
            raise ImportError(
                "PyTorch Lightning is required for distributed training but not installed.\n"
                "Install with: pip install pytorch-lightning\n"
                "Alternatively, set use_lightning=False to use the original trainer."
            ) from e

        from ..trainer.lightning_module import GarfieldLightningModule
        from ..data.lightning_datamodule import GarfieldDataModule
        from ..trainer.custom_callbacks import NotebookProgressBar, _is_notebook

        print("Initializing PyTorch Lightning components...")

        # 1. Create Lightning DataModule
        datamodule = GarfieldDataModule(
            adata=self.adata,
            label_name=self.sample_col_,
            used_pca_feat=self.used_pca_feat_,
            adj_key=self.adj_key_,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=self.edge_test_ratio_,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=self.node_test_ratio_,
            augment_type=self.augment_type_,
            num_neighbors=self.num_neighbors_,
            loaders_n_hops=self.loaders_n_hops_,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            seed=self.seed_,
            lightning_sampling_mode=self.lightning_sampling_mode_,
        )

        # 2. Create Lightning Module
        # Pass the GNN-VAE model (self.model) which is the actual neural network
        lightning_model = GarfieldLightningModule(
            model=self.model,
            learning_rate=self.learning_rate_,
            weight_decay=self.weight_decay_,
            gradient_clipping=self.gradient_clipping_,
            augment_type=self.augment_type_,
            lambda_edge_recon=self.lambda_edge_recon_,
            lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
            lambda_latent_adj_recon_loss=self.lambda_latent_adj_recon_loss_,
            lambda_latent_contrastive_instanceloss=self.lambda_latent_contrastive_instanceloss_,
            lambda_latent_contrastive_clusterloss=self.lambda_latent_contrastive_clusterloss_,
            lambda_omics_recon_mmd_loss=self.lambda_omics_recon_mmd_loss_,
        )

        # 3. Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_dir = self.args.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.args.user_cache_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='garfield-{epoch:02d}-{val_global_loss:.4f}',
            monitor='val_global_loss' if self.edge_val_ratio_ > 0 and self.node_val_ratio_ > 0 else 'train_global_loss',
            mode='min',
            save_top_k=self.args.save_top_k,
            save_last=self.args.save_last,
            verbose=self.monitor_,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback
        if self.use_early_stopping_:
            early_stop_kwargs = self.early_stopping_kwargs_ or {}
            early_stop_callback = EarlyStopping(
                monitor=early_stop_kwargs.get('early_stopping_metric',
                       'val_global_loss' if self.edge_val_ratio_ > 0 else 'train_global_loss'),
                min_delta=early_stop_kwargs.get('metric_improvement_threshold', 0.0),
                patience=early_stop_kwargs.get('patience', 8),
                mode='min',
                verbose=self.monitor_,
            )
            callbacks.append(early_stop_callback)

        # Learning rate monitor
        if self.monitor_:
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor)

        # Determine whether to use notebook-style progress bar
        use_notebook_style = False
        if self.log_style_ == 'auto':
            # Auto-detect notebook environment
            use_notebook_style = _is_notebook()
        elif self.log_style_ == 'notebook':
            # Force notebook style
            use_notebook_style = True
        # else: log_style_ == 'lightning', use default Lightning progress bar

        # Add notebook-style progress bar callback if needed
        if use_notebook_style:
            notebook_progress = NotebookProgressBar(
                n_epochs=self.n_epochs_,
                verbose=self.verbose_
            )
            callbacks.append(notebook_progress)

        # 4. Setup logger
        logger_type = self.args.logger
        if logger_type == 'tensorboard':
            logger = TensorBoardLogger(
                save_dir=self.args.user_cache_path,
                name='garfield_logs',
            )
        elif logger_type == 'csv':
            logger = CSVLogger(
                save_dir=self.args.user_cache_path,
                name='logs',
            )
        elif logger_type is None or logger_type == 'none':
            logger = False
        else:
            # Default to TensorBoard
            logger = TensorBoardLogger(
                save_dir=self.args.user_cache_path,
                name='garfield_logs',
            )

        # 5. Set deterministic mode if seed is provided
        if self.seed_ is not None:
            pl.seed_everything(self.seed_, workers=True)

        # 6. Create Lightning Trainer
        # Disable Lightning's default progress bar when using custom notebook style
        enable_lightning_progress_bar = self.monitor_ and not use_notebook_style

        trainer = pl.Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            num_nodes=self.args.num_nodes,
            strategy=self.args.strategy,
            precision=self.args.precision,
            max_epochs=self.n_epochs_,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=self.args.log_every_n_steps,
            accumulate_grad_batches=self.args.accumulate_grad_batches,  # Gradient accumulation
            deterministic=False,  # Disabled for performance; override via trainer_kwargs if needed
            enable_progress_bar=enable_lightning_progress_bar,
            enable_model_summary=self.monitor_,
            fast_dev_run=self.args.fast_dev_run,
            limit_train_batches=self.args.limit_train_batches,
            limit_val_batches=self.args.limit_val_batches,
            **trainer_kwargs,
        )

        # 7. Train
        trainer.fit(lightning_model, datamodule=datamodule)

        # 8. Load best model if requested
        if self.reload_best_model_ and checkpoint_callback.best_model_path:
            print(f"\nLoading best model from {checkpoint_callback.best_model_path}")
            best_lightning_model = GarfieldLightningModule.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                model=self.model,
            )
            # Update wrapped model with best weights
            self.model.load_state_dict(best_lightning_model.model.state_dict())

        # 9. Set model to eval mode
        self.model.eval()
        self.is_trained_ = True

        # Store node_batch_size from datamodule
        self.node_batch_size_ = datamodule.node_batch_size

        # Store trainer reference
        self.trainer = trainer

        # 9.5. Perform final model evaluation on validation set
        self._perform_final_evaluation(datamodule)

        # 10. Compute embeddings
        print("\n--- COMPUTING LATENT EMBEDDINGS ---")
        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
            adata=self.adata,
            adj_key=self.adj_key_,
            return_mu_std=True,
            node_batch_size=self.node_batch_size_,
        )
        print(f"Embeddings stored in adata.obsm['{self.latent_key_}']")

    def _perform_final_evaluation(self, datamodule):
        """
        Perform final model evaluation on validation set.

        This method computes evaluation metrics (AUROC, AUPRC, accuracy, F1, MSE)
        on the validation set after training completes, matching the behavior of
        the original trainer.

        Parameters
        ----------
        datamodule : GarfieldDataModule
            The Lightning DataModule containing validation dataloaders.
        """
        # Skip evaluation if no validation data
        if self.edge_val_ratio_ == 0 and self.node_val_ratio_ == 0:
            return

        import numpy as np
        import torch
        from ..trainer.metrics import eval_metrics

        # In DDP mode, only perform evaluation on rank 0 to avoid redundant computation
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        print("\n--- MODEL EVALUATION ---")

        # Get validation dataloader (returns a single DualGraphDataLoader)
        val_dataloader = datamodule.val_dataloader()

        if val_dataloader is None:
            print("No validation data available for evaluation.")
            return

        # Collect edge reconstruction predictions and labels
        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])

        # Collect gene expression predictions and ground truth
        omics_pred_dict_val_accumulated = np.array([])
        omics_truth_dict_val_accumulated = np.array([])

        # Get device from model parameters (standard PyTorch way)
        device = next(self.model.parameters()).device

        # Iterate over DualGraphDataLoader which yields (edge_batch, node_batch) tuples
        with torch.no_grad():
            for edge_val_data_batch, node_val_data_batch in val_dataloader:
                # Process edge batch
                if hasattr(edge_val_data_batch, 'to'):
                    edge_val_data_batch = edge_val_data_batch.to(device)

                edge_val_model_output = self.model(
                    data_batch=edge_val_data_batch,
                    decoder_type="graph",
                    augment_type=None
                )

                # Calculate edge evaluation metrics
                edge_recon_probs_val = torch.sigmoid(
                    edge_val_model_output["edge_recon_logits"]
                )
                edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
                edge_recon_probs_val_accumulated = np.append(
                    edge_recon_probs_val_accumulated,
                    edge_recon_probs_val.detach().cpu().numpy(),
                )
                edge_recon_labels_val_accumulated = np.append(
                    edge_recon_labels_val_accumulated,
                    edge_recon_labels_val.detach().cpu().numpy(),
                )

                # Process node batch
                if hasattr(node_val_data_batch, 'to'):
                    node_val_data_batch = node_val_data_batch.to(device)

                node_val_model_output = self.model(
                    data_batch=node_val_data_batch,
                    decoder_type="omics",
                    augment_type=self.augment_type_,
                )

                # Calculate node evaluation metrics
                omics_truth_dict_val = node_val_model_output["truth_x"]
                omics_pred_dict_val = node_val_model_output["recon_features"]
                omics_pred_dict_val_accumulated = np.append(
                    omics_pred_dict_val_accumulated,
                    omics_pred_dict_val.detach().cpu().numpy(),
                )
                omics_truth_dict_val_accumulated = np.append(
                    omics_truth_dict_val_accumulated,
                    omics_truth_dict_val.detach().cpu().numpy(),
                )

        # Compute evaluation metrics
        # DualGraphDataLoader always provides both edge and node batches
        if len(edge_recon_probs_val_accumulated) > 0 and len(omics_pred_dict_val_accumulated) > 0:
            val_eval_dict = eval_metrics(
                edge_recon_probs=edge_recon_probs_val_accumulated,
                edge_labels=edge_recon_labels_val_accumulated,
                omics_recon_pred=omics_pred_dict_val_accumulated,
                omics_recon_truth=omics_truth_dict_val_accumulated,
            )
            print(f"val AUROC score: {val_eval_dict['auroc_score']:.4f}")
            print(f"val AUPRC score: {val_eval_dict['auprc_score']:.4f}")
            print(f"val best accuracy score: {val_eval_dict['best_acc_score']:.4f}")
            print(f"val best F1 score: {val_eval_dict['best_f1_score']:.4f}")
            print(f"val MSE score: {val_eval_dict['gene_expr_mse_score']:.4f}")
        else:
            print("Warning: Incomplete validation data collected.")

    def _train_with_original_trainer(self):
        """
        Train the Garfield model using the original optimized trainer.

        This trainer is faster for single GPU scenarios but does not support
        distributed training across multiple GPUs or nodes.
        """
        from ..trainer.trainer import GarfieldTrainer

        print("Initializing original Garfield trainer...")

        # Initialize the original trainer
        trainer = GarfieldTrainer(
            adata=self.adata,
            model=self.model,
            label_name=self.sample_col_,
            used_pca_feat=self.used_pca_feat_,
            adj_key=self.adj_key_,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=self.edge_test_ratio_,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=self.node_test_ratio_,
            augment_type=self.augment_type_,
            num_neighbors=self.num_neighbors_,
            loaders_n_hops=self.loaders_n_hops_,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            reload_best_model=self.reload_best_model_,
            use_early_stopping=self.use_early_stopping_,
            early_stopping_kwargs=self.early_stopping_kwargs_,
            monitor=self.monitor_,
            verbose=self.verbose_,
            device_id=self.device_id_ if self.device_id_ is not None else 0,
            seed=self.seed_,
        )

        # Train the model
        trainer.train(
            n_epochs=self.n_epochs_,
            n_epochs_no_edge_recon=self.n_epochs_no_edge_recon_,
            learning_rate=self.learning_rate_,
            weight_decay=self.weight_decay_,
            gradient_clipping=self.gradient_clipping_,
            lambda_edge_recon=self.lambda_edge_recon_,
            lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
            lambda_latent_adj_recon_loss=self.lambda_latent_adj_recon_loss_,
            lambda_latent_contrastive_instanceloss=self.lambda_latent_contrastive_instanceloss_,
            lambda_latent_contrastive_clusterloss=self.lambda_latent_contrastive_clusterloss_,
            lambda_omics_recon_mmd_loss=self.lambda_omics_recon_mmd_loss_,
        )

        # Store trainer reference
        self.trainer = trainer

        # Set model to trained
        self.is_trained_ = True

        # Store node_batch_size
        self.node_batch_size_ = trainer.node_batch_size_

        # Compute embeddings
        print("\n--- COMPUTING LATENT EMBEDDINGS ---")
        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
            adata=self.adata,
            adj_key=self.adj_key_,
            return_mu_std=True,
            node_batch_size=self.node_batch_size_,
        )
        print(f"Embeddings stored in adata.obsm['{self.latent_key_}']")

    def train(self, mode: bool = True, **trainer_kwargs):
        """
        Train the Garfield model or set training mode.

        This method overrides torch.nn.Module.train() to provide dual functionality:
        - If mode=False: Sets model to evaluation mode (PyTorch behavior)
        - If mode=True with no kwargs: Sets model to training mode (PyTorch behavior)
        - If mode=True with kwargs: Trains the model using train_legacy()

        Parameters
        ----------
        mode : bool
            If True, sets training mode or trains the model.
            If False, sets evaluation mode.
        **trainer_kwargs
            If provided, triggers actual model training via train_legacy().

        Returns
        -------
        self
            For method chaining.

        Examples
        --------
        >>> # Train the model (actual training)
        >>> model = Garfield({'adata_list': [adata]})
        >>> model.train()  # Trains the model

        >>> # Set to evaluation mode
        >>> model.train(False)  # or model.eval()

        >>> # Pass additional trainer kwargs
        >>> model.train(some_trainer_arg=value)
        """
        # If mode is False, set to eval mode (PyTorch behavior)
        if not mode:
            return super().train(False)

        # If no trainer_kwargs and mode is True, this is likely actual training request
        # not just setting training mode
        if mode and not trainer_kwargs:
            # Actual training - delegate to train_legacy
            return self.train_legacy(**trainer_kwargs)

        # If trainer_kwargs provided, definitely training
        if trainer_kwargs:
            return self.train_legacy(**trainer_kwargs)

        # Fallback to PyTorch behavior (set training mode)
        return super().train(True)

    # embedding
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        adj_key: str = "connectivities",
        return_mu_std: bool = False,
        node_batch_size: int = 64,
        dtype: type = np.float64,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the latent representation / gene program scores from a trained model.

        Parameters
        ----------
        adata:
            AnnData object to get the latent representation for. If ´None´, uses
            the adata object stored in the model instance.
        adj_key:
            Key under which the sparse adjacency matrix is stored in
            ´adata.obsp´.
        return_mu_std:
            If `True`, return ´mu´ and ´std´ instead of latent features ´z´.
        node_batch_size:
            Batch size used during data loading.
        dtype:
            Precision to store the latent representations.

        Returns
        ----------
        z:
            Latent space features (dim: n_obs x n_active_gps or n_obs x n_gps).
        mu:
            Expected values of the latent posterior (dim: n_obs x n_active_gps
            or n_obs x n_gps).
        std:
            Standard deviations of the latent posterior (dim: n_obs x
            n_active_gps or n_obs x n_gps).
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            adj_key=adj_key,
            used_pca_feat=self.used_pca_feat_,
            edge_val_ratio=0.0,
            edge_test_ratio=0.0,
            node_val_ratio=0.0,
            node_test_ratio=0.0,
        )
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False,
        )
        node_loader = loader_dict["node_train_loader"]

        # Initialize latent vectors
        if return_mu_std:
            mu = np.empty(
                shape=(adata.shape[0], self.bottle_neck_neurons_),
                dtype=dtype,
            )
            std = np.empty(
                shape=(adata.shape[0], self.bottle_neck_neurons_),
                dtype=dtype,
            )
        else:
            z = np.empty(shape=(adata.shape[0], self.bottle_neck_neurons_), dtype=dtype)

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device)
            if return_mu_std:
                mu_batch, std_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    augment_type=self.augment_type_,
                    return_mu_std=True,
                )
                mu[n_obs_before_batch:n_obs_after_batch, :] = (
                    mu_batch.detach().cpu().numpy()
                )
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch.detach().cpu().numpy()
                )
            else:
                z_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    augment_type=self.augment_type_,
                    return_mu_std=False,
                )
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch.detach().cpu().numpy()
                )
        if return_mu_std:
            return mu, std
        else:
            return z

    # Loss curve
    def plot_loss_curves(self, title="Losses Curve"):
        return self.trainer.plot_loss_curves(title=title)

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            # Preprocessing options
            "adata_list": dct["adata_list_"],
            "profile": dct["profile_"],
            "data_type": dct["data_type_"],
            "sub_data_type": dct["sub_data_type_"],
            "sample_col": dct["sample_col_"],
            "weight": dct["weight_"],
            "graph_const_method": dct["graph_const_method_"],
            "genome": dct["genome_"],
            "use_gene_weight": dct["use_gene_weight_"],
            "user_cache_path": dct["user_cache_path_"],
            "use_top_pcs": dct["use_top_pcs_"],
            "used_hvg": dct["used_hvg_"],
            "min_features": dct["min_features_"],
            "min_cells": dct["min_cells_"],
            "keep_mt": dct["keep_mt_"],
            "target_sum": dct["target_sum_"],
            "rna_n_top_features": dct["rna_n_top_features_"],
            "atac_n_top_features": dct["atac_n_top_features_"],
            "n_components": dct["n_components_"],
            "n_neighbors": dct["n_neighbors_"],
            "metric": dct["metric_"],
            "svd_solver": dct["svd_solver_"],
            "used_pca_feat": dct["used_pca_feat_"],
            "adj_key": dct["adj_key_"],
            # data split parameters
            "edge_val_ratio": dct["edge_val_ratio_"],
            "edge_test_ratio": dct["edge_test_ratio_"],
            "node_val_ratio": dct["node_val_ratio_"],
            "node_test_ratio": dct["node_test_ratio_"],
            # model parameters
            "augment_type": dct["augment_type_"],
            "svd_q": dct["svd_q_"],
            "use_FCencoder": dct["use_FCencoder_"],
            "gnn_layer": dct["gnn_layer_"],
            "conv_type": dct["conv_type_"],
            "hidden_dims": dct["hidden_dims_"],
            "bottle_neck_neurons": dct["bottle_neck_neurons_"],
            "cluster_num": dct["cluster_num_"],
            "num_heads": dct["num_heads_"],
            "dropout": dct["dropout_"],
            "concat": dct["concat_"],
            "drop_feature_rate": dct["drop_feature_rate_"],
            "drop_edge_rate": dct["drop_edge_rate_"],
            "used_edge_weight": dct["used_edge_weight_"],
            "used_DSBN": dct["used_DSBN_"],
            "used_mmd": dct["used_mmd_"],
            # data loader parameters
            "num_neighbors": dct["num_neighbors_"],
            "loaders_n_hops": dct["loaders_n_hops_"],
            "edge_batch_size": dct["edge_batch_size_"],
            "node_batch_size": dct["node_batch_size_"],
            # loss parameters
            "include_edge_recon_loss": dct["include_edge_recon_loss_"],
            "include_gene_expr_recon_loss": dct["include_gene_expr_recon_loss_"],
            "lambda_latent_contrastive_instanceloss": dct["lambda_latent_contrastive_instanceloss_"],
            "lambda_latent_contrastive_clusterloss": dct["lambda_latent_contrastive_clusterloss_"],
            "lambda_gene_expr_recon": dct["lambda_gene_expr_recon_"],
            "lambda_latent_adj_recon_loss": dct["lambda_latent_adj_recon_loss_"],
            "lambda_edge_recon": dct["lambda_edge_recon_"],
            "lambda_omics_recon_mmd_loss": dct["lambda_omics_recon_mmd_loss_"],
            # train parameters
            "n_epochs": dct["n_epochs_"],
            "n_epochs_no_edge_recon": dct["n_epochs_no_edge_recon_"],
            "learning_rate": dct["learning_rate_"],
            "weight_decay": dct["weight_decay_"],
            "gradient_clipping": dct["gradient_clipping_"],
            # other parameters
            "latent_key": dct["latent_key_"],
            "reload_best_model": dct["reload_best_model_"],
            "use_early_stopping": dct["use_early_stopping_"],
            "early_stopping_kwargs": dct["early_stopping_kwargs_"],
            "monitor": dct["monitor_"],
            "device_id": dct["device_id_"],
            "seed": dct["seed_"],
            "verbose": dct["verbose_"],
            "log_style": dct["log_style_"],
            "use_lightning": dct["use_lightning_"],
            "lightning_sampling_mode": dct["lightning_sampling_mode_"],
        }

        return init_params

    def label_transfer(
        self,
        ref_adata,
        ref_adata_emb,
        query_adata,
        query_adata_emb,
        ref_adata_obs,
        label_keys,
        n_neighbors=50,
        threshold=1,
        pred_unknown=False,
        mode="package",
    ):
        knn_transformer = weighted_knn_trainer(
            train_adata=ref_adata,
            train_adata_emb=ref_adata_emb,  # location of our joint embedding
            n_neighbors=n_neighbors,
        )

        labels, uncert = weighted_knn_transfer(
            query_adata=query_adata,
            query_adata_emb=query_adata_emb,  # location of our embedding, query_adata.X in this case
            label_keys=label_keys,  # (start of) obs column name(s) for which to transfer labels
            knn_model=knn_transformer,
            ref_adata_obs=ref_adata_obs,
            threshold=threshold,
            pred_unknown=pred_unknown,
            mode=mode,
        )
        # 定义列名的映射
        cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
        if pred_unknown:
            rename_mapping_labels = {col: f"transferred_{col}_filtered" for col in cols}
        else:
            rename_mapping_labels = {
                col: f"transferred_{col}_unfiltered" for col in cols
            }

        # 定义 uncertainty 映射
        rename_mapping_uncert = {col: f"transferred_{col}_uncert" for col in cols}

        # 重命名列并加入到 'query_adata.obs'
        query_adata.obs = query_adata.obs.join(
            labels.rename(columns=rename_mapping_labels)
        )

        # 重命名列并加入到 'query_adata.obs'
        query_adata.obs = query_adata.obs.join(
            uncert.rename(columns=rename_mapping_uncert)
        )
        ## 去除 query_adata obs 中 NA 的列
        query_adata.obs = query_adata.obs.dropna(axis=1, how="all")

        return query_adata