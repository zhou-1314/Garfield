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
from ..trainer.trainer import GarfieldTrainer

class Garfield(torch.nn.Module, BaseModelMixin):
    """
    Garfield: Graph-based Contrastive Learning enable Fast Single-Cell Embedding
    """
    def __init__(self, gf_params):
        """"
        Garfield Model Parameters, which are included in gf_params.

        Parameters
        ----------
        adata_list : list
            List of AnnData objects containing data from multiple batches or samples.
        profile : str
            Specifies the data profile type (e.g., 'RNA', 'ATAC', 'ADT', 'multi-modal', 'spatial').
        data_type : str
            Type of the multi-omics dataset (e.g., Paired, UnPaired) for preprocessing.
        sub_data_type : list[str]
            List of data types for multi-modal datasets (e.g., ['rna', 'atac'] or ['rna', 'adt']).
        sample_col : str
            Column in the dataset that indicates batch or sample identifiers (default: 'batch').
        weight : float or None
            Weighting factor that determines the contribution of different modalities or types of graphs in multi-omics or spatial data.
            - For non-spatial single-cell multi-omics data (e.g., RNA + ATAC),
            `weight` specifies the contribution of the graph constructed from RNA data.
            The remaining (1 - weight) represents the contribution from the other modality.
            - For spatial single-modality data,
            `weight` refers to the contribution of the graph constructed from the physical spatial information,
            while (1 - weight) reflects the contribution from the molecular graph.
        graph_const_method : str
            Method for constructing the graph (e.g., 'mu_std', 'Radius', 'KNN', 'Squidpy').
        genome : str
            Reference genome to use during preprocessing (e.g., 'mm10', 'mm9', 'hg38', 'hg19').
        use_gene_weight : bool
            Whether to apply gene weights in the preprocessing step.
        use_top_pcs : bool
            Whether to use the top principal components during gene score preprocessing step.
        used_hvg : bool
            Whether to use highly variable genes (HVGs) for analysis.
        min_features : int
            Minimum number of features required for a cell to be included in the dataset.
        min_cells : int
            Minimum number of cells required for a feature to be retained in the dataset.
        keep_mt : bool
            Whether to retain mitochondrial genes in the analysis.
        target_sum : float
            Target sum used for normalization (e.g., 1e4 for counts per cell).
        rna_n_top_features : int
            Number of top features to retain for RNA datasets (e.g., 3000).
        atac_n_top_features : int
            Number of top features to retain for ATAC datasets (e.g., 10000).
        n_components : int
            Number of components to use for dimensionality reduction (e.g., PCA).
        n_neighbors : int
            Number of neighbors to use in graph-based algorithms (e.g., KNN).
        metric : str
            Distance metric used during graph construction (e.g., 'correlation', 'euclidean').
        svd_solver : str
            Solver for singular value decomposition (SVD), such as 'arpack' or 'randomized'.
        adj_key : str
            Key in the AnnData object that holds the adjacency matrix.
        edge_val_ratio : float
            Ratio of edges to use for validation in edge-level tasks.
        edge_test_ratio : float
            Ratio of edges to use for testing in edge-level tasks.
        node_val_ratio : float
            Ratio of nodes to use for validation in node-level tasks.
        node_test_ratio : float
            Ratio of nodes to use for testing in node-level tasks.
        augment_type : str
            Type of augmentation to use (e.g., 'dropout', 'svd').
        svd_q : int
            Rank for the low-rank SVD approximation.
        use_FCencoder : bool
            Whether to use a fully connected encoder before the graph layers.
        hidden_dims : list[int]
            List of hidden layer dimensions for the encoder.
        bottle_neck_neurons : int
            Number of neurons in the bottleneck (latent) layer.
        num_heads : int
            Number of attention heads for each graph attention layer.
        dropout : float
            Dropout rate applied during training.
        concat : bool
            Whether to concatenate attention heads or not.
        drop_feature_rate : float
            Dropout rate applied to node features.
        drop_edge_rate : float
            Dropout rate applied to edges during augmentation.
        used_edge_weight : bool
            Whether to use edge weights in the graph layers.
        used_DSBN : bool
            Whether to use domain-specific batch normalization.
        conv_type : str
            Type of graph convolution to use ('GATv2Conv', 'GAT', 'GCN').
        gnn_layer : int
            Number of times the encoder is repeated in the forward pass, not the number of GNN layers.
        cluster_num : int
            Number of clusters for latent feature clustering.
        num_neighbors : int
            Number of neighbors to sample for graph-based data loaders.
        loaders_n_hops : int
            Number of hops for neighbors during graph construction.
        edge_batch_size : int
            Batch size for edge-level tasks.
        node_batch_size : int
            Batch size for node-level tasks.
        include_edge_recon_loss : bool
            Whether to include edge reconstruction loss in the training objective.
        include_gene_expr_recon_loss : bool
            Whether to include gene expression reconstruction loss in the training objective.
        used_mmd : bool
            Whether to use maximum mean discrepancy (MMD) for domain adaptation.
        lambda_latent_contrastive_instanceloss : float
            Weight for the instance-level contrastive loss.
        lambda_latent_contrastive_clusterloss : float
            Weight for the cluster-level contrastive loss.
        lambda_gene_expr_recon : float
            Weight for the gene expression reconstruction loss.
        lambda_edge_recon : float
            Weight for the edge reconstruction loss.
        lambda_omics_recon_mmd_loss : float
            Weight for the MMD loss in omics reconstruction tasks.
        n_epochs : int
            Number of training epochs.
        n_epochs_no_edge_recon : int
            Number of epochs without edge reconstruction loss.
        learning_rate : float
            Learning rate for the optimizer.
        weight_decay : float
            Weight decay (L2 regularization) for the optimizer.
        gradient_clipping : float
            Maximum norm for gradient clipping.
        latent_key : str
            Key for storing latent features in the AnnData object.
        reload_best_model : bool
            Whether to reload the best model after training.
        use_early_stopping : bool
            Whether to use early stopping during training.
        early_stopping_kwargs : dict
            Arguments for configuring early stopping (e.g., patience, delta).
        monitor : bool
            Whether to print training progress.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Whether to display detailed logs during training.
        """
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
        self.lambda_latent_contrastive_instanceloss_ = self.args.lambda_latent_contrastive_instanceloss
        self.lambda_latent_contrastive_clusterloss_ = self.args.lambda_latent_contrastive_clusterloss
        self.lambda_gene_expr_recon_ = self.args.lambda_gene_expr_recon
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
        self.verbose_ = self.args.verbose

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
            use_gene_weigt=self.use_gene_weight_,
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
            svd_solver=self.svd_solver_
        )
        # set up model
        if not self.used_pca_feat_:
            self.num_features_ = self.adata.n_vars
        else:
            self.num_features_ = self.adata.obsm['feat'].shape[1]

        if self.sample_col_ is not None:
            self.n_domain_ = len(self.adata.obs[self.sample_col_].unique())
        else:
            self.n_domain_ = None
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        ## 选择 encoder
        ## 设定参数
        if self.conv_type_ in ['GAT', 'GATv2Conv']:
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
                used_DSBN=self.used_DSBN_
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
                used_DSBN=self.used_DSBN_
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
            used_mmd=self.used_mmd_
        )
        self.is_trained_ = False

    def train(self, **trainer_kwargs):
        self.trainer = GarfieldTrainer(
            adata=self.adata,
            model=self.model,
            label_name=self.sample_col_,
            used_pca_feat=self.used_pca_feat_,
            adj_key=self.adj_key_,
            # data split
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=self.edge_test_ratio_,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=self.node_test_ratio_,
            # data process
            augment_type=self.augment_type_,
            # data loader
            num_neighbors=self.num_neighbors_,
            loaders_n_hops=self.loaders_n_hops_,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            # other parameters
            reload_best_model=self.reload_best_model_,
            use_early_stopping=self.use_early_stopping_,
            early_stopping_kwargs=self.early_stopping_kwargs_,
            monitor=self.monitor_,
            verbose=self.verbose_,
            seed=self.seed_,
            **trainer_kwargs)

        self.trainer.train(
            n_epochs=self.n_epochs_,
            n_epochs_no_edge_recon=self.lambda_edge_recon_,  # : int=0
            learning_rate=self.learning_rate_,
            weight_decay=self.weight_decay_,
            gradient_clipping=self.gradient_clipping_,
            lambda_edge_recon=self.lambda_edge_recon_,
            lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
            lambda_latent_contrastive_instanceloss=self.lambda_latent_contrastive_instanceloss_,
            lambda_latent_contrastive_clusterloss=self.lambda_latent_contrastive_clusterloss_,
            lambda_omics_recon_mmd_loss=self.lambda_omics_recon_mmd_loss_
        )

        self.node_batch_size_ = self.trainer.node_batch_size_

        self.is_trained_ = True
        self.model.eval()

        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
            adata=self.adata,
            adj_key=self.adj_key_,
            return_mu_std=True,
            node_batch_size=self.node_batch_size_
        )

    # embedding
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            adj_key: str = "spatial_connectivities",
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
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False)
        node_loader = loader_dict["node_train_loader"]

        # Initialize latent vectors
        if return_mu_std:
            mu = np.empty(shape=(adata.shape[0], self.bottle_neck_neurons_), dtype=dtype)
            std = np.empty(shape=(adata.shape[0], self.bottle_neck_neurons_), dtype=dtype)
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
                    return_mu_std=True)
                mu[n_obs_before_batch:n_obs_after_batch, :] = (
                    mu_batch.detach().cpu().numpy())
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch.detach().cpu().numpy())
            else:
                z_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    augment_type=self.augment_type_,
                    return_mu_std=False)
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch.detach().cpu().numpy())
        if return_mu_std:
            return mu, std
        else:
            return z

    # Loss curve
    def plot_loss_curves(self,
                         title="Losses Curve"
                         ):
        return self.trainer.plot_loss_curves(title=title)

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            # Preprocessing options
            'adata_list': dct['adata_list_'],
            'profile': dct['profile_'],
            'data_type': dct['data_type_'],
            'sub_data_type': dct['sub_data_type_'],
            'sample_col': dct['sample_col_'],
            'weight': dct['weight_'],
            'graph_const_method': dct['graph_const_method_'],
            'genome': dct['genome_'],
            'use_gene_weight': dct['use_gene_weight_'],
            'use_top_pcs': dct['use_top_pcs_'],
            'used_hvg': dct['used_hvg_'],
            'min_features': dct['min_features_'],
            'min_cells': dct['min_cells_'],
            'keep_mt': dct['keep_mt_'],
            'target_sum': dct['target_sum_'],
            'rna_n_top_features': dct['rna_n_top_features_'],
            'atac_n_top_features': dct['atac_n_top_features_'],
            'n_components': dct['n_components_'],
            'n_neighbors': dct['n_neighbors_'],
            'metric': dct['metric_'],
            'svd_solver': dct['svd_solver_'],
            'adj_key': dct['adj_key_'],
            # data split parameters
            'edge_val_ratio': dct['edge_val_ratio_'],
            'edge_test_ratio': dct['edge_test_ratio_'],
            'node_val_ratio': dct['node_val_ratio_'],
            'node_test_ratio': dct['node_test_ratio_'],
            # model parameters
            'augment_type': dct['augment_type_'],
            'svd_q': dct['svd_q_'],
            'use_FCencoder': dct['use_FCencoder_'],
            'gnn_layer': dct['gnn_layer_'],
            'conv_type': dct['conv_type_'],
            'hidden_dims': dct['hidden_dims_'],
            'bottle_neck_neurons': dct['bottle_neck_neurons_'],
            'cluster_num': dct['cluster_num_'],
            'num_heads': dct['num_heads_'],
            'dropout': dct['dropout_'],
            'concat': dct['concat_'],
            'drop_feature_rate': dct['drop_feature_rate_'],
            'drop_edge_rate': dct['drop_edge_rate_'],
            'used_edge_weight': dct['used_edge_weight_'],
            'used_DSBN': dct['used_DSBN_'],
            'used_mmd': dct['used_mmd_'],
            # data loader parameters
            'num_neighbors': dct['num_neighbors_'],
            'loaders_n_hops': dct['loaders_n_hops_'],
            'edge_batch_size': dct['edge_batch_size_'],
            'node_batch_size': dct['node_batch_size_'],
            # loss parameters
            'include_edge_recon_loss': dct['include_edge_recon_loss_'],
            'include_gene_expr_recon_loss': dct['include_gene_expr_recon_loss_'],
            'lambda_latent_contrastive_instanceloss': dct['lambda_latent_contrastive_instanceloss_'],
            'lambda_latent_contrastive_clusterloss': dct['lambda_latent_contrastive_clusterloss_'],
            'lambda_gene_expr_recon': dct['lambda_gene_expr_recon_'],
            'lambda_edge_recon': dct['lambda_edge_recon_'],
            'lambda_omics_recon_mmd_loss': dct['lambda_omics_recon_mmd_loss_'],
            # train parameters
            'n_epochs': dct['n_epochs_'],
            'n_epochs_no_edge_recon': dct['n_epochs_no_edge_recon_'],
            'learning_rate': dct['learning_rate_'],
            'weight_decay': dct['weight_decay_'],
            'gradient_clipping': dct['gradient_clipping_'],
            # other parameters
            'latent_key': dct['latent_key_'],
            'reload_best_model': dct['reload_best_model_'],
            'use_early_stopping': dct['use_early_stopping_'],
            'early_stopping_kwargs': dct['early_stopping_kwargs_'],
            'monitor': dct['monitor_'],
            'seed': dct['seed_'],
            'verbose': dct['verbose_'],
        }

        return init_params

    def label_transfer(self,
                       ref_adata,
                       ref_adata_emb,
                       query_adata,
                       query_adata_emb,
                       ref_adata_obs,
                       label_keys,
                       n_neighbors=50,
                       threshold=1,
                       pred_unknown=False,
                       mode="package"):
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
            mode=mode
        )
        # 定义列名的映射
        cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
        if pred_unknown:
            rename_mapping_labels = {col: f"transferred_{col}_filtered" for col in cols}
        else:
            rename_mapping_labels = {col: f"transferred_{col}_unfiltered" for col in cols}

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
        query_adata.obs = query_adata.obs.dropna(axis=1, how='all')

        return query_adata



