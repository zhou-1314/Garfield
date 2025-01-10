"""
This module contains the encoder used by the Garfield model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from .utils import DSBatchNorm, drop_feature


class Projection(nn.Module):
    def __init__(self, in_dim: int, encoder_dim: int):
        """
        Initializes the Projection layer.

        Parameters
        ----------
        in_dim : int
            The dimension of the input features.
        encoder_dim : int
            The dimension of the encoded features.
        """
        super().__init__()
        self.layer = nn.Linear(in_dim, encoder_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the Projection layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, encoder_dim) after applying a linear transformation and ReLU activation.
        """
        return self.relu(self.layer(x))


class GATEncoder(nn.Module):
    """
    The GATEncoder class implements a Graph Attention Network (GAT) encoder with multiple layers,
    normalization, and optional fully connected (FC) encoder. It supports different types of GAT
    convolutions and augmentations for omics and graph data.

    Methods
    ----------
    forward(data, decoder_type, augment_type)
        Performs the forward pass through the GAT encoder, applying either omics or graph decoding,
        with optional augmentation.
    _forward_through_layers(x, edge_index, edge_weight, y)
        Helper function to pass the input features through multiple GAT layers and apply normalization.

    Parameters
    ----------
    in_channels : int
        Number of input feature dimensions (length of each node's feature vector).
    hidden_dims : list[int]
        List of output dimensions for each hidden layer in the GAT.
    latent_dim : int
        Dimension of the latent feature representation produced by the encoder.
    conv_type : str
        Type of GAT convolution to use ('GAT' or 'GATv2Conv').
    use_FCencoder : bool
        Whether to use an additional fully connected encoder before the GAT layers.
    drop_feature_rate : float
        Dropout rate for node features during augmentation.
    drop_edge_rate : float
        Dropout rate for edges during augmentation.
    svd_q : int
        Rank for the low-rank SVD approximation used in augmentations. Default is 5.
    num_heads : int
        Number of attention heads for each GAT layer.
    dropout : float
        Dropout rate for GAT layers.
    concat : bool
        Whether to concatenate the outputs of all attention heads.
    num_domains : int, optional
        Number of domains for domain-specific batch normalization (DSBN). If `1`, regular batch normalization is used. Default is 1.
    used_edge_weight : bool, optional
        Whether to use edge weights in the GAT layers. Default is False.
    used_DSBN : bool, optional
        Whether to use domain-specific batch normalization (DSBN). Default is False.
    """

    def __init__(
        self,
        in_channels,
        hidden_dims,
        latent_dim,
        conv_type,
        use_FCencoder,
        drop_feature_rate,
        drop_edge_rate,
        svd_q,
        num_heads,
        dropout,
        concat,
        num_domains=1,
        used_edge_weight=False,
        used_DSBN=False,
    ):
        """
        Initializes the GATEncoder with multiple Graph Attention Network (GAT) layers, normalization layers,
        and optional fully connected (FC) encoder.
        """
        super(GATEncoder, self).__init__()
        self.use_FCencoder = use_FCencoder
        self.drop_feature_rate = drop_feature_rate
        self.drop_edge_rate = drop_edge_rate
        self.svd_q = svd_q
        self.used_DSBN = used_DSBN
        self.used_edge_weight = used_edge_weight
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Choose GAT layer type based on `conv_type`
        GATLayer = GATConv if conv_type == "GAT" else GATv2Conv

        # Initialize normalization layers based on `num_domains`
        for hidden_dim in hidden_dims:
            current_dim = hidden_dim * num_heads if concat else hidden_dim
            if num_domains == 1:
                norm_layer = nn.BatchNorm1d(current_dim)
            else:
                norm_layer = DSBatchNorm(current_dim, num_domains)
            self.norm_layers.append(norm_layer)

        # Initialize projection layer if `use_FCencoder` is True
        if self.use_FCencoder:
            encoder_dim = hidden_dims[0] * 2
            self.proj = Projection(in_channels, encoder_dim)
            current_dim = encoder_dim
        else:
            current_dim = in_channels

        # Initialize GAT layers
        for hidden_dim in hidden_dims:
            layer = GATLayer(
                in_channels=current_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat,
                edge_dim=1 if self.used_edge_weight else None,
            )
            self.layers.append(layer)
            current_dim = hidden_dim * num_heads if concat else hidden_dim

        # Initialize the final mean and log standard deviation layers
        self.conv_mean = GATLayer(
            in_channels=current_dim,
            out_channels=latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1 if self.used_edge_weight else None,
            dropout=dropout,
        )
        self.conv_log_std = GATLayer(
            in_channels=current_dim,
            out_channels=latent_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1 if self.used_edge_weight else None,
            dropout=dropout,
        )

    def forward(self, data, decoder_type, augment_type):
        """
        Performs the forward pass through the GAT encoder, applying either omics or graph decoding, with optional augmentation.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object containing node features, edge index, and other graph-related information.
        decoder_type : str
            Specifies the type of decoder to use, either 'omics' or 'graph'.
        augment_type : str, optional
            Specifies the type of augmentation to apply, either 'dropout' or 'svd'.
            If None, no augmentation is applied.

        Returns
        -------
        Tuple of torch.Tensor
            For 'omics' decoder type: (z_mean1, z_log_std1, z_mean2, z_log_std2), where:
            - z_mean1: Mean of the latent space from original input.
            - z_log_std1: Log standard deviation of the latent space from original input.
            - z_mean2: Mean of the latent space from augmented input.
            - z_log_std2: Log standard deviation of the latent space from augmented input.

            For 'graph' decoder type: (z_mean1, z_log_std1), where:
            - z_mean1: Mean of the latent space.
            - z_log_std1: Log standard deviation of the latent space.
        """
        x, edge_index_all, y = data.x, data.edge_index, data.y
        edge_index = edge_index_all[:, :2]

        if decoder_type == "omics":
            if augment_type is not None and augment_type == "dropout":
                edge_weight = edge_index_all[:, 2] if self.used_edge_weight else None
                x_aug = drop_feature(x=x, drop_prob=self.drop_feature_rate)
                edge_index_aug = dropout_adj(edge_index, p=self.drop_edge_rate)[0]
                edge_weight_aug = edge_weight
            elif augment_type is not None and augment_type == "svd":
                edge_weight = edge_index_all[:, 2].float()
                num_nodes = int(edge_index.max().item()) + 1
                sparse_adj = torch.sparse_coo_tensor(
                    indices=edge_index.t(),
                    values=edge_weight,
                    size=(num_nodes, num_nodes),
                )
                q = min(self.svd_q, sparse_adj.shape[1])  # 确保 q <= 矩阵的列数
                u, s, v = torch.svd_lowrank(sparse_adj, q=q)
                recon_adj = (u @ torch.diag(s)) @ v.T
                sparse_adj = recon_adj.to_sparse()
                edge_index_aug = sparse_adj.indices()
                edge_weight_aug = sparse_adj.values().unsqueeze(1)
                x_aug = drop_feature(x=x, drop_prob=self.drop_feature_rate)
                # x_aug = data.x
            else:
                raise NotImplementedError(f"Unknown augment type: {augment_type}")

            if self.use_FCencoder:
                x = self.proj(x)
                x_aug = self.proj(x_aug)

            if not self.used_edge_weight and augment_type == "svd":
                edge_weight = None
                edge_weight_aug = None
            z_mean1, z_log_std1 = self._forward_through_layers(
                x, edge_index, edge_weight, y
            )
            z_mean2, z_log_std2 = self._forward_through_layers(
                x_aug, edge_index_aug, edge_weight_aug, y
            )

            return z_mean1, z_log_std1, z_mean2, z_log_std2

        elif decoder_type == "graph":
            edge_weight = (
                torch.ones(edge_index.shape[1]).unsqueeze(1).to(edge_index.device)
                if self.used_edge_weight
                else None
            )

            if self.use_FCencoder:
                x = self.proj(x)

            z_mean1, z_log_std1 = self._forward_through_layers(
                x, edge_index, edge_weight, y
            )
            return z_mean1, z_log_std1

        else:
            raise NotImplementedError(f"Unknown decoder type: {decoder_type}")

    def _forward_through_layers(self, x, edge_index, edge_weight, y):
        """
        Helper function to pass the input features through multiple GAT layers and apply normalization.

        Parameters
        ----------
        x : torch.Tensor
            Node features (shape: [num_nodes, feature_dim]).
        edge_index : torch.Tensor
            Edge indices (shape: [2, num_edges]).
        edge_weight : torch.Tensor or None
            Edge weights for weighted GAT layers (optional, shape: [num_edges]).
        y : torch.Tensor
            Domain labels used for domain-specific batch normalization (DSBN) (shape: [num_nodes]).

        Returns
        -------
        Tuple of torch.Tensor
            - z_mean: Mean of the latent space after passing through the GAT layers (shape: [num_nodes, latent_dim]).
            - z_log_std: Log standard deviation of the latent space (shape: [num_nodes, latent_dim]).
        """
        for idx, layer in enumerate(self.layers):
            x, _ = layer(
                x.float(),
                edge_index,
                edge_attr=edge_weight.float() if self.used_edge_weight else None,
                return_attention_weights=True,
            )
            if self.used_DSBN and len(x) > 1:
                norm_layer = self.norm_layers[idx]
                if isinstance(norm_layer, DSBatchNorm):
                    x = norm_layer(x, y)
                else:
                    x = norm_layer(x)
            x = F.relu(x)

        z_mean, _ = self.conv_mean(
            x.float(),
            edge_index,
            edge_attr=edge_weight.float() if self.used_edge_weight else None,
            return_attention_weights=True,
        )
        z_log_std, _ = self.conv_log_std(
            x.float(),
            edge_index,
            edge_attr=edge_weight.float() if self.used_edge_weight else None,
            return_attention_weights=True,
        )

        return z_mean, z_log_std


### GCN encoder
class GCNEncoder(nn.Module):
    """
    The GCNEncoder class implements a Graph Convolutional Network (GCN) encoder with multiple layers,
    normalization, and optional fully connected (FC) encoder. It supports different types of augmentations
    for omics and graph data.

    Methods
    ----------
    forward(data, decoder_type, augment_type)
        Performs the forward pass through the GCN encoder, applying either omics or graph decoding,
        with optional augmentation.
    _forward_through_layers(x, edge_index, edge_weight, y)
        Helper function to pass the input features through multiple GCN layers and apply normalization.
    _apply_normalization(x, y, idx)
        Applies batch normalization or domain-specific batch normalization (DSBN) based on the model's configuration.

    Parameters
    ----------
    in_channels : int
        Number of input feature dimensions (length of each node's feature vector).
    hidden_dims : list[int]
        List of output dimensions for each hidden layer in the GCN.
    latent_dim : int
        Dimension of the latent feature representation produced by the encoder.
    use_FCencoder : bool
        Whether to use a fully connected encoder (FC encoder) before the GCN layers.
    drop_feature_rate : float
        Dropout rate for node features during augmentation.
    drop_edge_rate : float
        Dropout rate for edges during augmentation.
    svd_q : int
        Rank for the low-rank SVD approximation used in augmentations.
    dropout : float, optional
        Dropout rate applied to GCN layers, default is 0.2.
    num_domains : int, optional
        Number of domains for domain-specific batch normalization (DSBN). If `1`, regular batch normalization is used. Default is 1.
    used_edge_weight : bool, optional
        Whether to use edge weights in the GCN layers. Default is False.
    used_DSBN : bool, optional
        Whether to use domain-specific batch normalization (DSBN). Default is False.
    """

    def __init__(
        self,
        in_channels,
        hidden_dims,
        latent_dim,
        use_FCencoder,
        drop_feature_rate,
        drop_edge_rate,
        svd_q,
        dropout=0.2,
        num_domains=1,
        used_edge_weight=False,
        used_DSBN=False,
    ):
        """
        Initializes the GCNEncoder with configurable options for feature projection, dropout, and domain-specific batch normalization (DSBN).
        """
        super(GCNEncoder, self).__init__()

        self.use_FCencoder = use_FCencoder
        self.drop_feature_rate = drop_feature_rate
        self.drop_edge_rate = drop_edge_rate
        self.svd_q = svd_q
        self.used_DSBN = used_DSBN
        self.used_edge_weight = used_edge_weight
        self.norm = nn.ModuleList()

        # Apply feature projection if specified
        if self.use_FCencoder:
            encoder_dim = hidden_dims[0] * 2
            self.proj = Projection(in_channels, encoder_dim)
            current_dim = encoder_dim
        else:
            current_dim = in_channels

        # Initialize GCN layers
        self.gcn_layers = nn.ModuleList()
        total_layers = [current_dim] + hidden_dims
        for i in range(len(total_layers) - 1):
            self.gcn_layers.append(
                GCNConv(total_layers[i], total_layers[i + 1], dropout=dropout)
            )

        # Initialize mean and log standard deviation layers
        self.gcn_mu = GCNConv(hidden_dims[-1], latent_dim, dropout=dropout)
        self.gcn_logvar = GCNConv(hidden_dims[-1], latent_dim, dropout=dropout)

        # Initialize normalization layers
        for current_dim in hidden_dims:
            if num_domains == 1:
                norm = nn.BatchNorm1d(current_dim)
            else:
                norm = DSBatchNorm(current_dim, num_domains)
            self.norm.append(norm)

    def forward(self, data, decoder_type, augment_type=None):
        """
        Forward pass through the GCN encoder, with optional augmentations such as dropout or SVD, and multiple decoding options.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object containing node features, edge index, and other graph-related information.
        decoder_type : str
            Specifies the type of decoder to use, either 'omics' for gene expression data or 'graph' for graph structure.
        augment_type : str, optional
            Specifies the type of augmentation to apply, either 'dropout' or 'svd'. If None, no augmentation is applied.

        Returns
        -------
        Tuple of torch.Tensor
            For 'omics' decoder type: (z_mean1, z_log_std1, z_mean2, z_log_std2), where:
            - z_mean1: Mean of the latent space from original input.
            - z_log_std1: Log standard deviation of the latent space from original input.
            - z_mean2: Mean of the latent space from augmented input.
            - z_log_std2: Log standard deviation of the latent space from augmented input.

            For 'graph' decoder type: (z_mean1, z_log_std1), where:
            - z_mean1: Mean of the latent space.
            - z_log_std1: Log standard deviation of the latent space.
        """
        x, edge_index_all, y = data.x, data.edge_index, data.y
        edge_index = edge_index_all[:, :2]

        if decoder_type == "omics":
            if augment_type is not None and augment_type == "dropout":
                edge_weight = edge_index_all[:, 2] if self.used_edge_weight else None
                x_aug = drop_feature(x=x, drop_prob=self.drop_feature_rate)
                edge_index_aug = dropout_adj(edge_index, p=self.drop_edge_rate)[0]
                edge_weight_aug = edge_weight
            elif augment_type is not None and augment_type == "svd":
                edge_weight = edge_index_all[:, 2].float()
                num_nodes = int(edge_index.max().item()) + 1
                sparse_adj = torch.sparse_coo_tensor(
                    indices=edge_index.t(),
                    values=edge_weight,
                    size=(num_nodes, num_nodes),
                )
                q = min(self.svd_q, sparse_adj.shape[1])  # 确保 q <= 矩阵的列数
                u, s, v = torch.svd_lowrank(sparse_adj, q=q)
                recon_adj = (u @ torch.diag(s)) @ v.T
                sparse_adj = recon_adj.to_sparse()
                edge_index_aug = sparse_adj.indices()
                edge_weight_aug = sparse_adj.values().unsqueeze(1)
                # x_aug = drop_feature(x=x, drop_prob=self.drop_feature_rate)
                x_aug = data.x
            else:
                raise NotImplementedError(f"Unknown augment type: {augment_type}")

            if self.use_FCencoder:
                x = self.proj(x)
                x_aug = self.proj(x_aug)

            if not self.used_edge_weight and augment_type == "svd":
                edge_weight = None
                edge_weight_aug = None
            z_mean1, z_log_std1 = self._forward_through_layers(
                x, edge_index, edge_weight, y
            )
            z_mean2, z_log_std2 = self._forward_through_layers(
                x_aug, edge_index_aug, edge_weight_aug, y
            )

            return z_mean1, z_log_std1, z_mean2, z_log_std2

        elif decoder_type == "graph":
            edge_weight = (
                torch.ones(edge_index.shape[1]).unsqueeze(1).to(edge_index.device)
                if self.used_edge_weight
                else None
            )

            if self.use_FCencoder:
                x = self.proj(x)

            z_mean1, z_log_std1 = self._forward_through_layers(
                x, edge_index, edge_weight, y
            )
            return z_mean1, z_log_std1

        else:
            raise NotImplementedError(f"Unknown decoder type: {decoder_type}")

    def _forward_through_layers(self, x, edge_index, edge_weight, y):
        """
        Helper function that passes the node features through GCN layers and applies normalization.

        Parameters
        ----------
        x : torch.Tensor
            Node features (shape: [num_nodes, feature_dim]).
        edge_index : torch.Tensor
            Edge indices (shape: [2, num_edges]).
        edge_weight : torch.Tensor or None
            Edge weights for weighted GCN layers (optional, shape: [num_edges]).
        y : torch.Tensor
            Domain labels used for domain-specific batch normalization (DSBN) (shape: [num_nodes]).

        Returns
        -------
        Tuple of torch.Tensor
            - z_mean: Mean of the latent space after passing through the GCN layers (shape: [num_nodes, latent_dim]).
            - z_log_std: Log standard deviation of the latent space (shape: [num_nodes, latent_dim]).
        """
        for idx, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self._apply_normalization(x, y, idx)
            x = F.relu(x)

        z_mean = self.gcn_mu(x, edge_index, edge_weight=edge_weight)
        z_log_std = self.gcn_logvar(x, edge_index, edge_weight=edge_weight)
        return z_mean, z_log_std

    def _apply_normalization(self, x, y, idx):
        """
        Applies batch normalization or domain-specific batch normalization (DSBN) based on the model's configuration.

        Parameters
        ----------
        x : torch.Tensor
            Node features (shape: [num_nodes, feature_dim]).
        y : torch.Tensor
            Domain labels for DSBN (shape: [num_nodes]).
        idx : int
            Index of the current layer for normalization.

        Returns
        -------
        torch.Tensor
            Normalized node features after applying either batch normalization or DSBN (shape: [num_nodes, feature_dim]).
        """
        if self.used_DSBN and len(x) > 1:
            norm_layer = self.norm[idx]
            if isinstance(norm_layer, DSBatchNorm):
                x = norm_layer(x, y)
            else:
                x = norm_layer(x)
        return x
