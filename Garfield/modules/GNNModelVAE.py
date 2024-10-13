import numpy as np
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn import GAE, InnerProductDecoder

from .utils import extract_subgraph, to_float_tensor
from ..nn.decoders import GATDecoder, GCNDecoder, CosineSimGraphDecoder
from .loss import (
    compute_omics_recon_mse_loss,
    compute_omics_recon_mmd_loss,
    compute_edge_recon_loss,
    compute_kl_reg_loss,
    compute_contrastive_instanceloss,
    compute_contrastive_clusterloss,
    compute_adj_recon_loss,
)

# Based on VGAE class in PyTorch Geometric
class GNNModelVAE(GAE):
    """
    Garfield model class. This class contains the implementation of GNNModel Variational Auto-encoder.

    Parameters
    ----------
    encoder : nn.Module
        The encoder module used in the variational graph autoencoder. 'GAT' or 'GCN'.
    bottle_neck_neurons : int
        Number of neurons in the bottleneck layer representing the latent dimension.
    hidden_dims : int
        Number of hidden dimensions for the encoder.
    feature_dim : int
        Number of feature dimensions in the input data.
    num_heads : int
        Number of attention heads used in the GAT encoder.
    dropout : float
        Dropout rate used in the encoder and decoder.
    concat : bool
        Whether to concatenate outputs of different attention heads.
    n_domain : int
        Number of domains for domain-specific batch normalization (DSBN).
    used_edge_weight : bool
        Whether to use edge weights in the graph convolution operation.
    used_DSBN : bool
        Whether to use domain-specific batch normalization (DSBN).
    conv_type : str
        Type of graph convolution to use, e.g., 'GAT', 'GATv2Conv', 'GCN'.
    gnn_layer : int, optional
        Number of layers in the GNN encoder. Default is 2.
    cluster_num : int, optional
        Number of clusters for the clustering layer. Default is 20.
    include_edge_recon_loss : bool, optional
        Whether to include edge reconstruction loss in the model. Default is True.
    include_gene_expr_recon_loss : bool, optional
        Whether to include gene expression reconstruction loss in the model. Default is True.
    used_mmd : bool, optional
        Whether to use MMD (Maximum Mean Discrepancy) loss for domain adaptation. Default is False.
    """

    def __init__(
        self,
        encoder,
        bottle_neck_neurons,
        hidden_dims,
        feature_dim,
        num_heads,
        dropout,
        concat,
        n_domain,
        used_edge_weight,
        used_DSBN,
        conv_type,
        gnn_layer=2,
        cluster_num=20,
        include_edge_recon_loss=True,
        include_gene_expr_recon_loss=True,
        used_mmd=False,
    ):
        super(GNNModelVAE, self).__init__(encoder)
        # model configurations
        self.encoder = encoder
        self.latent = bottle_neck_neurons
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.n_domain = n_domain
        self.used_edge_weight = used_edge_weight
        self.used_DSBN = used_DSBN
        self.include_edge_recon_loss = include_edge_recon_loss
        self.include_gene_expr_recon_loss = include_gene_expr_recon_loss
        self.used_mmd = used_mmd
        self.conv_type = conv_type
        self.cluster_num = cluster_num
        assert self.conv_type in [
            "GAT",
            "GATv2Conv",
            "GCN",
        ], 'Convolution must be "GCN", "GAT" or "GATv2Conv".'
        self.gnn_layer = gnn_layer

        # 使用 Xavier 初始化权重
        self.eps_weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.latent, self.latent))
        )
        self.eps_bias = nn.Parameter(torch.zeros(self.latent))
        self.instance_projector = nn.Sequential(
            nn.Linear(self.latent, self.latent),
            nn.LayerNorm(self.latent, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(self.latent, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.latent, self.latent),
            nn.LayerNorm(self.latent, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(self.latent, self.cluster_num),
            nn.Softmax(dim=1),
        )
        # Initialize graph decoder module
        self.graph_decoder = CosineSimGraphDecoder(dropout_rate=self.dropout)
        # Initialize adj decoder module
        self.adj_decoder = InnerProductDecoder()

        ## 重构表达谱
        if self.conv_type in ["GAT", "GATv2Conv"]:
            self.GAT_decoder = GATDecoder(
                in_channels=self.latent,
                hidden_dims=self.hidden_dims,
                out_channels=self.feature_dim,
                conv_type=self.conv_type,
                num_heads=self.num_heads,
                dropout=self.dropout,
                concat=self.concat,
                num_domains=self.n_domain,  # DSBN
                used_edge_weight=self.used_edge_weight,
                used_DSBN=self.used_DSBN,
            )
        elif self.conv_type == "GCN":
            self.GCN_decoder = GCNDecoder(
                in_channels=self.latent,
                hidden_dims=self.hidden_dims,
                out_channels=self.feature_dim,
                dropout=self.dropout,
                num_domains=self.n_domain,  # DSBN
                used_edge_weight=self.used_edge_weight,
                used_DSBN=self.used_DSBN,
            )
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

    def reparameterize(self, mu: Tensor, logstd: Tensor, eps=None) -> Tensor:
        """
        Applies the reparameterization trick to sample a latent vector from the latent distribution during training.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution (shape: [batch_size, latent_dim]).
        logstd : torch.Tensor
            Logarithm of the standard deviation of the latent distribution (shape: [batch_size, latent_dim]).
        eps : torch.Tensor, optional
            Noise tensor used for sampling. If not provided, a standard normal distribution will be used (shape: [batch_size, latent_dim]).

        Returns
        -------
        torch.Tensor
            A reparameterized latent vector sampled from the distribution (shape: [batch_size, latent_dim]).
        """

        if self.training:
            if eps is not None:
                return mu + eps * torch.randn_like(logstd) * torch.exp(logstd)
            else:
                return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, data_batch, decoder_type, augment_type):
        """
        Processes the input data through the encoder to obtain the latent representations and uses the decoder to reconstruct features or edges, depending on the task.

        Parameters
        ----------
        data_batch : Data
            A PyTorch Geometric Data object containing node features, edge information, and any other relevant data.
        decoder_type : str
            Specifies which type of decoder to use, either 'omics' for gene expression data or 'graph' for edge reconstruction tasks.
        augment_type : str
            Specifies the type of data augmentation used during encoding, e.g., 'svd' for singular value decomposition or 'dropout' for regular dropout.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "recon_features" or "edge_recon_logits": Reconstructed features or edge logits, depending on the decoder type.
            - "z": The latent representation of the input data.
            - "mu": The mean of the latent distribution.
            - "logstd": The log standard deviation of the latent distribution.
            - "truth_x": Ground truth input features (for omics tasks).
            - "truth_y": Ground truth labels (for MMD or classification tasks).
        """
        # Get index of sampled nodes for current batch (neighbors of sampled
        # nodes are also part of the batch for message passing layers but
        # should be excluded in backpropagation)
        if decoder_type == "omics":
            # ´data_batch´ will be a node batch and first node_batch_size
            # elements are the sampled nodes, leading to a dim of ´batch_idx´ of
            # ´node_batch_size´
            # batch_idx = slice(None, data_batch.batch_size)

            all_mu1 = []
            all_mu2 = []
            for _ in range(self.gnn_layer):
                data_batch.x = to_float_tensor(data_batch.x)
                encoder_outputs = self.encoder(data_batch, decoder_type, augment_type)
                mu1 = encoder_outputs[0]  # [batch_idx, :]
                mu2 = encoder_outputs[2]  # [batch_idx, :]
                all_mu1.append(mu1)
                all_mu2.append(mu2)

            mean1 = torch.stack(all_mu1).mean(dim=0)  # sum
            logstd1 = torch.matmul(mean1, self.eps_weight) + self.eps_bias
            z1 = self.reparameterize(mean1, logstd1)

            mean2 = torch.stack(all_mu2).mean(dim=0)
            logstd2 = torch.matmul(mean2, self.eps_weight) + self.eps_bias
            z2 = self.reparameterize(mean2, logstd2)

            z_1 = normalize(self.instance_projector(z1), dim=1)
            z_2 = normalize(self.instance_projector(z2), dim=1)

            c_1 = self.cluster_projector(z1)
            c_2 = self.cluster_projector(z2)

            ## 重构邻接矩阵
            pos_adj = self.adj_decoder(z1, data_batch.edge_index, sigmoid=True)
            # Do not include self-loops in negative samples
            pos_edge_index = data_batch.edge_index
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            # negative_sampling
            neg_edge_index = negative_sampling(pos_edge_index, z1.size(0))
            neg_adj = self.adj_decoder(z1, neg_edge_index.long(), sigmoid=True)

            ## 重构表达矩阵
            output = {}
            # with torch.no_grad(): ## TODO
            # data_batch = extract_subgraph(data_batch, batch_idx)
            if self.conv_type in ["GAT", "GATv2Conv"]:
                recon_features = self.GAT_decoder(z1, data_batch)
            else:
                recon_features = self.GCN_decoder(z1, data_batch)
            output["truth_x"] = to_float_tensor(data_batch.x)
            output["truth_y"] = data_batch.y
            output["recon_features"] = to_float_tensor(recon_features)

            output["z"] = z1
            output["z_1"] = z_1
            output["z_2"] = z_2
            output["c_1"] = c_1
            output["c_2"] = c_2
            output["mu"] = mean1
            output["logstd"] = logstd1
            output["pos_adj"] = pos_adj
            output["neg_adj"] = neg_adj
            return output

        elif decoder_type == "graph":
            # ´data_batch´ will be an edge batch with sampled positive and
            # negative edges of size ´edge_batch_size´ respectively. Each edge
            # has a source and target node, leading to a dim of ´batch_idx´ of
            # 4 * ´edge_batch_size´
            batch_idx = torch.cat(
                (data_batch.edge_label_index[0], data_batch.edge_label_index[1]), 0
            )

            all_mu1 = []
            for _ in range(self.gnn_layer):
                data_batch.x = to_float_tensor(data_batch.x)
                encoder_outputs = self.encoder(data_batch, decoder_type, augment_type)
                mu1 = encoder_outputs[0][batch_idx, :]
                all_mu1.append(mu1)

            mean1 = torch.stack(all_mu1).mean(dim=0)  # sum
            logstd1 = torch.matmul(mean1, self.eps_weight) + self.eps_bias
            z1 = self.reparameterize(mean1, logstd1)

            ## 重构表达矩阵
            output = {}
            # Store edge labels in output for loss computation
            output["edge_recon_labels"] = data_batch.edge_label

            # Use decoder to get the edge reconstruction logits
            output["edge_recon_logits"] = self.graph_decoder(z1)

            output["z"] = z1
            output["mu"] = mean1
            output["logstd"] = logstd1
            return output

    def loss(
        self,
        edge_model_output: dict,
        node_model_output: dict,
        lambda_edge_recon,
        lambda_gene_expr_recon,
        lambda_latent_adj_recon_loss,
        lambda_latent_contrastive_instanceloss,
        lambda_latent_contrastive_clusterloss,
        lambda_omics_recon_mmd_loss,
    ) -> dict:
        """
        Computes the total loss for the model by combining different loss components such as KL divergence, edge reconstruction loss, gene expression reconstruction loss, and contrastive losses.

        Parameters
        ----------
        edge_model_output : dict
            A dictionary containing outputs from the forward pass for edge reconstruction, including edge logits and latent variables.
        node_model_output : dict
            A dictionary containing outputs from the forward pass for node reconstruction, including gene expression reconstruction and latent variables.
        lambda_edge_recon : float
            A scaling factor to adjust the contribution of edge reconstruction loss.
        lambda_gene_expr_recon : float
            A scaling factor to adjust the contribution of gene expression reconstruction loss.
        lambda_latent_adj_recon_loss : float
            A scaling factor to adjust the contribution of adjacency reconstruction loss in the latent space.
        lambda_latent_contrastive_instanceloss : float
            A scaling factor to adjust the contribution of instance-level contrastive loss between different latent representations.
        lambda_latent_contrastive_clusterloss : float
            A scaling factor to adjust the contribution of cluster-level contrastive loss between different clusters in the latent space.
        lambda_omics_recon_mmd_loss : float
            A scaling factor to adjust the contribution of Maximum Mean Discrepancy (MMD) loss in omics data reconstruction.

        Returns
        -------
        dict
            A dictionary containing individual loss terms and the total loss:
            - "kl_reg_loss": KL divergence loss between the latent distributions.
            - "edge_recon_loss": Binary cross-entropy loss for edge reconstruction (if applicable).
            - "gene_expr_recon_loss": Mean squared error loss for gene expression reconstruction (if applicable).
            - "lambda_latent_contrastive_instanceloss": Contrastive loss between instance-level latent vectors.
            - "lambda_latent_contrastive_clusterloss": Contrastive loss between clusters in the latent space.
            - "gene_expr_mmd_loss": MMD loss for omics data (if applicable).
            - "global_loss": Sum of all the individual losses used for model optimization.
            - "optim_loss": Sum of the losses used for backpropagation.
        """
        loss_dict = {}

        # 1. Compute Kullback-Leibler divergence loss for edge and node batch
        loss_dict["kl_reg_loss"] = compute_kl_reg_loss(
            mu=node_model_output["mu"], logstd=node_model_output["logstd"]
        )  # * 1 / node_model_output["mu"].size(0)
        loss_dict["kl_reg_loss"] += compute_kl_reg_loss(
            mu=edge_model_output["mu"], logstd=edge_model_output["logstd"]
        )  # * 1 / edge_model_output["mu"].size(0)

        # 2. Compute edge reconstruction binary cross entropy loss for edge batch
        loss_dict["edge_recon_loss"] = (
            (
                lambda_edge_recon
                * compute_edge_recon_loss(
                    edge_recon_logits=edge_model_output["edge_recon_logits"],
                    edge_recon_labels=edge_model_output["edge_recon_labels"],
                )
            )
            * edge_model_output["mu"].size(0)
            / 10
        )

        # 3. Compute gene expression reconstruction with MSE loss for node batch
        loss_dict["gene_expr_recon_loss"] = (
            lambda_gene_expr_recon
            * compute_omics_recon_mse_loss(
                recon_x=node_model_output["recon_features"],
                x=node_model_output["truth_x"],
            )
        ) * 20000  # node_model_output['truth_x'].size(-1)

        # 4. compute reconstructed adj loss through node feedforward
        loss_dict["lambda_latent_adj_recon_loss"] = (
            compute_adj_recon_loss(
                node_model_output["pos_adj"],
                node_model_output["neg_adj"],
                lambda_latent_adj_recon_loss,
            )
            * 100
        )  # * node_model_output['truth_x'].size(-1)

        # 5. compute Contrastive instance losses
        loss_dict[
            "lambda_latent_contrastive_instanceloss"
        ] = compute_contrastive_instanceloss(
            node_model_output["z_1"],
            node_model_output["z_2"],
            lambda_latent_contrastive_instanceloss,
        )
        # 6. compute Contrastive cluster losses
        loss_dict[
            "lambda_latent_contrastive_clusterloss"
        ] = compute_contrastive_clusterloss(
            node_model_output["c_1"],
            node_model_output["c_2"],
            self.cluster_num,
            lambda_latent_contrastive_clusterloss,
        )

        # 7. compute MMD loss
        if self.used_mmd:
            cell_batch = node_model_output["truth_y"]
            device = cell_batch.device
            cell_batch = cell_batch.detach().cpu()
            unique_groups, group_indices = np.unique(cell_batch, return_inverse=True)
            grouped_z_cell = {
                group: node_model_output["z"][group_indices == i]
                for i, group in enumerate(unique_groups)
            }
            group_labels = list(unique_groups)
            num_groups = len(group_labels)

            loss_dict["gene_expr_mmd_loss"] = torch.tensor(0, dtype=torch.float).to(
                device
            )
            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    z_i = grouped_z_cell[group_labels[i]]
                    z_j = grouped_z_cell[group_labels[j]]
                    mmd_loss_tmp = compute_omics_recon_mmd_loss(
                        z_i, z_j
                    ) * node_model_output["z"].size(0)
                    loss_dict["gene_expr_mmd_loss"] += (
                        mmd_loss_tmp * lambda_omics_recon_mmd_loss
                    )

        # Compute optimization loss used for backpropagation as well as global
        # loss used for early stopping of model training and best model saving
        loss_dict["global_loss"] = 0
        loss_dict["optim_loss"] = 0
        loss_dict["global_loss"] += loss_dict["kl_reg_loss"]
        loss_dict["optim_loss"] += loss_dict["kl_reg_loss"]
        if self.include_edge_recon_loss:
            loss_dict["global_loss"] += loss_dict["edge_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["edge_recon_loss"]
        if self.include_gene_expr_recon_loss:
            loss_dict["global_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["global_loss"] += loss_dict["lambda_latent_adj_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["lambda_latent_adj_recon_loss"]
            loss_dict["global_loss"] += loss_dict[
                "lambda_latent_contrastive_instanceloss"
            ]
            loss_dict["optim_loss"] += loss_dict[
                "lambda_latent_contrastive_instanceloss"
            ]
            loss_dict["global_loss"] += loss_dict[
                "lambda_latent_contrastive_clusterloss"
            ]
            loss_dict["optim_loss"] += loss_dict[
                "lambda_latent_contrastive_clusterloss"
            ]
            if self.used_mmd:
                loss_dict["global_loss"] += loss_dict["gene_expr_mmd_loss"]
                loss_dict["optim_loss"] += loss_dict["gene_expr_mmd_loss"]
        return loss_dict

    def get_latent_representation(
        self,
        node_batch: Data,
        augment_type: Literal["svd", "dropout"] = "svd",
        return_mu_std: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes the input data into latent space, either returning the latent features (z) or
        the distribution parameters (mu and std) based on the input option.

        Parameters
        ----------
        node_batch : Data
            A PyTorch Geometric Data object containing features and graph structure for the node-level batch.
        augment_type : str, optional
            Specifies the type of augmentation used in the encoder, e.g., 'svd' (default) or 'dropout'.
        return_mu_std : bool, optional
            If True, the function returns the mean (mu) and standard deviation (std) of the latent distribution.
            Otherwise, it returns the reparameterized latent features (z). Default is False.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            - If `return_mu_std` is False, it returns the reparameterized latent features (z) (shape: [batch_size, latent_dim]).
            - If `return_mu_std` is True, it returns a tuple of the mean (mu) and standard deviation (std) of the latent distribution (each of shape: [batch_size, latent_dim]).
        """

        # Get latent distribution parameters
        encoder_outputs = self.encoder(
            node_batch, augment_type=augment_type, decoder_type="omics"
        )  # z_mean1, z_log_std1, z_mean2, z_log_std2
        mu = encoder_outputs[0][: node_batch.batch_size, :]
        logstd = encoder_outputs[1][: node_batch.batch_size, :]

        if return_mu_std:
            std = torch.exp(logstd)
            return mu, std
        else:
            z = self.reparameterize(mu, logstd)
            return z
