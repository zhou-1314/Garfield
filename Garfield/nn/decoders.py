"""
This module contains the decoder used by the Garfield model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from .utils import DSBatchNorm, compute_cosine_similarity


class GATDecoder(nn.Module):
    """
    Graph Attention Network (GAT) Decoder class.

    This class implements a GAT-based decoder for reconstructing node features
    from latent representations. It supports domain-specific batch normalization
    (DSBN) and edge weights.

    Parameters
    ----------
    in_channels : int
        Number of input feature dimensions.
    hidden_dims : list[int]
        List of output dimensions for each hidden GAT layer, in ascending order.
    out_channels : int
        Number of output feature dimensions.
    conv_type : str
        Type of GAT convolution layer to use ('GAT' or 'GATv2Conv').
    num_heads : int
        Number of attention heads for each GAT layer.
    dropout : float
        Dropout rate applied to GAT layers.
    concat : bool
        Whether to concatenate the output of all attention heads or not.
    num_domains : int or str
        Number of domains for domain-specific batch normalization (DSBN). If `1`, regular batch normalization is used.
    used_edge_weight : bool, optional
        Whether to use edge weights in the GAT layers. Default is False.
    used_DSBN : bool, optional
        Whether to use domain-specific batch normalization (DSBN). Default is False.
    """

    def __init__(
        self,
        in_channels,
        hidden_dims,
        out_channels,
        conv_type,
        num_heads,
        dropout,
        concat,
        num_domains="",
        used_edge_weight=False,
        used_DSBN=False,
    ):
        """
        Initializes the GATDecoder, which consists of multiple Graph Attention Network (GAT) layers
        followed by domain-specific normalization (if applicable).
        """
        super(GATDecoder, self).__init__()
        self.used_DSBN = used_DSBN
        self.used_edge_weight = used_edge_weight
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = dropout
        if conv_type == "GAT":
            GATLayer = GATConv
        elif conv_type == "GATv2Conv":
            GATLayer = GATv2Conv

        num_hidden_layers = len(hidden_dims)
        num_heads_list = [num_heads] * num_hidden_layers
        concat_list = [concat] * num_hidden_layers
        for i in range(num_hidden_layers):
            if concat_list[i]:
                current_dim = hidden_dims[::-1][i] * num_heads_list[i]
            else:
                current_dim = hidden_dims[::-1][i]  # [::-1] 代表反转

            norm = None
            if type(num_domains) == int:
                if num_domains == 1:  # TO DO
                    norm = nn.BatchNorm1d(current_dim)
                elif num_domains > 1 and self.used_DSBN:
                    # num_domains >1 represent domain-specific batch normalization of n domain
                    norm = DSBatchNorm(current_dim, num_domains)
            self.norm.append(norm)

        current_dim = in_channels  # in_channels
        dropout_list = [dropout] * num_hidden_layers
        for i in range(num_hidden_layers):
            layer = GATLayer(
                in_channels=current_dim,
                out_channels=hidden_dims[::-1][i],
                heads=num_heads_list[i],
                dropout=dropout_list[i],
                concat=concat_list[i],
                edge_dim=1 if self.used_edge_weight else None,
            )
            self.layers.append(layer)
            if concat_list[i]:
                current_dim = hidden_dims[::-1][i] * num_heads_list[i]
            else:
                current_dim = hidden_dims[::-1][i]

        ### 数据集总重构的layers
        self.conv_recon = GATLayer(
            in_channels=current_dim,
            out_channels=out_channels,
            heads=num_heads,
            concat=False,
            edge_dim=1 if self.used_edge_weight else None,
            dropout=dropout,
        )

    def forward(self, x, data):
        """
        Performs a forward pass through the GAT decoder layers and reconstructs the node features.

        Parameters
        ----------
        x : torch.Tensor
            Node features (shape: [num_nodes, feature_dim]).
        data : Data
            PyTorch Geometric Data object containing edge index, edge attributes, and domain labels.

        Returns
        -------
        torch.Tensor
            Reconstructed node features after passing through the GAT layers and applying normalization (if applicable).
        """
        edge_index, y, edge_index_all = data.edge_index, data.y, data.edge_attr
        edge_weight = edge_index_all[:, 2]
        # edge_weight = torch.ones(edge_index.shape[1]).cpu().numpy()  # 先将张量转移到CPU，再转为numpy

        for idx, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                edge_index,
                edge_attr=edge_weight if self.used_edge_weight else None,
                return_attention_weights=True,
            )
            if self.used_DSBN:
                if self.norm:
                    if len(x) == 1:
                        pass
                    elif self.norm[0].__class__.__name__ == "DSBatchNorm":
                        x = self.norm[idx](x, y)
                    else:
                        x = self.norm[idx](x)
                    x = F.relu(x)

        recon_x, _ = self.conv_recon(
            x,
            edge_index,
            edge_attr=edge_weight if self.used_edge_weight else None,
            return_attention_weights=True,
        )

        return recon_x


### GCN decoder
class GCNDecoder(nn.Module):
    """
    Graph Convolutional Network (GCN) Decoder class.

    This class implements a GCN-based decoder for reconstructing node features
    from latent representations. It supports domain-specific batch normalization
    (DSBN) and edge weights.

    Parameters
    ----------
    in_channels : int
        Number of input feature dimensions.
    hidden_dims : list[int]
        List of output dimensions for each hidden GCN layer, in ascending order.
    out_channels : int
        Number of output feature dimensions.
    dropout : float, optional
        Dropout rate applied to GCN layers. Default is 0.2.
    num_domains : int or str
        Number of domains for domain-specific batch normalization (DSBN). If `1`, regular batch normalization is used.
    used_edge_weight : bool, optional
        Whether to use edge weights in the GCN layers. Default is False.
    used_DSBN : bool, optional
        Whether to use domain-specific batch normalization (DSBN). Default is False.
    """

    def __init__(
        self,
        in_channels,
        hidden_dims,
        out_channels,
        dropout=0.2,
        num_domains="",
        used_edge_weight=False,
        used_DSBN=False,
    ):
        """
        Initializes the GCNDecoder, consisting of multiple Graph Convolutional Network (GCN) layers followed by domain-specific normalization (if applicable).
        """
        super(GCNDecoder, self).__init__()
        # 如果 hidden_channels 是单一数字，将其转换成单元素列表
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.used_DSBN = used_DSBN
        self.used_edge_weight = used_edge_weight
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()

        # 创建一个包含所有GCN层的列表
        gcn_layers = []
        hidden_dims = hidden_dims[::-1]  # 反转
        total_layers = [in_channels] + hidden_dims
        for i in range(len(total_layers) - 1):
            gcn_layers.append(
                GCNConv(total_layers[i], total_layers[i + 1], dropout=dropout)
            )

        # 使用 nn.ModuleList 以确保所有层都被正确注册
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # 输出层
        self.gcn_recon = GCNConv(hidden_dims[-1], out_channels, dropout=dropout)

        # self.norm 层
        num_hidden_layers = len(hidden_dims)
        for i in range(num_hidden_layers):
            current_dim = hidden_dims[i]

            norm = None
            if type(num_domains) == int:
                if num_domains == 1:  # TO DO
                    norm = nn.BatchNorm1d(current_dim)
                elif num_domains > 1 and self.used_DSBN:
                    norm = DSBatchNorm(
                        current_dim, num_domains
                    )  # num_domains >1 represent domain-specific batch normalization of n domain
            self.norm.append(norm)

    def forward(self, x, data):
        """
        Performs a forward pass through the GCN decoder layers and reconstructs the node features.

        Parameters
        ----------
        x : torch.Tensor
            Node features (shape: [num_nodes, feature_dim]).
        data : Data
            PyTorch Geometric Data object containing edge index, edge attributes, and domain labels.

        Returns
        -------
        torch.Tensor
            Reconstructed node features after passing through the GCN layers and applying normalization (if applicable).
        """
        edge_index, y, edge_index_all = data.edge_index, data.y, data.edge_attr
        edge_weight = edge_index_all[:, 2]
        # edge_weight = torch.ones(edge_index.shape[1]).cpu().numpy()  # 先将张量转移到CPU，再转为numpy

        ### latent
        for idx, layer in enumerate(self.gcn_layers):
            x = layer(
                x,
                edge_index,
                edge_weight=edge_weight if self.used_edge_weight else None,
            )

            if self.used_DSBN:
                if self.norm:
                    if len(x) == 1:
                        pass
                    elif self.norm[0].__class__.__name__ == "DSBatchNorm":
                        print("Perform DSBN normalization...")
                        x = self.norm[idx](x, y)
                    else:
                        print("Perform batch normalization...")
                        x = self.norm[idx](x)
                    x = F.relu(x)

        x_recon = self.gcn_recon(
            x, edge_index, edge_weight=edge_weight if self.used_edge_weight else None
        )

        return x_recon


class CosineSimGraphDecoder(nn.Module):
    """
    Cosine similarity graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    target nodes as input, and calculates the element-wise cosine similarity
    between source and target nodes to return the reconstructed edge logits.

    The sigmoid activation function to compute reconstructed edge probabilities
    is integrated into the binary cross entropy loss for computational
    efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """

    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        print("COSINE SIM GRAPH DECODER -> " f"dropout_rate: {dropout_rate}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine similarity graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and target nodes
            (dim: 4 * edge_batch_size x n_gps due to negative edges).

        Returns
        ----------
        edge_recon_logits:
            Reconstructed edge logits (dim: 2 * edge_batch_size due to negative
            edges).
        """
        z = self.dropout(z)

        # Compute element-wise cosine similarity
        edge_recon_logits = compute_cosine_similarity(
            z[: int(z.shape[0] / 2)], z[int(z.shape[0] / 2) :]  # ´edge_label_index[0]´
        )  # ´edge_label_index[1]´
        return edge_recon_logits
