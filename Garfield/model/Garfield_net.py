import warnings
from types import SimpleNamespace

import torch
from ._layers import GATEncoder, GCNEncoder, GNNModelVAE

class Garfield(torch.nn.Module):
    """
    Garfield: Graph-based Contrastive Learning enable Fast Single-Cell Embedding
    """
    def __init__(self, number_of_genes, hidden_dims, bottle_neck_neurons, num_classes,
                 num_heads, dropout, concat, svd_q, used_edge_weight, used_DSBN,
                 used_recon_exp, conv_type, cluster_num, gnn_layer):
        """
        :param args: Arguments object.
        :param number_of_genes: Number of feature of each node.
        :param num_classes: Number of batch.
        """
        super(Garfield, self).__init__()
        self.num_features = number_of_genes
        self.hidden_dims = hidden_dims
        self.bottle_neck_neurons = bottle_neck_neurons
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.svd_q = svd_q
        self.used_edge_weight = used_edge_weight
        self.used_DSBN = used_DSBN
        self.n_domain = num_classes
        self.used_recon_exp = used_recon_exp
        self.conv_type = conv_type
        self.cluster_num = cluster_num
        self.gnn_layer = gnn_layer
        self.freeze = False
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        ## 选择 encoder
        ## 设定参数
        if self.conv_type in ['GAT', 'GATv2Conv']:
            encoder = GATEncoder(
                in_channels=self.num_features,
                hidden_dims=self.hidden_dims,
                latent_dim=self.bottle_neck_neurons,
                conv_type=self.conv_type,
                num_heads=self.num_heads,
                dropout=self.dropout,
                concat=self.concat,
                num_domains=1,  # batch normalization
                svd_q=self.svd_q,
                used_edge_weight=self.used_edge_weight,
                used_DSBN=self.used_DSBN
            )
        else:
            encoder = GCNEncoder(
                in_channels=self.num_features,
                hidden_channels=self.hidden_dims,
                out_channels=self.bottle_neck_neurons,
                dropout=self.dropout,
                num_domains=1,  # batch normalization
                svd_q=self.svd_q,
                used_edge_weight=self.used_edge_weight,
                used_DSBN=self.used_DSBN
            )

        ## GCNModelVAE
        self.VGAE = GNNModelVAE(
            encoder,
            self.bottle_neck_neurons,
            self.hidden_dims,
            self.num_features,
            self.num_heads,
            self.dropout,
            self.concat,
            self.n_domain,
            self.used_edge_weight,
            self.used_DSBN,
            self.used_recon_exp,
            self.conv_type,
            self.cluster_num,
            self.gnn_layer
        )

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    ## embedding
    def encodeBatch(self, data):
        if self.used_recon_exp:
            return self.VGAE.encodeBatch(data)  # mean, recon_features
        else:
            return self.VGAE.encodeBatch(data)  # mean

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        """
        if self.used_recon_exp:
            z, z_1, z_2, c_1, c_2, mu, logstd, recon_features = self.VGAE(data)
            return z, z_1, z_2, c_1, c_2, mu, logstd, recon_features
        else:
            z, z_1, z_2, c_1, c_2, mu, logstd = self.VGAE(data)
            return z, z_1, z_2, c_1, c_2, mu, logstd