import warnings
from types import SimpleNamespace

import torch
from ._layers import GATEncoder, GCNEncoder, GCNModelVAE

class Garfield(torch.nn.Module):
    """
    Garfield: Graph-based Contrastive Learning enable Fast Single-Cell Embedding
    """
    def __init__(self, args: SimpleNamespace, number_of_genes, num_classes):
        """
        :param args: Arguments object.
        :param number_of_genes: Number of feature of each node.
        :param num_classes: Number of batch.
        """
        super(Garfield, self).__init__()

        self.args = args
        self.num_features = number_of_genes
        self.num_classes = num_classes
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        ## 选择 encoder
        ## 设定参数
        if self.args.conv_type in ['GAT']:
            encoder = GATEncoder(
                in_channels=self.num_features,
                hidden_dims=self.args.hidden_dims,
                latent_dim=self.args.bottle_neck_neurons,
                num_heads=self.args.num_heads,
                svd_q=self.args.svd_q,
                dropout=self.args.dropout,
                concat=self.args.concat,
                num_domains=1,  # batch normalization
                used_edge_weight=self.args.used_edge_weight,
                used_DSBN=self.args.used_DSBN
            )
        else:
            encoder = GCNEncoder(
                in_channels=self.num_features,
                hidden_channels=self.args.hidden_dims,
                out_channels=self.args.bottle_neck_neurons,
                svd_q=self.args.svd_q,
                dropout=self.args.dropout,
                num_domains=1,  # batch normalization
                used_edge_weight=self.args.used_edge_weight,
                used_DSBN=self.args.used_DSBN
            )

        ## GCNModelVAE
        self.VGAE = GCNModelVAE(encoder, self.num_features, self.num_classes, self.args)

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
        if self.args.used_recon_exp:
            return self.VGAE.encodeBatch(data)  # mean, recon_features
        else:
            return self.VGAE.encodeBatch(data)  # mean

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        """
        if self.args.used_recon_exp:
            z, z_1, z_2, c_1, c_2, mu, logstd, recon_features = self.VGAE(data)
            return z, z_1, z_2, c_1, c_2, mu, logstd, recon_features
        else:
            z, z_1, z_2, c_1, c_2, mu, logstd = self.VGAE(data)
            return z, z_1, z_2, c_1, c_2, mu, logstd