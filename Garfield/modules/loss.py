"""
This module contains all loss functions used by the Garfield module.
"""

from typing import List, Literal, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_omics_recon_mse_loss(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data.

    Parameters
    ----------
    recon_x: torch.Tensor
         Torch Tensor of reconstructed data
    x: torch.Tensor
         Torch Tensor of ground truth data

    Returns
    -------
    MSE loss value
    """
    mse_loss = F.mse_loss(recon_x, x)  # , reduction='sum'
    return mse_loss


def compute_adj_recon_loss(pos_adj, neg_adj, temperature, EPS=1e-15):
    """
    Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.
    """
    pos_loss = -torch.log(pos_adj + EPS).mean()
    neg_loss = -torch.log(1 - neg_adj + EPS).mean()
    total_loss = (pos_loss + neg_loss) * temperature

    return total_loss


def compute_edge_recon_loss(
    edge_recon_logits: torch.Tensor,
    edge_recon_labels: torch.Tensor,
    edge_incl: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute edge reconstruction weighted binary cross entropy loss with logits
    using ground truth edge labels and predicted edge logits.

    Parameters
    ----------
    edge_recon_logits:
        Predicted edge reconstruction logits for both positive and negative
        sampled edges (dim: 2 * ´edge_batch_size´).
    edge_recon_labels:
        Edge ground truth labels for both positive and negative sampled edges
        (dim: 2 * ´edge_batch_size´).
    edge_incl:
        Boolean mask which indicates edges to be included in the edge recon loss
        (dim: 2 * ´edge_batch_size´). If ´None´, includes all edges.

    Returns
    ----------
    edge_recon_loss:
        Weighted binary cross entropy loss between edge labels and predicted
        edge probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    if edge_incl is not None:
        # Remove edges whose node pair has different categories in categorical
        # covariates for which no cross-category edges are present
        edge_recon_logits = edge_recon_logits[edge_incl]
        edge_recon_labels = edge_recon_labels[edge_incl]

    # Determine weighting of positive examples
    pos_labels = (edge_recon_labels == 1.0).sum(dim=0)
    neg_labels = (edge_recon_labels == 0.0).sum(dim=0)
    pos_weight = neg_labels / pos_labels

    # Compute weighted bce loss from logits for numerical stability
    edge_recon_loss = F.binary_cross_entropy_with_logits(
        edge_recon_logits, edge_recon_labels, pos_weight=pos_weight
    )
    return edge_recon_loss


def compute_kl_reg_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M.
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013). Equation (10).
    This will encourage encodings to distribute evenly around the center of
    a continuous and complete latent space, producing similar (for points close
    in latent space) and meaningful content after decoding.

    For detailed derivation, see
    https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes.

    Parameters
    ----------
    mu:
        Expected values of the normal latent distribution of each node (dim:
        n_nodes_current_batch, n_gps).
    logstd:
        Log standard deviations of the normal latent distribution of each node
        (dim: n_nodes_current_batch, n_gps).

    Returns
    ----------
    kl_reg_loss:
        Kullback-Leibler divergence.
    """
    # Sum over n_gps and mean over n_nodes_current_batch
    kl_reg_loss = -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu**2 - torch.exp(logstd) ** 2, 1)
    )
    return kl_reg_loss


## contrastive loss
def compute_contrastive_instanceloss(z_i, z_j, temperature):
    """
    Compute the contrastive loss given two batches of feature vectors z_i and z_j.

    Parameters:
    z_i (Tensor): Feature vectors from the first view.
    z_j (Tensor): Feature vectors from the second view.
    temperature (float): Temperature parameter to scale the dot products.

    Returns:
    loss (Tensor): The computed contrastive loss.
    """

    # Initialize Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(batch_size):
        """
        Creates a mask to zero out correlations between the same samples.

        Parameters:
        batch_size (int): The number of samples in one batch.

        Returns:
        mask (Tensor): A mask of shape (2*batch_size, 2*batch_size) where correlated samples have zero value.
        """
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    # Compute batch size and mask for correlated samples
    batch_size = z_i.size(0)
    mask = mask_correlated_samples(batch_size)
    N = 2 * batch_size
    # Concatenate feature vectors
    z = torch.cat((z_i, z_j), dim=0)
    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature
    # Extract positive samples (diagonal elements)
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    # Extract negative samples
    negative_samples = sim[mask].reshape(N, -1)
    # Create labels (positive samples are labeled as 0)
    labels = torch.zeros(N).to(positive_samples.device).long()
    # Concatenate positive and negative samples
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    # Compute loss
    loss = criterion(logits, labels)
    loss /= N

    return loss


def compute_contrastive_clusterloss(c_i, c_j, class_num, temperature):
    """
    Cluster loss function.

    Args:
        c_i (torch.Tensor): First set of cluster probabilities.
        c_j (torch.Tensor): Second set of cluster probabilities.
        class_num (int): Number of classes.
        temperature (float): Temperature scaling factor.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: The computed loss value.
    """

    # Create the mask for correlated clusters
    def mask_correlated_clusters(class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    # Initialize necessary components
    criterion = nn.CrossEntropyLoss(reduction="sum")
    similarity_f = nn.CosineSimilarity(dim=2)

    # Compute negative entropy loss for c_i and c_j
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

    p_j = c_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

    ne_loss = ne_i + ne_j

    # Concatenate c_i and c_j
    c_i = c_i.t()
    c_j = c_j.t()
    N = 2 * class_num
    c = torch.cat((c_i, c_j), dim=0)

    # Compute similarity
    sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature
    sim_i_j = torch.diag(sim, class_num)
    sim_j_i = torch.diag(sim, -class_num)

    # Select positive and negative clusters
    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_clusters(class_num)
    negative_clusters = sim[mask].reshape(N, -1)

    # Compute loss
    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = criterion(logits, labels)
    loss /= N

    return loss + ne_loss


### mmd function
def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

    Parameters
    ----------
    x: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    y: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    alphas: Tensor

    Returns
    -------
    Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def compute_omics_recon_mmd_loss(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

    Parameters
    ----------
    source_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]

    Returns
    -------
    Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(
        gaussian_kernel_matrix(source_features, target_features, alphas)
    )

    return cost
