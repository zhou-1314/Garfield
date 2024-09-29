import numpy as np

import torch
import torch.nn as nn

# mask feature function
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

# Domain-specific Batch Normalization（领域特定的批次归一化）是一种调整神经网络中批次归一化层以适应不同输入数据源或领域的技术。
class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)]).to(self.device)

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        device = x.device  # 获取输入数据的设备
        out = torch.zeros(x.size(0), self.num_features, device=device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]
            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                self.bns[i].training = False
                out[indices] = self.bns[i](x[indices])
                self.bns[i].training = True
        return out


def compute_cosine_similarity(tensor1: torch.Tensor,
                              tensor2: torch.Tensor,
                              eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the element-wise cosine similarity between two 2D tensors.

    Parameters
    ----------
    tensor1:
        First tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).
    tensor2:
        Second tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).

    Returns
    ----------
    cosine_sim:
        Result tensor that contains the computed element-wise cosine
        similarities (dim: n_obs).
    """
    tensor1_norm = tensor1.norm(dim=1)[:, None]
    tensor2_norm = tensor2.norm(dim=1)[:, None]
    tensor1_normalized = tensor1 / torch.max(
        tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normalized = tensor2 / torch.max(
        tensor2_norm, eps * torch.ones_like(tensor2_norm))
    cosine_sim = torch.mul(tensor1_normalized, tensor2_normalized).sum(1)
    return cosine_sim
