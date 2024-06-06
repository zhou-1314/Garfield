import torch
import torch.nn as nn
import math
from torch.autograd import Variable

### VGAE loss
def VGAE_loss(VGAE_model, z, mu, logstd, pos_edge_index, neg_edge_index=None):
    """
    Computes the variational graph autoencoder (VGAE) loss.

    Parameters
    ----------
    VGAE_model : torch.nn.Module
        The VGAE model instance used to compute the reconstruction and KL divergence losses.
    z : torch.Tensor
        The latent space representation :math:`\mathbf{Z}`.
    mu : torch.Tensor
        The mean of the latent space distribution.
    logstd : torch.Tensor
        The logarithm of the standard deviation of the latent space distribution.
    pos_edge_index : torch.Tensor
        Indices of positive edges to train against, in COO format.
    neg_edge_index : torch.Tensor, optional
        Indices of negative edges to train against, in COO format. If not provided, negative sampling is used.
        Defaults to `None`.

    Returns
    -------
    torch.Tensor
        The sum of reconstruction loss and KL divergence loss.
    """
    rec_loss = VGAE_model.recon_loss(z, pos_edge_index, neg_edge_index)
    kl_loss = 1 / z.size(0) * VGAE_model.kl_loss(mu, logstd)

    return rec_loss + kl_loss

## contrastive loss
class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        mask = self.mask_correlated_samples(batch_size)
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
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
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
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
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        C, D] matrix
    p
        p-norm

    Return
    ------
    [R, C] matrix
        distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def unbalanced_ot(tran, mu1, mu2, device, Couple, reg=0.1, reg_m=1.0):
    '''
    Calculate a unbalanced optimal transport matrix between batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix.
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    # tran = torch.tensor(pi0, dtype=torch.float).to(device)
    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)
    # cost_pp = ot.dist(mu1, mu2)

    # if query_weight is None:
    p_s = torch.ones(ns, 1) / ns
    # else:
    # query_batch_weight = query_weight[idx_q]
    # p_s = query_batch_weight/torch.sum(query_batch_weight)

    # if ref_weight is None:
    p_t = torch.ones(nt, 1) / nt
    # else:
    # ref_batch_weight = ref_weight[idx_r]
    # p_t = ref_batch_weight/torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        # tran = tran_Init
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if Couple is not None:
            # print(cost_pp)
            # print(Couple)
            cost = cost_pp * Couple
        else:
            cost = cost_pp

        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual = (p_s / (kernel @ b)) ** f
            b = (p_t / (torch.t(kernel) @ dual)) ** f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    # pho = tran.mean()
    # h_func = 1 - 0.5 * ( 1 + torch.sign(pho - tran) )
    # hat_tran = tran * h_func
    # d_fgw1 = (cost_pp * hat_tran.detach().data).sum()
    # d_fgw2 = ((tran.detach().data - hat_tran.detach().data) * torch.log(1 + torch.exp(-cost_pp))).sum()
    # d_fgw = d_fgw1 + d_fgw2

    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()


def unbalanced_ot_parameter(tran, mu1, mu2, device, Couple, reg=0.1, reg_m_1=1, reg_m_2=1):
    '''
    Calculate a unbalanced optimal transport matrix between batches with different reg_m parameters.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix.
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m_1
        Unbalanced OT parameter 1. Larger values means more balanced OT. Default: 1.0
    reg_m_2
        Unbalanced OT parameter 2. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)
    # cost_pp = ot.dist(mu1, mu2)

    # if query_weight is None:
    p_s = torch.ones(ns, 1) / ns
    # else:
    # query_batch_weight = query_weight[idx_q]
    # p_s = query_batch_weight/torch.sum(query_batch_weight)

    # if ref_weight is None:
    p_t = torch.ones(nt, 1) / nt
    # else:
    # ref_batch_weight = ref_weight[idx_r]
    # p_t = ref_batch_weight/torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        # tran = tran_Init
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f1 = reg_m_1 / (reg_m_1 + reg)
    f2 = reg_m_2 / (reg_m_2 + reg)

    for m in range(10):
        if Couple is not None:
            cost = cost_pp * Couple
        else:
            cost = cost_pp.to(device)

        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual = (p_s / (kernel @ b)) ** f1
            b = (p_t / (torch.t(kernel) @ dual)) ** f2
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    # pho = tran.mean()
    # h_func = 1 - 0.5 * ( 1 + torch.sign(pho - tran) )
    # hat_tran = tran * h_func
    # d_fgw1 = (cost_pp * hat_tran.detach().data).sum()
    # d_fgw2 = ((tran.detach().data - hat_tran.detach().data) * torch.log(1 + torch.exp(-cost_pp))).sum()
    # d_fgw = d_fgw1 + d_fgw2

    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()

## 跨模态loss
def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim

