"""
This module contains the GarphAnnTorchDataset class to provide a standardized
dataset format for the training of Garfield model.
"""

from typing import Literal, List, Optional, Tuple, Union

import numpy as np
import torch

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor

from anndata import AnnData
from mudata import MuData
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import add_self_loops, remove_self_loops


class GraphAnnTorchDataset():
    """
    Spatially annotated torch dataset class to extract node features, node
    labels, adjacency matrix and edge indices in a standardized format from an
    AnnData object.

    Parameters
    ----------
    adata:
        AnnData object with counts stored in ´adata.layers[counts_key]´ or
        ´adata.X´ depending on ´counts_key´, and sparse adjacency matrix stored
        in ´adata.obsp[adj_key]´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    self_loops:
        If ´True´, add self loops to the adjacency matrix to model autocrine
        communication.

    Returns
    ----------
    None
    """
    def __init__(self,
                 adata, # : AnnData,
                 label_name: str = None,
                 adj_key: Literal["spatial_connectivities", "connectivities"]="spatial_connectivities",
                 edge_label_adj_key: str = "edge_label_spatial_connectivities",
                 used_feat: bool = False,
                 self_loops: bool = True):
        super(GraphAnnTorchDataset, self).__init__()

        if not used_feat:
            x = adata.X
        else:
            x = np.array(adata.obsm['feat'])

        # Store features in dense format
        if sp.issparse(x):
            self.x = torch.tensor(x.toarray())
        else:
            self.x = torch.tensor(x)

        ## celltype label_encoder Todo 如果标签信息本身就是数值则不需要转换
        if label_name is not None:
            label_encoder = LabelEncoder()
            meta = np.array(adata.obs[label_name].astype('str'))
            meta = label_encoder.fit_transform(meta)
            # meta = meta.astype(np.float32)
            inverse = label_encoder.inverse_transform(range(0, np.max(meta) + 1))
            self.y = torch.from_numpy(meta.astype(np.int64))
        else:
            self.y = torch.from_numpy(np.array([1] * adata.shape[0]))

        # Store adjacency matrix in torch_sparse SparseTensor format
        if sp.issparse(adata.obsp[adj_key]):
            self.adj = sparse_mx_to_sparse_tensor(adata.obsp[adj_key])
        else:
            self.adj = sparse_mx_to_sparse_tensor(
                sp.csr_matrix(adata.obsp[adj_key]))

        # Store edge label adjacency matrix
        if edge_label_adj_key in adata.obsp:
            self.edge_label_adj = sp.csr_matrix(adata.obsp[edge_label_adj_key])
        else:
            self.edge_label_adj = None

        # Validate adjacency matrix symmetry
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")

        # 获取 edge_index
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()
        # # 获取 edge_index 对应的value 或 weight
        # self.edge_weight = self.adj.storage.value() # 对应的value
        self.edge_weight = self.adj.to_torch_sparse_coo_tensor()._values()  # 非零索引对应的值
        # 确保 edge_weight 的形状是 (num_edges, 1)，因为 edge_attr 通常是 (num_edges, attr_dim)
        self.edge_weight = torch.unsqueeze(self.edge_weight, dim=-1)

        if self_loops:
            # Add self loops to account for autocrine communication
            # Remove self loops in case there are already before adding new ones
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(self.edge_index,
                                                num_nodes=self.x.size(0))

            # 确保添加的自环数目正确
            new_edge_index_shape = self.edge_index.shape[1]
            original_edge_index_shape = self.edge_weight.shape[0]
            num_self_loops = new_edge_index_shape - original_edge_index_shape
            # 创建自环边的权重，假设权重值为1
            self_loops_weight = torch.ones((num_self_loops, 1), dtype=self.edge_weight.dtype)

            # 获取新增自环的边的数量
            # num_self_loops = self.x.size(0)
            # # 创建自环边的权重，假设权重值为1
            # self_loops_weight = torch.ones((num_self_loops, 1), dtype=self.edge_weight.dtype)
            # 将原有的edge_weight和新生成的自环边的权重拼接起来
            self.edge_weight = torch.cat([self.edge_weight, self_loops_weight], dim=0)

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x.sum(1)  # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x.size(0)


def sparse_mx_to_sparse_tensor(sparse_mx: csr_matrix) -> SparseTensor:
    """
    Convert a scipy sparse matrix into a torch_sparse SparseTensor.

    Parameters
    ----------
    sparse_mx:
        Sparse scipy csr_matrix.

    Returns
    ----------
    sparse_tensor:
        torch_sparse SparseTensor object that can be utilized by PyG.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    torch_sparse_coo_tensor = torch.sparse.FloatTensor(indices, values, shape)
    sparse_tensor = SparseTensor.from_torch_sparse_coo_tensor(
        torch_sparse_coo_tensor)
    return sparse_tensor