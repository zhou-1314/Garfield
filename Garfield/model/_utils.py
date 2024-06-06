from sklearn.preprocessing import LabelEncoder
import numpy as np
import muon as mu
# from muon import MuData
import torch
from torch_geometric.data import Data

def Transfer_scData(adata, label_name=None, data_type=None):
    """
    Converts an AnnData or MuData object into a PyTorch Geometric Data object.

    Parameters
    ----------
    adata : AnnData or MuData
        The annotated data matrix to be converted.
    label_name : str, optional
        Column name in `adata.obs` containing the labels for each node.
        If None, all nodes are assigned a default label of 1.
    data_type : str, optional
        Indicates the type of data matrix ('muData' or 'AnnData').
    Returns
    -------
    torch_geometric.data.Data
        A PyTorch Geometric Data object containing the graph data.
    """
    # Initialize an empty list for edges and their weights
    row_col = []
    if data_type == 'Paired':
        # adj = adata.uns['obsp_rna']['connectivities']
        adj = adata.obsp['connectivities_combined']
    else:
        adj = adata.obsp['connectivities']
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz()
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)
    edge_attr = torch.unsqueeze(edge_attr, dim=-1)  # 在第一维度添加维度

    ## celltype label_encoder Todo 如果标签信息本身就是数值则不需要转换
    if label_name is not None:
        label_encoder = LabelEncoder()
        if data_type == 'Paired':
            meta = np.array(adata.obs['rna:' + label_name].astype('str'))
        else:
            meta = np.array(adata.obs[label_name].astype('str'))
        meta = label_encoder.fit_transform(meta)
        # meta = meta.astype(np.float32)
        inverse = label_encoder.inverse_transform(range(0, np.max(meta) + 1))
        y = torch.from_numpy(meta.astype(np.int64))
    else:
        y = torch.from_numpy(np.array([1] * adata.shape[0]))

    if type(adata.X) == np.ndarray:
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.FloatTensor(adata.X),  # .todense()
                    y=torch.LongTensor(y))
    else:
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.FloatTensor(adata.X.todense()),  # .todense()
                    y=torch.LongTensor(y))
    return data


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape) # torch.sparse.FloatTensor(indices, values, shape)


def extract_subgraph(data, label):
    # 找出所有具有特定标签的节点索引
    node_indices = (data.y == label).nonzero(as_tuple=True)[0]

    # 提取这些节点的特征
    sub_x = data.x[node_indices]
    sub_y = data.y[node_indices]

    # 提取包含这些节点的边
    edge_index = data.edge_index
    mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
    sub_edge_index = edge_index[:, mask]

    # 将边索引调整为新的节点索引
    _, new_indices = node_indices.unique(return_inverse=True)
    remap = {old_idx.item(): new_idx for old_idx, new_idx in zip(node_indices, new_indices)}
    for i in range(sub_edge_index.shape[1]):
        sub_edge_index[0, i] = remap[sub_edge_index[0, i].item()]
        sub_edge_index[1, i] = remap[sub_edge_index[1, i].item()]

    # 提取对应的边特征
    sub_edge_attr = data.edge_attr[mask] if data.edge_attr is not None else None

    return Data(x=sub_x, edge_index=sub_edge_index, y=sub_y, edge_attr=sub_edge_attr)
