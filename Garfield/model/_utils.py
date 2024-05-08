from sklearn.preprocessing import LabelEncoder
import numpy as np
import muon as mu
# from muon import MuData
import torch
from torch_geometric.data import Data

def Transfer_scData(adata, label_name=None, profile=None):
    # create sparse matrix
    row_col = []
    if profile == 'muData':
        adj = adata.uns['obsp_rna']['connectivities']
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
        if profile == 'muData':
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