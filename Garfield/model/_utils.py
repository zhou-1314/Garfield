import random
import numpy as np
import muon as mu
# from muon import MuData
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
# from torch_scatter import scatter


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
    # if data_type == 'Paired':
    #     # adj = adata.uns['obsp_rna']['connectivities']
    #     adj = adata.obsp['connectivities_combined']
    # else:
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


def split_data(num_y, val_split, test_split):
    split_idx = list(range(num_y))
    random.shuffle(split_idx)
    train_split = (1 - test_split - val_split)
    train_idx = split_idx[: int(len(split_idx) * train_split)] # Train mask
    train_mask = torch.zeros(num_y, dtype = torch.bool)
    train_mask[train_idx] = 1
    val_idx = split_idx[ int(len(split_idx) * train_split) : int(len(split_idx) * (train_split + val_split))] # Val mask
    val_mask = torch.zeros(num_y, dtype=torch.bool)
    val_mask[val_idx] = 1
    test_idx = split_idx[int(len(split_idx) * (train_split + val_split)) :] # Test mask
    test_mask = torch.zeros(num_y, dtype=torch.bool)
    test_mask[test_idx] = 1
    return train_mask, val_mask, test_mask


from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, GraphSAINTEdgeSampler
from torch_geometric.utils import structured_negative_sampling
def generate_batch(data, edge_attr_dict, mask, batch_size, loader_type="graphsaint",
                   num_layers=2):
    # Positive edges
    if mask == "train":
        pos_edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_attr[data.train_mask]
        y = data.y[data.train_mask]
    elif mask == "val":
        pos_edge_index = data.edge_index[:, data.val_mask]
        edge_type = data.edge_attr[data.val_mask]
        y = data.y[data.train_mask]
    elif mask == "test":
        pos_edge_index = data.edge_index[:, data.test_mask]
        edge_type = data.edge_attr[data.test_mask]
        y = data.y[data.train_mask]
    else:
        pos_edge_index = data.edge_index
        edge_type = data.edge_attr
        y = data.y[data.train_mask]

    # Negative edges
    neg_edge_index, neg_edge_type = negative_sampler(pos_edge_index, edge_type, edge_attr_dict)

    # All edges and labels
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    total_edge_type = torch.cat([edge_type, neg_edge_type], dim=-1)

    # Save information for the loader
    data = Data(x=data.x, edge_index=total_edge_index, edge_attr=total_edge_type, y=y)
    data.n_id = torch.arange(data.num_nodes)
    if loader_type == "neighbor":
        loader = NeighborLoader(data, num_neighbors=[-1] * num_layers, batch_size=batch_size,
                                input_nodes=torch.arange(data.num_nodes), shuffle=True)
    elif loader_type == "graphsaint":
        # loader = GraphSAINTRandomWalkSampler(data, batch_size = batch_size, walk_length = num_layers)
        loader = GraphSAINTEdgeSampler(data, batch_size=batch_size, num_steps=16)
    else:
        raise NotImplementedError

    return loader


def negative_sampler(pos_edge_index, edge_type, edge_attr_dict):
    if len(edge_type) == 0: return pos_edge_index, edge_type
    neg_edge_index = None
    neg_edge_type = []
    for attr, idx in edge_attr_dict.items():
        mask = (edge_type == idx)
        if mask.sum() == 0: continue
        pos_rel_edge_index = pos_edge_index.T[mask].T
        neg_source, neg_target, neg_rand = structured_negative_sampling(pos_rel_edge_index)
        neg_rel_edge_index = torch.stack((neg_source, neg_rand), dim=0)
        """
        neg_rel_edge_index = pos_rel_edge_index.clone()
        rand_axis = random.sample([0, 1], 1)[0]
        rand_index = torch.randperm(pos_rel_edge_index.size(1))
        neg_rel_edge_index[rand_axis, :] = pos_rel_edge_index[rand_axis, rand_index]
        """
        if neg_edge_index == None:
            neg_edge_index = neg_rel_edge_index
        else:
            neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1)

        neg_edge_type.extend([idx] * mask.sum())

    return neg_edge_index, torch.tensor(neg_edge_type)


def to_dense_adj(edge_index, edge_attr, num_nodes, device, alpha=2):
    # edge_index: [2, E] tensor of edge indices
    # edge_attr: [E, D] tensor of edge attributes
    # num_nodes: number of nodes in the graph

    # 初始化一个邻接矩阵，大小为[num_nodes, num_nodes]
    adj_matrix = alpha * torch.ones(num_nodes, num_nodes).to(device)

    # 使用scatter函数将边属性加到邻接矩阵的对应位置
    # edge_index[0] 是起始节点，edge_index[1] 是终止节点
    adj_matrix[edge_index[0], edge_index[1]] = edge_attr/alpha

    return adj_matrix


def get_prior(celltype1, celltype2, alpha=2):
    """
    Create a prior correspondence matrix according to cell labels

    Parameters
    ----------
    celltype1
        cell labels of dataset X
    celltype2
        cell labels of dataset Y
    alpha
        the confidence of label, ranges from (1, inf). Higher alpha means better confidence. Default: 2.0

    Return
    ------
    torch.tensor
        a prior correspondence matrix between cells
    """

    Couple = alpha * torch.ones(len(celltype1), len(celltype2))

    for i in set(celltype1):
        index1 = np.where(celltype1 == i)
        if i in set(celltype2):
            index2 = np.where(celltype2 == i)
            for j in index1[0]:
                Couple[j, index2[0]] = 1 / alpha

    return Couple

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


import io
import logging
import pickle

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack

logger = logging.getLogger(__name__)


def _validate_var_names(adata, source_var_names):
    # Warning for gene percentage
    user_var_names = adata.var_names
    try:
        percentage = (len(user_var_names.intersection(source_var_names)) / len(user_var_names)) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(f"WARNING: Query shares {percentage}% of its genes with the reference."
                           "This may lead to inaccuracy in the results.")
    except Exception:
        logger.warning("WARNING: Something is wrong with the reference genes.")

    user_var_names = user_var_names.astype(str)
    new_adata = adata

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in source_var_names:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print("Query data is missing expression data of ",
              len(ref_genes_not_in_query),
              " genes which were contained in the reference dataset.")
        print("The missing information will be filled with zeroes.")

        filling_X = np.zeros((len(adata), len(ref_genes_not_in_query)))
        if isinstance(adata.X, csr_matrix):
            filling_X = csr_matrix(filling_X)  # support csr sparse matrix
            new_target_X = hstack((adata.X, filling_X))
        else:
            new_target_X = np.concatenate((adata.X, filling_X), axis=1)
        new_target_vars = adata.var_names.tolist() + ref_genes_not_in_query
        new_adata = AnnData(new_target_X, dtype="float32")
        new_adata.var_names = new_target_vars
        new_adata.obs = adata.obs.copy()

    if len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)) > 0:
        print(
            "Query data contains expression data of ",
            len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)),
            " genes that were not contained in the reference dataset. This information "
            "will be removed from the query data object for further processing.")

        # remove unseen gene information and order anndata
        new_adata = new_adata[:, source_var_names].copy()

    print(new_adata)

    return new_adata


class UnpicklerCpu(pickle.Unpickler):
    """Helps to pickle.load a model trained on GPU to CPU.

    See also https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219.
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)