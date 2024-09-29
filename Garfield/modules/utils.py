import torch
from torch_geometric.data import Data

def extract_subgraph(data, node_indices):
    # 如果 node_indices 是 slice 类型，转换为具体的索引列表
    if isinstance(node_indices, slice):
        node_indices = torch.arange(data.num_nodes)[node_indices]

    # Check if node_indices is a tensor, if not convert it
    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)

    # 提取这些节点的特征
    sub_x = data.x[node_indices]
    if data.y is not None:
        sub_y = data.y[node_indices]
    else:
        sub_y = None

    # 提取包含这些节点的边
    edge_index = data.edge_index
    node_indices = node_indices.to(edge_index.device)
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

def to_float_tensor(tensor):
    # 检查张量的数据类型是否为浮点类型
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
        # 如果不是浮点类型，则将其转换为浮点类型
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        return tensor
    # 如果已经是浮点类型，判断是否为32位，否则直接返回原张量
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
