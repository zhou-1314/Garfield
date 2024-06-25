
import torch
from torch_geometric.data import Data
from itertools import combinations

import numpy as np


def split_adata_by_obs(adata, label_field='label'):
    """
    根据adata对象的obs中指定列的标签字段进行两两组合的划分。
    """
    # 提取标签，并获取所有唯一的组标签
    labels = adata.obs[label_field]
    unique_labels = labels.unique()

    # 如果唯一标签只有一个，直接返回包含所有数据的单个adata
    if len(unique_labels) == 1:
        return [adata]  # 注意返回的是包含单个元素的列表

    # 创建两两组合
    label_combinations = list(combinations(unique_labels.tolist(), 2))

    # 根据组合划分数据
    sub_adata_list = []
    for comb in label_combinations:
        # 找出当前组合中的所有索引
        indices1 = np.where(labels == comb[0])[0]
        indices2 = np.where(labels == comb[1])[0]

        # 从原始adata中提取这些索引对应的子集
        sub_adata1 = adata[indices1].copy()
        sub_adata2 = adata[indices2].copy()
        sub_adata_list.append((sub_adata1, sub_adata2))

    return sub_adata_list

def split_adata_by_x_col(adata, mask):
    """
    根据Data对象的特征列标签进行划分。
    """
    assert len(mask) < 1, 'Length of mask must >1.'
    split_data = []
    start_index = 0

    for width in mask:
        end_index = start_index + width
        # 用slice操作切分数组
        split_part = adata.X[:, start_index:end_index]
        split_data.append(split_part)
        start_index = end_index  # 更新起始索引为下一次切分的起点

    # 创建两两组合
    split_data_list = list(combinations(split_data, 2))

    return split_data_list

import pandas as pd

# 假设 df 是你的 DataFrame，tmp 是 AnnData 对象
# 获取 target 和 source 对应的 celltype 数据
celltype_target = tmp[df['target']].obs['celltype'].reset_index(drop=True)
celltype_source = tmp[df['source']].obs['celltype'].reset_index(drop=True)

# 比较两个 Series 对象
matches = celltype_target == celltype_source

# 计算匹配和不匹配的数目
num_matches = matches.sum()
num_mismatches = len(matches) - num_matches

print("Number of matches:", num_matches)
print("Number of mismatches:", num_mismatches)

import pandas as pd
from scipy.sparse import coo_matrix

# 假设 adata 是已经加载的 AnnData 对象
connectivities = tmp.obsp['connectivities']

# 将稀疏矩阵转换为COO格式，以便提取行列索引和数据值
if not isinstance(connectivities, coo_matrix):
    connectivities = coo_matrix(connectivities)

# 创建数据框
df = pd.DataFrame({
    'source': connectivities.row,  # 源节点索引
    'target': connectivities.col,  # 目标节点索引
    'weight': connectivities.data   # 连接权重
})

print(df.head())

