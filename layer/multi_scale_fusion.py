import math
from typing import Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from models.pna import PNAConv
import torch

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv, APPNP, SGConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax, degree, subgraph, to_scipy_sparse_matrix, segregate_self_loops, add_remaining_self_loops
import numpy as np
import scipy.sparse as sp
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pad_sequence




class BilinearInteractionFusion(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, heads, act='ReLU', dropout=0.2, k=3):
        super(BilinearInteractionFusion, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.heads = heads

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        if heads <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, heads, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, heads, 1, 1).normal_())

        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, heads), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, clique_x_ban_lengths, softmax=True, drug=False):
        if not drug:
            clique_x_ban_lengths = clique_x_ban_lengths.to(v[0].device)
            max_v_len = max(clique_x_ban_lengths)
            v_padded = pad_sequence(v, batch_first=True, padding_value=0)


            v_mask = torch.arange(max_v_len).to(v_padded.device) < clique_x_ban_lengths[:, None]
            v_mask = v_mask.unsqueeze(1).unsqueeze(3)


            v_ = self.v_net(v_padded)  # [B, max_v_len, h_dim*k]
            q_ = self.q_net(q)  # [B, num_cluster, h_dim*k]

            # attention maps
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', self.h_mat, v_, q_) + self.h_bias
            att_maps = att_maps.masked_fill(~v_mask, float('-inf'))

        else:

            B, _ = v.size()
            v = v.unsqueeze(1)  # [B, 1, v_dim]
            v_ = self.v_net(v)  # [B, 1, h_dim*k]
            q_ = self.q_net(q)  # [B, num_cluster, h_dim*k]

            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', self.h_mat, v_, q_) + self.h_bias

        if torch.isnan(att_maps).any():
            print("Attention map contains NaN after masking!")

        if softmax:
            att_maps = F.softmax(att_maps, dim=2)


        att_map_v = att_maps.mean(dim=1)
        q_updated = torch.bmm(att_map_v, q)

        att_map_q = att_map_v.transpose(1, 2)
        if not drug:
            v_updated = torch.bmm(att_map_q, v_padded)
        else:
            v_updated = torch.bmm(att_map_q, v)


        alpha = att_maps.sum(dim=2).transpose(1, 2)
        if softmax:
            alpha = F.softmax(alpha, dim=1)
        alpha = alpha.squeeze(0)

        if not drug:

            q_updated_trimmed = [q_updated[i, :l] for i, l in enumerate(clique_x_ban_lengths)]
            q_updated_all = torch.cat(q_updated_trimmed, dim=0)
            v_updated_all = v_updated.reshape(-1, v_updated.size(-1))
        else:

            q_updated_all = q_updated.reshape(-1, q_updated.size(-1))
            v_updated_all = v_updated.reshape(-1, v_updated.size(-1))

        return q_updated_all, v_updated_all, alpha

    def reset_parameters(self):

        if hasattr(self.v_net, 'reset_parameters'):
            self.v_net.reset_parameters()
        if hasattr(self.q_net, 'reset_parameters'):
            self.q_net.reset_parameters()

        if hasattr(self, 'p_net'):
            pass
        if hasattr(self, 'h_net'):
            self.h_net.reset_parameters()
        else:

            nn.init.normal_(self.h_mat)
            nn.init.normal_(self.h_bias)

        self.bn.reset_parameters()

class FCNet(nn.Module):

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    def reset_parameters(self):
        for m in self.main:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

def group_features_by_batch(features, batch_indices):
    """
    Group features by batch indices.

    Args:
        features (torch.Tensor): Feature tensor of shape (num_items, feature_dim).
        batch_indices (torch.Tensor): Batch index tensor of shape (num_items,).

    Returns:
        list of torch.Tensor: List of feature tensors grouped by batch.
        torch.Tensor: Tensor containing the number of features in each batch.
    """
    # 确定批次数量
    batchsize = batch_indices.max().item() + 1

    # 初始化一个列表，用于存储每个批次的特征
    grouped_features = [[] for _ in range(batchsize)]

    # 遍历 batch_indices，将特征按批次分组
    for i in range(len(batch_indices)):
        batch_idx = batch_indices[i].item()
        grouped_features[batch_idx].append(features[i])

    # 将每个批次的特征列表转换为张量
    grouped_features = [torch.stack(batch) for batch in grouped_features]

    # 获取每个批次的特征数量
    lengths = torch.tensor([len(batch) for batch in grouped_features], dtype=torch.long)

    return grouped_features, lengths


def weighted_sum(features, weights):
    """
    Perform a weighted sum of multiple feature tensors.

    Args:
        features (list of torch.Tensor): List of feature tensors to be combined.
        weights (torch.Tensor): Weights for each feature tensor, shape (num_features,).

    Returns:
        torch.Tensor: Weighted sum of the feature tensors.
    """
    # Normalize weights using softmax
    normalized_weights = F.softmax(weights, dim=0)

    # Compute the weighted sum
    weighted_features = [w * f for w, f in zip(normalized_weights, features)]
    return sum(weighted_features)