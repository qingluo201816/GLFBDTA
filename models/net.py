import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch.nn.parameter import Parameter

from torch.nn import Embedding, Linear
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops

import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from models.layers import MLP, AtomEncoder, Drug_PNAConv, Protein_PNAConv, DrugProteinConv, PosLinear, GCNCluster, \
    SAGECluster, SGCluster, APPNPCluster, dropout_edge
from layer.multi_scale_fusion import BilinearInteractionFusion,group_features_by_batch,weighted_sum
from copy import deepcopy
## for drug pooling
from models.drug_pool import MotifPool
## for cluster
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph, softmax, \
    add_remaining_self_loops

from models.protein_pool import dense_mincut_pool, dense_dmon_pool, simplify_pool
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric
# from models.gps_layer import GPSLayer
from torch_geometric.data import Batch
EPS = 1e-15
import math
from layer.atom_encoder import AtomGlobalSeqEncoder

class net(torch.nn.Module):
    def __init__(self, mol_deg, prot_deg,
                 mol_in_channels=43, prot_in_channels=33, prot_evo_channels=1280,
                 hidden_channels=200,
                 pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,
                 K=[5, 10, 20],
                 t=1,
                 # training
                 heads=5,
                 dropout=0,
                 dropout_attn_score=0.2,
                 drop_atom=0,
                 drop_residue=0,
                 dropout_cluster_edge=0,
                 gaussian_noise=0,
                 # objective
                 regression_head=True,
                 device='cuda:0'):
        super(net, self).__init__()
        self.total_layer = total_layer
        self.regression_head = regression_head
        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels
        self.dropout = dropout
        self.drop_atom = drop_atom
        self.drop_residue = drop_residue
        self.gaussian_noise = gaussian_noise
        self.dropout_cluster_edge = dropout_cluster_edge
        self.device = device

        self.atom_type_encoder = Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)


        self.clique_encoder = Embedding(4, hidden_channels)



        self.prot_evo = MLP([prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)



        self.mol_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()
        self.inter_convs = torch.nn.ModuleList()


        self.mol_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()



        self.num_cluster = K
        self.t = t
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.mol_norms = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()

        self.atom_lins = torch.nn.ModuleList()
        self.residue_lins = torch.nn.ModuleList()

        self.c2a_mlps = torch.nn.ModuleList()
        self.c2r_mlps = torch.nn.ModuleList()

        # self.group_enhancer = ProteinClusterEnhancer1D(groups=10)



        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels, edge_channels=hidden_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.prot_convs.append(GPSLayer(dim_h=200,
                                            local_gnn_type='PNA',
                                            global_model_type='Mamba',
                                            num_heads=4,
                                            pna_degrees=[1, 2, 3, 4],
                                            equivstable_pe=False,
                                            dropout=0.1,
                                            attn_dropout=0.1,
                                            layer_norm=True,
                                            batch_norm=False,
                                            prot_deg=prot_deg,
                                            ))



            # self.cluster.append(SAGECluster([hidden_channels, hidden_channels*2, self.num_cluster[idx]],
            #                     in_norm=True, add_self_loops=True, root_weight=False))
            # self.cluster.append(MLP([hidden_channels*2, hidden_channels*2, self.num_cluster[idx]]))
            self.cluster.append(GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]], in_norm=True))
            # self.cluster.append(SGCluster(hidden_channels, self.num_cluster[idx], K=2))
            # self.cluster.append(APPNPCluster(hidden_channels, self.num_cluster[idx],a=0.1, K=10))



            self.inter_convs.append(
                BilinearInteractionFusion(v_dim=hidden_channels, q_dim=hidden_channels, h_dim=hidden_channels, heads=heads, act='ReLU',
                         dropout=0.2, k=3))



            self.mol_pools.append(MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom))
            self.mol_norms.append(torch.nn.LayerNorm(hidden_channels))

            self.prot_norms.append(torch.nn.LayerNorm(hidden_channels))
            self.atom_lins.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.residue_lins.append(Linear(hidden_channels, hidden_channels, bias=False))

            self.c2a_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))
        self.atom_attn_lin = PosLinear(heads * total_layer, 1, bias=False,
                                       init_value=1 / heads)
        self.residue_attn_lin = PosLinear(heads * total_layer, 1, bias=False,
                                          init_value=1 / heads)
        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)


        self.reg_out = MLP([hidden_channels * 2, hidden_channels, 1])

        self.atom_encoder = AtomGlobalSeqEncoder(hidden_channels)  # 使用新的 AtomEncoder

        # self.atom_bilstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, num_layers=1,
        #                            batch_first=True, bidirectional=True)
        #
        # self.adjust_layer = nn.Linear(hidden_channels * 2, hidden_channels)
        #
        # self.atom_fusion_mlp = MLP(
        #     dims=[2 * hidden_channels, hidden_channels],
        #     out_norm=True
        # )
    def reset_parameters(self):

        self.atom_feat_encoder.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()

        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()


            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()

            self.cluster[idx].reset_parameters()

            self.mol_pools[idx].reset_parameters()
            self.mol_norms[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()

            self.inter_convs[idx].reset_parameters()

            self.atom_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()

            self.c2a_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()

        self.atom_attn_lin.reset_parameters()
        self.residue_attn_lin.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()

        # self.group_enhancer.reset_parameters()
        self.atom_encoder.reset_parameters()
        self.reg_out.reset_parameters()

    def forward(self,
                mol_x, mol_x_feat, bond_x, atom_edge_index,
                clique_x, clique_edge_index, atom2clique_index,
                residue_x, residue_evo_x, residue_edge_index, residue_edge_weight,
                mol_batch=None, prot_batch=None, clique_batch=None,
                save_cluster=False):


        reg_pred = None
        mol_pool_feat = []
        prot_pool_feat = []

        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)


        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)

        atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(
            mol_x_feat)
        # atom_x_bilstm = self.encode_atom_bilstm(atom_x, mol_batch)
        atom_x_bilstm = self.atom_encoder.encode_atom_bilstm(atom_x, mol_batch)

        clique_x = self.clique_encoder(clique_x.squeeze())


        spectral_loss = torch.tensor(0.).to(self.device)
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)


        clique_scores = []
        residue_scores = []
        layer_s = {}
        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
            atom_x_concat = torch.cat([atom_x, atom_x_bilstm], dim=1)
            atom_x = self.atom_encoder.atom_fusion_mlp(atom_x_concat)



            batch = Batch(
                x=residue_x,
                edge_index=residue_edge_index,
                edge_attr=residue_edge_attr,
                batch=torch.zeros(residue_x.size(0), dtype=torch.long, device=residue_x.device)
            )

            residue_x = self.prot_convs[idx](batch).x



            drug_x, clique_x, clique_score = self.mol_pools[idx](atom_x, clique_x, atom2clique_index, clique_batch,
                                                                 clique_edge_index)


            drug_x = self.mol_norms[idx](drug_x)

            clique_scores.append(clique_score)


            dropped_residue_edge_index, _ = dropout_edge(residue_edge_index, p=self.dropout_cluster_edge,
                                                         force_undirected=True, training=self.training)

            s = self.cluster[idx](residue_x, dropped_residue_edge_index)

            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)


            if save_cluster:
                layer_s[idx] = s

            s, _ = to_dense_batch(s, prot_batch)

            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            cluster_mask = residue_mask
            cluster_drop_mask = None
            if self.drop_residue != 0 and self.training:
                _, _, residue_drop_mask = dropout_node(residue_edge_index, self.drop_residue, residue_x.size(0),
                                                       prot_batch,
                                                       self.training)
                residue_drop_mask, _ = to_dense_batch(residue_drop_mask.reshape(-1, 1),
                                                      prot_batch)
                residue_drop_mask = residue_drop_mask.squeeze()
                cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()



            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, s, cluster_mask,
                                                                           cluster_drop_mask)
            # spectral_loss += sp_loss
            ortho_loss += o_loss
            cluster_loss += cl_loss

            cluster_x = self.prot_norms[idx](cluster_x)

            cluster_x_ban = cluster_x


            # 处理团特征
            clique_x_ban, clique_x_ban_lengths = group_features_by_batch(clique_x, clique_batch)
            # 处理原子特征
            atom_x_ban, atom_x_ban_lengths = group_features_by_batch(atom_x, mol_batch)


            batch_size = s.size(0)

            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)


            # cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)
            #
            #
            # p2m_edge_index = torch.stack([torch.arange(batch_size * self.num_cluster[idx]),
            #                               torch.arange(batch_size).repeat_interleave(self.num_cluster[idx])]
            #                              ).to(self.device)


            clique_x, cluster_from_clique, clique_inter_attn = self.inter_convs[idx](clique_x_ban, cluster_x_ban, clique_x_ban_lengths)
            atom_x_ban, cluster_from_atom, atom_inter_attn = self.inter_convs[idx](
                atom_x_ban, cluster_x_ban, atom_x_ban_lengths
            )

            drug_x_ban, cluster_from_drug, drug_inter_attn = self.inter_convs[idx](drug_x, cluster_x_ban, clique_x_ban_lengths,drug=True)

            features = [cluster_from_clique, cluster_from_atom, cluster_from_drug]
            # Perform weighted sum
            self.gate = nn.Parameter(torch.randn(3))
            cluster_x = weighted_sum(features, self.gate)
            row, col = atom2clique_index


            atom_x = atom_x + F.relu(self.atom_lins[idx](
                scatter(clique_x[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean')))


            atom_x = atom_x + self.c2a_mlps[idx](atom_x)

            atom_x = F.dropout(atom_x, self.dropout, training=self.training)




            residue_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)



            residue_x = residue_x + F.relu(self.residue_lins[idx]((s @ residue_hx)[residue_mask]))

            residue_x = residue_x + self.c2r_mlps[idx](residue_x)

            residue_x = F.dropout(residue_x, self.dropout, training=self.training)

            inter_attn = (s @ clique_inter_attn)[residue_mask]

            residue_scores.append(inter_attn)

            atom_x = self.mol_gn2[idx](atom_x, mol_batch)

            residue_x = self.prot_gn2[idx](residue_x, prot_batch)



        row, col = atom2clique_index

        clique_scores = torch.cat(clique_scores, dim=-1)

        atom_scores = scatter(clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean')

        atom_score = self.atom_attn_lin(atom_scores)

        atom_score = softmax(atom_score, mol_batch)

        mol_pool_feat = global_add_pool(atom_x * atom_score, mol_batch)


        residue_scores = torch.cat(residue_scores, dim=-1)

        residue_score = softmax(self.residue_attn_lin(residue_scores), prot_batch)

        prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)


        mol_pool_feat = self.mol_out(mol_pool_feat)

        prot_pool_feat = self.prot_out(prot_pool_feat)

        mol_prot_feat = torch.cat([mol_pool_feat, prot_pool_feat], dim=-1)



        reg_pred = self.reg_out(mol_prot_feat)



        attention_dict = {
            'residue_final_score': residue_score,
            'atom_final_score': atom_score,
            'clique_layer_scores': clique_scores,
            'residue_layer_scores': residue_scores,
            'drug_atom_index': mol_batch,
            'drug_clique_index': clique_batch,
            'protein_residue_index': prot_batch,
            'mol_feature': mol_pool_feat,
            'prot_feature': prot_pool_feat,
            'interaction_fingerprint': mol_prot_feat,
            'cluster_s': layer_s

        }

        return reg_pred, spectral_loss, ortho_loss, cluster_loss, attention_dict


    def temperature_clamp(self):
        pass

    def encode_atom_bilstm(self, atom_x, mol_batch):
        atom_x_split = unbatch(atom_x, mol_batch, dim=0)
        seq_lengths = [x.size(0) for x in atom_x_split]
        max_seq_length = max(seq_lengths)
        batch_size = len(seq_lengths)
        feature_dim = atom_x.shape[1]

        padded_atom_x = torch.zeros(
            batch_size,
            max_seq_length,
            feature_dim,
            dtype=torch.float32,
            device=atom_x.device
        )

        start_idx = 0
        for i, length in enumerate(seq_lengths):
            end_idx = start_idx + length
            padded_atom_x[i, :length] = atom_x[start_idx:end_idx]
            start_idx = end_idx

        packed_atom_x = nn.utils.rnn.pack_padded_sequence(
            padded_atom_x,
            lengths=torch.tensor(seq_lengths),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.atom_bilstm(packed_atom_x)
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        atom_x_bilstm_split = [unpacked_out[i, :length, :] for i, length in enumerate(seq_lengths)]
        atom_x_bilstm = torch.cat(atom_x_bilstm_split, dim=0)

        atom_x_out = self.adjust_layer(atom_x_bilstm)
        return atom_x_out

    def connect_mol_prot(self, mol_batch, prot_batch):
        mol_num_nodes = mol_batch.size(0)
        prot_num_nodes = prot_batch.size(0)
        mol_adj = mol_batch.reshape(-1, 1).repeat(1, prot_num_nodes)
        pro_adj = prot_batch.repeat(mol_num_nodes, 1)

        m2p_edge_index = (mol_adj == pro_adj).nonzero(as_tuple=False).t().contiguous()

        return m2p_edge_index



    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):


        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or pn.endswith('mean_scale'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif 'group_enhancer' in mn:
                    if pn.endswith('weight'):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        for pn in param_dict.keys() - union_params:
            no_decay.add(pn)
        union_params = decay | no_decay

        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer


def _rbf(D, D_min=0., D_max=1., D_count=200, device='cpu'):


    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device))
    D_mu = torch.linspace(D_min, D_max, D_count,
                          device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def unbatch(src, batch, dim: int = 0):
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(edge_index, batch):
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def dropout_node(edge_index, p, num_nodes, batch, training):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []

    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)

    node_mask = torch.cat(node_mask_list)

    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask


class ProteinClusterEnhancer1D(nn.Module):
    def __init__(self, groups=32):
        super(ProteinClusterEnhancer1D, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        nn.init.constant_(self.bias, 1)

    def forward(self, x):
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups,
                    -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x