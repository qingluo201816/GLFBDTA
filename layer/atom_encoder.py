import torch
import torch.nn as nn
from models.layers import MLP  # 确保 MLP 类在正确的模块中
from torch_geometric.utils import degree

def unbatch(src, batch, dim: int = 0):
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)

class AtomGlobalSeqEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super(AtomGlobalSeqEncoder, self).__init__()
        self.atom_bilstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, num_layers=1,
                                   batch_first=True, bidirectional=True)
        self.adjust_layer = nn.Linear(hidden_channels * 2, hidden_channels)
        self.atom_fusion_mlp = MLP(
            dims=[2 * hidden_channels, hidden_channels],
            out_norm=True
        )

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

    def reset_parameters(self):
        self.atom_bilstm.reset_parameters()
        self.adjust_layer.reset_parameters()
        self.atom_fusion_mlp.reset_parameters()


