import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter


class SLMPConv(nn.Module):
    """Smooth-less Message Passing"""

    def __init__(self, in_feats, out_feats, aggregator_type='max', alpha=1.0,
                 with_bias=False, with_neigh_norm=True, separate_encoder=True, **kwargs):
        super(SLMPConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        assert aggregator_type in ['max', 'min', 'mean', 'sum']
        self.aggr = aggregator_type
        self.with_bias = with_bias
        self.with_neigh_norm = with_neigh_norm
        self.alpha = torch.tensor(alpha)

        self.fc1 = nn.Linear(in_feats, out_feats, bias=with_bias)
        if separate_encoder:
            self.fc2 = nn.Linear(in_feats, out_feats, bias=with_bias)
        else:
            self.fc2 = self.fc1

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        if self.with_bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x, edge_index):
        h_src, h_dst = x, x
        n_nodes = h_src.size(0)

        h_src = self.fc2(h_src)

        out_degrees = degree(edge_index[0], num_nodes=n_nodes)
        h_src = h_src * torch.pow(out_degrees + 1.0, -0.5).unsqueeze(-1)

        # Normalization
        if self.with_neigh_norm:
            h_src = F.normalize(h_src, p=2, dim=1)

        # Message aggr.
        h_src = scatter(h_src[edge_index[0]], edge_index[1], dim=0, dim_size=n_nodes, reduce=self.aggr)

        outs = self.alpha * self.fc1(h_dst) + h_src
        return outs
