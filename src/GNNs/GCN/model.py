#This is a light GCN to refine the embeddings on a local subgraph

import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(layers-1)])
        self.dropout = nn.Dropout(dropout)

        def forward(self, data):
            x = data["restaurant"].x
            edge_index = data["restaurant", "near", "restaurant"].edge_index
            for conv in self.convs:
                x = self.dropout(conv(x, edge_index)).relu()
            return x