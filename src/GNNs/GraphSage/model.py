#This is the small GraphSage model with 2 layers
#   Why two layers?
#       Layer 1: First take all the restuaruants to see the ones connected
#       Layer 2: Find the restuaraunts nearby (2-hops) the ones where there is 1 restuaruant in between
#Reminds me of the 7-person connection to all people in the world

import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero

class _SAGE(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        #Finding the nodes neighboors and 1 connection away
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden)] + [SAGEConv(hidden, hidden) for _ in range(layers-1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            #Just use ReLU because trying to min computational cost
            h = self.droput(conv(h, edge_index)).relu()
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.sage = _SAGE(in_dim, hidden, layers, dropout)

        def forward(self, data):
            #Minimal Viable Product: only restauraunt graph (near edges)
            x = data["restaurant"].x
            edge_index = data["restaurant", "near", "restaurant"].edge_index
            return self.sage(x, edge_index)
        


            
