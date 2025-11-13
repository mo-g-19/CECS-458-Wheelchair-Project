#Adjacency matrix and self-loops (so each node sees itself\)
#Reminder: Adjacency matrix -> shows a connection between nodes
#   Ex: if A[0][1] = 1, node 0 is connected to node 1
#Adding the identity matrix to the adjacency matrix because there is a connection of node 0 to node 0 (its the same one)
#   A_hat = A + I

#D_hat: degree matrix of A_hat (how many in/out connections each node has)
#Normalizing -> allow smaller restuaruants to still show up in the aggregation
#This is the "smoothing out"; every node's information is treated equally
#   Taking the inverse square root of D_hat
#         Simple normaization (inverse): average of neighbors (not sum), but not symmetric
#         Symmetric normalization (inverse square root): operator unbiased, each edge influence is balanced
#   D_hat^(-1/2) * A_hat * D_hat^(-1/2)

#Now together:
#   Aggregate neighborhood featurs (D_hat^(-1/2) * A_hat * D_hat^(-1/2)) * H^(l)
#   Transform by trainable weights

#   H^(l+1) = ReLU(() D_hat^(-1/2) * A_hat * D_hat^(-1/2)) * H^(l) * W^(l))
#   Or a different activation function instead of ReLU

import torch
import torch.nn as nn
import torch.nn.fnctional as F
from torch_geometric.nn import HetroConv, GCNConv, Linear
from torch_geometric.nn import HGTLoader, NeighborLoader    #Choose 1
from torch_geometric.data import HeteroData
from typing import Dict, Tuple

#2-layer HeteroGCN *Heterogeneous Graph Convolutional Network*
class HeteroGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network (HeteroGCN) model for node regression tasks.
        ('user', 'reviews', 'restaraunt') and reverse
        ('restauraunt, 'near', 'restauraunt')
    Accpets arbitary input feature sizes per node type
    """

    def __init__(
            self,
            metadata: Tuple,    #Graph metadata (node and edge types)
            in_channels: Dict[str, int],   #Input feature sizes per node type (user: d_u, restaurant: d_r)
            hidden_channels: int = 128,
            out_channels: int = 128,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout

        #Per-type input projection in the hidden so it concatenates the embeddings and raw features
        self.in_lin = nn.ModleDict({
            ntype: Linear(in_channels[ntype], hidden_channels, bias=False)
            for ntype in metadata[0]
        })

        #First HeteroGCN layer: hetero message passing with GCNConv per relation
        conv1_dict = {
            etype: GCNConv((-1,-1), hidden_channels, add_self_loops=True, normalize=True)
            for etype in metadata[1]
        }
        self.conv1 = HetroConv(conv1_dict, aggr='sum')   #Aggregate by summing up messages from different edge types

        #Second HeteroGCN layer
        conv2_dict = {
            etype: GCNConv((-1,-1), hidden_channels, add_self_loops=True, normalize=True)
            for etype in metadata[1]
        }

        self.conv2 = HetroConv(conv2_dict, aggr='sum')

        #Optional output per node (focus on restuaurants)
        self.out_lin = nn.ModuleDict({
            ntype: Linear(hidden_channels, out_channels, bias=False)
            for ntype in metadata[0]
            })

def forward(self, x_dict, edge_index_dict):
        #Type specific input projection
        x_dict = {
            ntype: self.in_lin[ntype](x)
            for ntype, x in x_dict.items()
        }

        #Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k,v in x_dict.items()}

        #Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.out_lin[k](v) for k,v in x_dict.items()}

        return x_dict

#Example: building graph
def build_hetero_example() -> HeteroData:
        """
        This gets replaced with the real loader. This shows the schema.
        
        Assume created:
        
        """




        #Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        #Regression heads for each node type (predict scalar value per node)
        self.heads = nn.ModuleDict()
        for node_type in metadata[0]:   #metadata[0] contains node types
            self.heads[node_type] = nn.Linear(out_channels, 1)   #Output single value per node
    