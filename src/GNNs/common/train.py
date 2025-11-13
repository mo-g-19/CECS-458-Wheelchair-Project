"""
stage-aware trainer: sage | gcn"""
#This is the trainer for GraphSAGE or GCN depending on stage
#     Why does this use GraphSAGE or GCN?
#           GraphSAGE: global, inductive incoder
#               This is the base and it trains once on the full hetero graph
#           GCN: local, transductive encoder
#               This is the subgraph on the radius-filtere and smooths only nearby

#           No GAT?: because trying to mimize complexity here, so it is trained "offline"
#               When train GAT -> freeze GraphSAGE embeddings and train based on review aggregates
#Key: This is for local refinement, not global; GraphSAGE can handle new nodes (inductive property)
#Very small number of epochs since this is just local smoothing

import yaml, torch, torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from .graph_build import build_graph
from ..GraphSage.model import GraphSAGE
from ..GCN.model import GCN

#This is training the models
def train(cfg_path: str, stage: str):
    #Load config
    cfg = yaml.safe_load(open(cfg_path))
    data_source, data, maps = build_graph(cfg)


    #Select model based on stage (SAGE or GCN)
    if stage == "sage":
        model = GraphSAGE(data["restaurant"].x.size(1), **cfg["model"]["sage"])
        #self-supervised or simple supervised (minimal viable product: identity target / constrastive skipped)

    elif stage == "gcn":
        model = GCN(data["restaurant"].x.size(1), **cfg["model"]["gcn"])
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'sage' or 'gcn'.")
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg["model"][stage]["lr"])
    model.train()
    for epoch in range(cfg["model"][stage]["epochs"]):
        #Minimal viable product: toy loss to make weights non-trivial (e.g., graph smoothing)
        out = model(data) #return embeddings per restuaurant
        loss = (out**2).mean()  #Toy loss: minimize magnitude of embeddings
        opt.zero_grad(); loss.backward(); opt.step()

    #Save model and embeddings
    torch.save(model.state_dict(), f"{stage}.ckpt")
    torch.save(out.detach(), "restaurant_emb.pt")
    return f"{stage}.ckpt"
