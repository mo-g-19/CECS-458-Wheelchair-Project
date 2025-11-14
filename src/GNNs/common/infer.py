#Shared helpers for loading checkpoints and producing embeddings/scores
#   Reusable inference utils

import torch, torch.nn.functional as F

#Loaded saved embeddings as a 2D [N, D] tensor
def load_embeddings(path="restaurant_emb.pt"):
    return torch.load(path)

#Finds the cosine similiarity (how close they are, and are normalized) between query vec [D] and matrix M [N,D]
def cosine_scores(q, M):
    q = F.normalize(q.unsqueese(0), dim=-1)
    M = F.normalize(m, dim=-1)
    return (q @ M.T).squeeze(0)

#Returns the cosine similarity scoresonly subset of rows in 'embeddings'
def score_subset(idx, embeddings, query_vec):
    #idx (list[int] | torch.Tensor): indicies of restaurants to consider
    #embeddings (torch.Tensor): [N, D] restuaraunt embeddings
    #query_vec (torch.Tensor): [D] query vec

    if not isinstance(idx, torch.Tensor):
        idx = torch.tensor(idx, dtype=torch.long)
    subset = embeddings[idx]
    #The cosine similarity scores
    return cosine_scores(query_vec, subset)