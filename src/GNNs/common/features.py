#This is where the node/edge features are defined
#     Does this by computing simple, fast features for each node type

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def build_features(data, places, reviews, rid_map, uid_map, vid_map):
    #resturaunt features
    #First location
    lat = torch.tensor(places["lat"].values, dtype=torch.float32).unsqueeze(1)
    lon = torch.tensor(places["lon"].values, dtype=torch.float32).unsqueeze(1)
    #Then making sure distance is small for Minimal Viable Product
    restuaunt_feat = torch.cat([lat, lon], dim=1)
    data["resturaunt"].x = restuaunt_feat

    #User features (activity count): inspired by CECS 427 and hub/authority scores
    user_counts = reviews.groupby("user_id").size()
    user_x = torch.zeros((len(uid_map), 1))
    for uid, i in uid_map.items():
        user_x[i, 0] = user_counts.get(uid, 0)
    data["user"].x = user_x

    #Review features (text embeddings)
    #Make sure there are actual words in the review
    if len(reviews) > 0 and "text" in reviews:
        model = SentenceTransformer('all-MiniLM-L6-v2')    #The all-MiniLM-L6-v2 model from search.py
        embs = model.encode(reviews["text"].fillna("").tolist())
        review_texts = torch.tensor(np.asarray(embs), dtype=torch.float32)
    else:
        #Placeholder of zeros if no reviews
        review_texts = torch.zeros((len(rid_map), 0))
    data["review"].x = review_texts
