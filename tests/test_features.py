import os, random, numpy as np, torch
def _seed():
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

import pandas as pd, torch
from src.GNNs.common.features import build_features
from torch_geometric.data import HeteroData

def test_feature_shapes_minimal():
    places = pd.DataFrame([{"id":"r1","lat":34.0,"lon":-118.0}])
    reviews = pd.DataFrame([], columns=["id","user_id","restaurant_id","text","created_at","rating"])
    rid_map={"r1":0}; uid_map={}; vid_map={}
    data = HeteroData()
    data["restaurant"].num_nodes = 1
    data["user"].num_nodes = 0
    data["review"].num_nodes = 0

    build_features(data, places, reviews, rid_map, uid_map, vid_map)
    assert data["restaurant"].x.shape == (1,2)  # lat,lon
    assert hasattr(data["user"], "x") and data["user"].x.shape == (0,1)
