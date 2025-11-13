#This is the glue that gets the location and the query to:
#1 - Find restauraunts in radius
#2 - load GraphSAGE embeddings (or run GraphSAGE if missing)
#3 - runs GCN over the local subgraph
#4 - compute accessiblity + quality + GAT for review
#5 - fuses final_score, return top-K, and writes via DataSource.upser_scores

import pandas as pd
import numpy as np
import torch
import yaml
from ..common.infer import load_embeddings
from ..common.graph_build import build_graph
from ..common.geo import haversine_m
from ...fusion.late_fusion import fuse

def rank(lat, lon, query, topk=10, cfg_path="configs/mvp.yaml"):
    #Build graph and get embeddings
    cfg = yaml.safe_load(open(cfg_path))
    data_source, data, maps = build_graph(cfg)
    rest_emb = load_embeddings("restaurant_emb.pt") #[N,D] (remember M matrix from common.geo -> graph matrix)

    #Local filter by radius
    places = data_source.fetch_places()
    coords = places[["lat", "lon"]].values
    dists = haversine_m(np.array([[lat, lon]]), coords).flatten()
    keep = np.where(dists <= cfg["data"]["geo_radius_m"])[0]

    #stub: scores from embeddings cosine to query vector (if use embed query)
    #accessibility and quality from reviews added here (GAT)
    base = torch.rand(len(keep))    #MVP placeholder

    df = places.iloc[keep].copy()
    df["accessibility_score"] = base.numpy()
    df["quality_score"] = 1 - df["accessibility_score"]
    df["final_score"] = fuse(df["accessibility_score"].values, df["quality_score"].values)
    df = df.sort_values("final_score", ascending=False).head(topk)

    data_source.upsert_scores(df[["id", "accessibility_score", "quality_score", "final_score"]])
    return df[["name", "final_score", "accessibility_score", "quality_score", "lat", "lon"]]