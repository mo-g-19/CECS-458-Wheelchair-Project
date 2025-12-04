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
from ..common.graph_build import build_graph
from ..common.geo import haversine_m, MI_TO_KM
from ...fusion.late_fusion import fuse
from ...api.community_json import get_flags

def rank(lat, lon, query, topk=10, cfg_path="src/configs/mvp.yaml"):
    #Build graph and get embeddings
    cfg = yaml.safe_load(open(cfg_path))
    data_source, data, maps = build_graph(cfg)

    #Local filter by radius
    places = data_source.fetch_places()
    coords = places[["lat", "lon"]].values
    # Distance is computed in miles; convert to km for comparison with cfg
    dists_km = haversine_m(np.array([[lat, lon]]), coords).flatten() * MI_TO_KM
    data_cfg = cfg.get("data", {})
    radius_km = data_cfg.get("geo_radius_km")
    if radius_km is None:
        radius_m = data_cfg.get("geo_radius_m")
        radius_km = radius_m / 1000 if radius_m is not None else 5
    keep = np.where(dists_km <= radius_km)[0]

    df = places.iloc[keep].copy()
    df["quality_score"] = (df["rating"].fillna(3) / 5).clip(0, 1)
    df["accessibility_score"] = df.apply(
        lambda row: _access_from_flags(str(row.get("id", "")), str(row.get("name", ""))), axis=1
    )
    df["final_score"] = fuse(df["accessibility_score"].values, df["quality_score"].values)
    df = df.sort_values("final_score", ascending=False).head(topk)

    data_source.upsert_scores(df[["id", "accessibility_score", "quality_score", "final_score"]])
    return df[["id", "name", "rating", "final_score", "accessibility_score", "quality_score", "lat", "lon"]]

def _access_from_flags(biz_id: str, name: str = ""):
    flags = get_flags(biz_id) or get_flags(name)
    if not flags:
        return 0.5
    vals = [v for v in flags.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.5
    return sum(1 for v in vals if v > 0) / len(vals)
