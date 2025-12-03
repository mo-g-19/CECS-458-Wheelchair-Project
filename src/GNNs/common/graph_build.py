"""
This builds a heterogenous graph from places.csv and reviews
    It does this by turning the raw tables (from data collection) into a heterogenous graph (users, reviews, resturaunts) with initial features
"""

import torch
from torch_geometric.data import HeteroData
from .data_sources import LocalJSON
from .features import build_features
from .geo import geo_rings

#There is only one function: which is supposed to build the graph from the table
def build_graph(cfg):
    #Load the tables using LocalJSON from data_sources.py
    data_source = LocalJSON(cfg['data']['path'])
    places = data_source.fetch_places()
    reviews = data_source.fetch_reviews()
    data = HeteroData()

    #node id maps
    #restuaurnt
    rid_map = {rid:i for i, rid in enumerate(places["id"].tolist())}
    #User: from the review
    uid_map = {uid:i for i, uid in enumerate(reviews["user_id"].dropna().unique())}
    #review: from the text of review
    vid_map = {vid:i for i, vid in enumerate(reviews["id"].dropna().unique())}

    #edges: user -> review or review -> resturaunt
    #indexing by mapped ids
    #geo neighbor edgse for resturaunt to resturaunt
    data_cfg = cfg.get("data", {})
    radius_km = data_cfg.get("geo_radius_km")
    if radius_km is None:
        radius_m = data_cfg.get("geo_radius_m")
        radius_km = radius_m / 1000 if radius_m is not None else 5
    r2r_edge_index = geo_rings(places, radius_km=radius_km)
    #Making an index based on the edges
    data["resturaunt", "near", "resturaunt"].edge_index = r2r_edge_index

    #features
    build_features(data, places, reviews, rid_map, uid_map, vid_map)

    #Returning the: local table, the table sorted by torch, and a dictionary of resturaunt, user, and review
    return data_source, data, {"rest": rid_map, "user": uid_map, "rev ": vid_map}
