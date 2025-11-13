"""
This builds a heterogenous graph from places.csv and reviews
    It does this by turning the raw tables (from data collection) into a heterogenous graph (users, reviews, resturaunts) with initial features
"""

import torch
import torch_geometric.data import HeteroData
from .data_source import LocalCSV 
from .features import build_features
from .geo import geo_rings

#There is only one function: which is supposed to build the graph from the table
def build_graph(cfg):
    #Load the tables using LocalCSV from data_sources.py
    data_source = LocalCSV(cfg['data']['path'])
    places_df = data_source.load_places()
    reviews_df = data_source.load_reviews()
    data = HeteroData()

    #node id maps
    #restuaurnt
    rid_map = {rid:i for i, rid in enumerate(places["id"].tolist())}
    #User: from the review
    uid_map = {uid:i for i, uid in enumerate(reviews["user_id"].dropna().unique())}
    #review: from the text of review
    vid_map = {vid:i for i, vid in enumerate(reviews["id"].dropna().unique())}