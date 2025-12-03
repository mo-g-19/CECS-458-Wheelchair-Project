#The purpose of this file is the LocalCSV, and it is an optional adapter for outside storage

"""
This is a thin interface to switch between local CSV files and Supabase as the data source
for places, reviews, and upserting scores.
"""

from dataclasses import dataclass
import json
import os
import pandas as pd

@dataclass
class DataSource:
    #The ... will be replaced by the Subapabase or other data source implementations in the future
    def fetch_places(self) -> pd.DataFrame: ...
    def fetch_reviews(self) -> pd.DataFrame: ...
    def upsert_scores(self, df_scores: pd.DataFrame): ...

# Legacy CSV implementation (kept for compatibility with tests/fixtures)
class LocalCSV(DataSource):
    def __init__(self, root="data"):
        self.root = root
    
    def fetch_places(self):  # expects places.csv (id,name,lat,lon,categories,json_attrs?)
        return pd.read_csv(os.path.join(self.root, "places.csv"))
    
    def fetch_reviews(self):  # optional mock file
        p = os.path.join(self.root, "mock_reviews.csv")
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=[
            "id","user_id","restaurant_id","text","created_at","rating"
        ])
    
    #Writes results locally now, and later write to Supabase
    def upsert_scores(self, df_scores):  # no-op for MVP
        os.makedirs(os.path.join(self.root, "out"), exist_ok=True)
        df_scores.to_csv(os.path.join(self.root, "out","gnn_scores.csv"), index=False)

#MVP implementation that reads/writes from local JSON caches (Yelp + community)
class LocalJSON(DataSource):
    def __init__(self, root="src/data"):
        self.root = root
        self.yelp_path = os.path.join(root, "yelp_cache.json")
        self.community_path = os.path.join(root, "community.json")

    def _load_yelp_cache(self):
        if not os.path.exists(self.yelp_path):
            raise FileNotFoundError(f"Missing Yelp cache at {self.yelp_path}")
        with open(self.yelp_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_community(self):
        if not os.path.exists(self.community_path):
            return {}
        with open(self.community_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_places(self):
        cache = self._load_yelp_cache()
        rows = []
        for biz_id, p in cache.items():
            rows.append({
                "id": biz_id,
                "name": p.get("name"),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "categories": p.get("categories"),
                "rating": p.get("rating"),
                "review_count": p.get("review_count"),
                "url": p.get("url"),
                "city": p.get("city"),
                "cuisine": p.get("cuisine"),
            })
        return pd.DataFrame(rows)
    
    def fetch_reviews(self):
        cache = self._load_yelp_cache()
        community = self._load_community()
        rows = []

        # Yelp snippets
        for biz_id, p in cache.items():
            for idx, text in enumerate(p.get("reviews") or []):
                rows.append({
                    "id": f"{biz_id}_yelp_{idx}",
                    "user_id": None,
                    "restaurant_id": biz_id,
                    "text": text,
                    "created_at": None,
                    "rating": None
                })

        # Community-submitted reviews
        for biz_id, payload in community.items():
            for idx, text in enumerate(payload.get("reviews") or []):
                rows.append({
                    "id": f"{biz_id}_comm_{idx}",
                    "user_id": "community",
                    "restaurant_id": biz_id,
                    "text": text,
                    "created_at": None,
                    "rating": None
                })

        return pd.DataFrame(rows, columns=[
            "id","user_id","restaurant_id","text","created_at","rating"
        ])
    
    
    #Writes results locally now, and later write to Supabase
    def upsert_scores(self, df_scores):  # no-op for MVP
        os.makedirs(os.path.join(self.root, "out"), exist_ok=True)
        df_scores.to_csv(os.path.join(self.root, "out","gnn_scores.csv"), index=False)
