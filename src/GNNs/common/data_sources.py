""""
The purpose of this file is the LocalCSV, and it is an optional adapter for outside storage"""

from dataclasses import dataclass
import pandas as pd
import os

@dataclass
class DataSource:
    def fetch_places(self) -> pd.DataFrame: ...
    def fetch_reviews(self) -> pd.DataFrame: ...
    def upsert_scores(self, df_scores: pd.DataFrame): ...

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
    def upsert_scores(self, df_scores):  # no-op for MVP
        os.makedirs(os.path.join(self.root, "out"), exist_ok=True)
        df_scores.to_csv(os.path.join(self.root, "out","gnn_scores.csv"), index=False)