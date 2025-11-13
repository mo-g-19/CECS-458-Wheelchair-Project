import os, random, numpy as np, torch
from src.GNNs.common.data_sources import LocalCSV
import pandas as pd, os

def _seed():
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)


def test_localcsv_reads_fixture(tmp_path):
    # copy fixtures to tmp
    root = tmp_path/"data"; root.mkdir()
    (root/"places.csv").write_text("id,name,lat,lon\nr1,A,34.0,-118.0\nr2,B,34.01,-118.01\n")
    (root/"mock_reviews.csv").write_text("id,user_id,restaurant_id,text,created_at,rating\nv1,u1,r1,good ramp,2024-01-01,5\n")

    ds = LocalCSV(str(root))
    places = ds.fetch_places()
    reviews = ds.fetch_reviews()

    assert list(places.columns)[:4] == ["id","name","lat","lon"]
    assert len(places) == 2 and len(reviews) == 1

def test_upsert_scores_creates_output(tmp_path):
    root = tmp_path/"data"; root.mkdir()
    (root/"places.csv").write_text("id,name,lat,lon\nr1,A,34.0,-118.0\n")
    ds = LocalCSV(str(root))
    ds.upsert_scores(pd.DataFrame([{"id":"r1","final_score":0.9,"accessibility_score":0.8,"quality_score":0.7}]))
    assert os.path.exists(root/"out"/"gnn_scores.csv")