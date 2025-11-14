import os, random, numpy as np, torch
def _seed():
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

import pandas as pd, torch
from src.GNNs.common.geo import geo_rings

def test_geo_rings_radius_effect():
    places = pd.DataFrame([
        {"id":"r1","lat":34.000,"lon":-118.000},
        {"id":"r2","lat":34.001,"lon":-118.001},  # ~0.14 km
        {"id":"r3","lat":34.010,"lon":-118.010},  # ~1.4 km
    ])
    rid_map = {rid:i for i, rid in enumerate(places["id"])}
    ei_small = geo_rings(places, radius_miles=0.2)
    ei_big   = geo_rings(places, radius_miles=2.0)
    # With small radius, only r1<->r2 connects; with big radius, all close pairs
    assert ei_small.size(1) > 0
    assert ei_big.size(1) >= ei_small.size(1)
    assert ei_small.dtype == torch.long
