#Purpsoe: finding nearby resturaunts based on geospatial data
"""
Review (what are these geospatial helpers and why do we need them?)
     haversine: to calculate distance between two lat/lon points
     raius filter: to filter points within a certain radius (limited to not too far)

"""

#IMPORTANT NOTE: CHANGE HERE IF WANT TO INCREASE DEFAULT SIZE OF RADIUS

#We are using miles instead of km for US convenience
#To convert to km, multiply by 1.60934 or just use 6378

import numpy as np
import torch
from typing import Tuple, Optional

MI_TO_KM = 1.60934
KM_TO_MI = 1 / MI_TO_KM

EARTH_RADIUS_MI = 3958.7613  # miles
EARTH_RADIUS_KM = 6371.0088  # kilometers

def haversine_m(coord1_deg: np.ndarray, coord2_deg: np.ndarray) -> np.ndarray:
    #coord1, coord2 are tuples of [(lat, lon), (lat,long)] in degrees
    coord1 = np.radians(coord1_deg.astype(float))
    coord2 = np.radians(coord2_deg.astype(float))
    lat1, lon1 = coord1[:,0], coord1[:,1]
    lat2, lon2 = coord2[:,0],coord2[:,1]

    dlat = lat2 - lat1[..., None]
    dlon = lon2 - lon1[..., None]
    #haversine formula
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1)[..., None] * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(h))
    return dist.squeeze(0) if dist.shape[0] == 1 else dist

#If not given a radius, default to 5 miles
#This function returns edges between places within radius_miles or radius_km
def geo_rings(
    places,
    radius_miles: Optional[float] = None,
    radius_km: Optional[float] = None,
    bidirectional: bool = True,
    include_self: bool = False,
):
    if radius_km is not None:
        radius_miles = radius_km * KM_TO_MI
    if radius_miles is None:
        radius_miles = 5.0
    coords = places[["lat", "lon"]].values
    n = len(coords)
    src, dst = [], []

    for i in range(n):
        dists = haversine_m(np.repeat(coords[i:i+1], n, axis=0), coords)  # [n]
        mask = (dists <= radius_miles)
        if not include_self:
            mask &= (np.arange(n) != i)

        nbrs = np.where(mask)[0]
        for j in nbrs.tolist():
            src.append(i); dst.append(j)
            if bidirectional and j != i:
                src.append(j); dst.append(i)

    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
