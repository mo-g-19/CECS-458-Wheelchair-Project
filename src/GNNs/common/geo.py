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

def haversine_m(coord1, coord2):
    #coord1, coord2 are tuples of [(lat, lon), (lat,long)] in degrees
    R = 3963.0  # Earth radius in kilometers
    lat1, lon1 = np.radians(coord1[:,0]), np.radians(coord1[:,1])
    lat2, lon2 = np.radians(coord2[:,0]), np.radians(coord2[:,1])
    #haversine formula
    distance = 2*R*np.arcsin(np.sqrt(
        np.sin((lat2 - lat1)/2)**2 +
        np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))
    return distance

#If not given a radius, default to 5m
#This function returns edges between places within radius_m
def geo_rings(places, rid_map, radius_m=5):
    coords = places[["lat", "lon"]].values
    n = len(coords)
    src, dst = [], []
    for i in range(n):
        dists = haversine_m(np.repeat([coords[i]], n, axis=0), coords)
        numbers = np.where((dists > 0) & (dists <= radius_m))[0]
        for j in numbers:
            src.append(i); dst.append(j)
    return torch.tensor([src, dists], dtype=torch.long)
        