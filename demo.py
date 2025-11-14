#This should test it out
import argparse
from src.GNNs.pipeline.rank_local import rank

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--query", type=str, default="")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    df = rank(args.lat, args.lon, args.query, args.topk)
    print(df.to_string(index=False))
