#This should test it out
import argparse
from src.GNNs.pipeline.rank_local import rank
from src.NLP.query_parser import parse_query, build_results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--query", type=str, default="")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    df = rank(args.lat, args.lon, args.query, args.topk)
    print(df.to_string(index=False))

    # prompt for user input and parse their query
    query = input("Welcome, how can I assist you today?")
    query = parse_query(query)

    if query["intent"] == "search":
        city = query["location"][0] if query["location"][0] != "unspecified" else "Long Beach, CA"
        term = query["cuisine"][0] if query["cuisine"][0] != "unspecified" else "restaurants"

        results = build_results(city=city, term=term)
        required_cols = [lbl.replace(" ", "_") for lbl in query["accessibility"]]
        all_flags = results[required_cols].all(axis=1)

        matched = results[
            (results["cuisine"] == query["cuisine"][0]) &
            (results["location"] == query["location"][0]) &
            all_flags &
            (results["wheelchair_accessible"] == 1)
        ]

        # formatted results to include details relevant to gcn usage
        formatted_results = matched[[
            "id","name","cuisine","location","lat","lon","rating","review_count",
            "wheelchair_accessible","accessible_restroom","step-free_entrance","accessible_parking"
        ]].reset_index(drop=True)