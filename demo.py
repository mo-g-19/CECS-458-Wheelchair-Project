#This should test it out
import argparse
from src.GNNs.pipeline.rank_local import rank
from src.NLP.query_parser import parse_query, build_results
from src.api.community_json import record_flag, add_text_review, get_flags, get_text_reviews

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--lat", type=float, required=True)
#     ap.add_argument("--lon", type=float, required=True)
#     ap.add_argument("--query", type=str, default="")
#     ap.add_argument("--topk", type=int, default=10)
#     args = ap.parse_args()
#     df = rank(args.lat, args.lon, args.query, args.topk)
#     print(df.to_string(index=False))






# Simple mapping from city -> approximate coords
# (used internally by rank())
CITY_COORDS = {
    "los angeles": (34.0522, -118.2437),
    "long beach": (33.7701, -118.1937),
}


def guess_coords_for_city(city: str):
    if not city:
        return 33.7701, -118.1937  # long beach default

    city_l = city.lower()
    for key, (lat, lon) in CITY_COORDS.items():
        if key in city_l:
            return lat, lon

    # default if city not recognized
    return 33.7701, -118.1937


def yes_no(prompt: str):
    ans = input(prompt + " (y/n, Enter to skip): ").strip().lower()
    if ans in ("y", "yes"):
        return 1
    if ans in ("n", "no"):
        return 0
    return None


# ---- search flow ----

def search_flow(raw_query: str, city: str):
    print(f"City:       {city}")

    lat, lon = guess_coords_for_city(city)
    print("\nFinding best restaurants...")
    df = rank(lat, lon, raw_query, topk=5)

    if df is None or df.empty:
        print("No places found.\n")
        return

    for _, row in df.iterrows():
        # Prefer business id; if missing, fall back to name as key
        biz_id = str(row.get("id") or row.get("name") or "").strip()
        name = row.get("name", "Unknown")

        final_score = row.get("final_score", None)
        access_score = row.get("accessibility_score", None)
        quality_score = row.get("quality_score", None)

        print("\n==============================")
        print(name)
        # if biz_id:
        #     print(f"  Business ID:        {biz_id}")
        if access_score is not None:
            stars = (row.get("accessibility_score") or 0) * 5
            print(f"  Accessibility score: {stars:.1f}")
        if quality_score is not None:
            stars = (row.get("quality_score") or 0) * 5
            print(f"  Quality score:      {stars:.1f}")
        if final_score is not None:
            stars = (row.get("final_score") or 0) * 5
            print(f"  Final score:    {stars:.1f}/5 stars")

        # Community flags
        if biz_id:
            flags = get_flags(biz_id)
            label_map = {
                "wheelchair_accessible": "Wheelchair accessible",
                "accessible_restroom": "Accessible restroom",
                "step_free_entrance": "Step-free entrance",
                "accessible_parking": "Accessible parking",
                "elevator_access": "Elevator access",
            }
            flag_labels = [
                label for key, label in label_map.items()
                if flags.get(key) == 1
            ]
            if flag_labels:
                print("  Community flags: " + ", ".join(flag_labels))

            # Community text reviews (show first snippet)
            reviews = get_text_reviews(biz_id)
            if reviews:
                snippet = reviews[0][:160]
                if len(reviews[0]) > 160:
                    snippet += "..."
                print("  Community review:", snippet)

    print("\n(end of top 5)\n")


# --- review flow ---

def review_flow():
    # TODO: account for user entering just restaurant name
    biz_id = input("Business ID or restaurant name key:\n> ").strip()
    if not biz_id:
        print("No ID/name provided. Cancelling.\n")
        return

    print("\nAnswer a few yes/no questions (or press Enter to skip):")
    questions = [
        ("wheelchair_accessible", "Generally wheelchair-accessible layout?"),
        ("accessible_restroom", "Accessible restroom?"),
        ("step_free_entrance", "Ramp / step-free entrance?"),
        ("accessible_parking", "Accessible parking nearby?"),
        ("elevator_access", "Accessible elevator if multiple floors?"),
    ]

    for flag_key, prompt in questions:
        val = yes_no(prompt)
        if val is not None:
            record_flag(biz_id, flag_key, val)

    print("\nOptional short written accessibility review:")
    text = input("> ").strip()
    if text:
        add_text_review(biz_id, text)

    print("\nThanks! Your review has been recorded and will help other users.\n")



def main():
    while True:
        user_input = input(
            "\nHow can I assist you today?\n"
            "(Describe what you want, or type 'q' to quit)\n> "
        ).strip()

        if user_input.lower() in ("q", "quit", "exit"):
            print("\nGoodbye!")
            break

        query = parse_query(user_input)

        intent = (query.get("intent") or "").lower()
        locations = query.get("location") or ["unspecified"]
        city = locations[0] if locations else "unspecified"

        if city == "unspecified":
            city = input("I couldn't detect a city. Please enter a city (e.g. 'Los Angeles, CA'):\n> ").strip()
            if not city:
                print("No city given. Please try again.\n")
                continue

        # Decide intent: search vs review
        if intent == "search":
            search_flow(user_input, city)
        elif intent in ("review", "rate"):
            review_flow()
        else:
            # fallback heuristic based on raw text
            text_l = user_input.lower()
            if "review" in text_l or "rate" in text_l:
                review_flow()
            else:
                search_flow(user_input, city)


if __name__ == "__main__":
    main()

