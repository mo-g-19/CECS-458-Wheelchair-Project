# src/api/seed_yelp_cache.py
import os, json
from sentence_transformers import SentenceTransformer, util

from src.api.yelp_api import fetch_places

STORE = os.path.join("src", "data", "yelp_cache.json")

# accessibility labels (same as in query_parser)
LABELS = [
    "wheelchair accessible",
    "accessible restroom",
    "step-free entrance",
    "accessible parking",
    "automatic door",
    "elevator access"
]

st_model = SentenceTransformer("all-MiniLM-L6-v2")
LABEL_EMBS = st_model.encode(LABELS, normalize_embeddings=True)

def extract_accessibility_snippets(review_texts, threshold: float = 0.35):
    """
    Given a list of raw review texts, return only the sentence-level snippets
    that are semantically related to accessibility labels.
    """
    snippets = []
    for r in review_texts or []:
        for s in r.split("."):
            s = s.strip()
            if len(s) > 3:
                snippets.append(s)

    if not snippets:
        return []

    snippet_embs = st_model.encode(snippets, normalize_embeddings=True)
    sims = util.cos_sim(LABEL_EMBS, snippet_embs)  # shape [num_labels, num_snippets]

    kept = []
    for j, snip in enumerate(snippets):
        max_sim = float(sims[:, j].max().item())
        if max_sim >= threshold:
            kept.append(snip)

    return kept

def seed_yelp_cache(city="Long Beach, CA", term="thai", max_results=20):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)

    places = fetch_places(
        city=city,
        term=term,
        categories="restaurants",
        limit=20,
        max_results=max_results,
        include_reviews=True
    )

    cache = {}

    for p in places:
        # Only keep accessibility-related snippets from Yelp reviews
        access_snippets = extract_accessibility_snippets(p.reviews)

        cache[p.id] = {
            "name": p.name,
            "city": p.city,
            "cuisine": term.lower(),
            "rating": p.rating,
            "review_count": p.review_count,
            "lat": p.lat,
            "lon": p.lon,
            "url": p.url,
            "reviews": access_snippets,        # filtered snippets only
            "categories": p.categories or []
        }

    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(cache)} places to {STORE}")

if __name__ == "__main__":
    seed_yelp_cache("Long Beach, CA", "thai", max_results=30)
