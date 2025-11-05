import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from dotenv import load_dotenv
from src.api.yelp_api import fetch_places

load_dotenv()
pd.set_option("display.max_columns", None)

# load models
nlp = spacy.load("en_core_web_lg")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# example accessibility labels
labels = [
    "wheelchair accessible",
    "accessible restroom",
    "step-free entrance",
    "accessible parking"
]

cuisines_list = ["thai","italian","mexican","vegan","japanese"]  # can add more

def parse_query(query):
    doc = nlp(query)
    # extract location (GPE) and possible cuisine words
    location = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    cuisines = [token.text.lower() for token in doc if token.text.lower() in cuisines_list]
    
    # detect "near me" or similar phrases
    query_lower = query.lower()
    if any(p in query_lower for p in ["near me", "around here", "close by", "nearby", "close to me"]):
        location = ["Long Beach"]   # long beach as our default city
    
    # find closest accessibility intent
    query_embedding = st_model.encode(query)
    label_embedding = st_model.encode(labels)
    similarities = util.cos_sim(query_embedding, label_embedding)
    filter_idx = similarities.argmax().item()
    top_filter = labels[filter_idx]

    return {
        "cuisine": cuisines or ["unspecified"],
        "location": location or ["unspecified"],
        "accessibility": top_filter
    }

def _accessibility_flags_from_reviews(reviews):
    """Return dict of label->0/1 by checking semantic similarity in review snippets."""
    if not reviews:
        return {lbl: 0 for lbl in labels}
    snippets = []
    for r in reviews:
        for s in r.split("."):  # very light split to avoid heavy re-tokenization
            s = s.strip()
            if len(s) > 3:
                snippets.append(s)
    if not snippets:
        return {lbl: 0 for lbl in labels}

    lbl_embs = st_model.encode(labels, normalize_embeddings=True)
    snip_embs = st_model.encode(snippets, normalize_embeddings=True)
    sims = util.cos_sim(lbl_embs, snip_embs)
    THRESH = 0.40  # conservative starting point; tune per label if needed
    out = {}
    for i, lbl in enumerate(labels):
        out[lbl] = 1 if float(sims[i].max().item()) >= THRESH else 0
    return out

def _build_results_from_yelp(city="Long Beach", term="restaurants"):
    """Return a DataFrame with columns: name, cuisine, location, and 4 accessibility flags."""
    places = fetch_places(
        city=city, term=term, categories="restaurants",
        limit=50, max_results=150, include_reviews=True
    )
    rows = []
    for p in places:
        flags = _accessibility_flags_from_reviews(getattr(p, "reviews", []))
        row = {
            "name": getattr(p, "name", ""),
            "cuisine": (term or "restaurants").lower(),
            "location": getattr(p, "city", city),
        }
        # add boolean columns matching your original CSV convention
        for lbl in labels:
            row[lbl.replace(" ", "_")] = flags[lbl]
        rows.append(row)
    return pd.DataFrame(rows)

# test output
# print(parse_query("Find a Thai restaurant in Long Beach with a ramp or no stairs"))
# print(parse_query("I'm looking for Italian food near me with a ramp or no stairs"))

# results = pd.read_csv("./data/places.csv")
results = _build_results_from_yelp(city="Long Beach", term="restaurants")

query = parse_query("I'm looking for Thai food in Long Beach with an accessible restroom")

matched = results[
    (results["cuisine"] == query["cuisine"][0]) &
    (results["location"] == query["location"][0]) &
    (results[query["accessibility"].replace(' ', '_')] == 1)
]

print(matched)

