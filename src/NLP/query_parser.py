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

# label_thresholds = {}

cuisines_list = ["thai","italian","mexican","vegan","japanese","pizza","burger","ramen","sushi","indian"]  # can add more


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


def get_accessibility_flags(reviews):
    """
    Return dict of label->0/1 by checking semantic similarity in review snippets
    """
    # if no reviews
    if not reviews:
        return {lbl: 0 for lbl in labels}
    
    # else, we have reviews
    snippets = []
    for r in reviews:
        # split by sentences
        for s in r.split("."):
            s = s.strip()
            # skip short sentences since usually just noise
            if len(s) > 3:
                snippets.append(s)
    # if all sentences short -> no snippets
    if not snippets:
        return {lbl: 0 for lbl in labels}

    # create embeddings to find parts related to accessibility (flags)
    label_embeddings = st_model.encode(labels, normalize_embeddings=True)
    snippet_embeddings = st_model.encode(snippets, normalize_embeddings=True)
    similarities = util.cos_sim(label_embeddings, snippet_embeddings)
    
    threshold = 0.40 # set for now, can be refined later
    result = {}
    # if snippet similar enough to accessibility flags -> mark as review mentioning flag
    for i, lbl in enumerate(labels):
        result[lbl] = 1 if float(similarities[i].max().item()) >= threshold else 0
    return result


def build_results(city="Long Beach", term="restaurants"):
    """
    Return a DataFrame with columns: name, cuisine, location, and 4 accessibility flags
    """
    # get places from yelp
    places_labeled = fetch_places(city=city, 
                                  term=term, 
                                  categories="restaurants",
                                  limit=10, 
                                  max_results=20, 
                                  include_reviews=True,
                                  attributes="wheelchair_accessible")
    places_all = fetch_places(city=city, 
                              term=term, 
                              categories="restaurants",
                              limit=10, 
                              max_results=20, 
                              include_reviews=True)

    # combine: prefer labeled results, but also include unlabeled ones (deduplicate by id if available, else by name)
    seen = set()
    combined = []
    for p in places_labeled:
        key = getattr(p, "id", None) or getattr(p, "name", "")
        if key not in seen:
            combined.append((p, True))   # True -> Yelp says wheelchair-accessible
            seen.add(key)
    for p in places_all:
        key = getattr(p, "id", None) or getattr(p, "name", "")
        if key not in seen:
            combined.append((p, False))  # False -> not labeled by Yelp
            seen.add(key)

    rows = []
    for p, is_yelp_labeled in combined:
        # get accessibility flags for each review in each place
        flags = get_accessibility_flags(getattr(p, "reviews", []))
        # if Yelp labels it OR reviews suggest wheelchair/step-free, consider it wheelchair accessible for our pipeline
        inferred_wc = max(flags["wheelchair accessible"], flags["step-free entrance"])
        wheelchair_final = 1 if is_yelp_labeled or inferred_wc == 1 else 0

        row = {
            "id": getattr(p, "id", ""),
            "name": getattr(p, "name", ""),
            "cuisine": (term or "restaurants").lower(),
            "location": getattr(p, "city", city),
            "lat": getattr(p, "lat", None),
            "lon": getattr(p, "lon", None),
            "rating": getattr(p, "rating", None),
            "review_count": getattr(p, "review_count", None),
        }
        # add boolean columns
        for lbl in labels:
            if lbl == "wheelchair accessible":
                row[lbl.replace(" ", "_")] = wheelchair_final
            else:
                row[lbl.replace(" ", "_")] = flags[lbl]
        rows.append(row)
    return pd.DataFrame(rows)

# test output
# print(parse_query("Find a Thai restaurant in Long Beach with a ramp or no stairs"))
# print(parse_query("I'm looking for Italian food near me with a ramp or no stairs"))


query = parse_query("I'm looking for food in Long Beach with an accessible restroom")

city = query["location"][0] if query["location"][0] != "unspecified" else "Long Beach, CA"
term = query["cuisine"][0] if query["cuisine"][0] != "unspecified" else "restaurants"

results = build_results(city=city, term=term)

matched = results[
    (results["cuisine"] == query["cuisine"][0]) &
    (results["location"] == query["location"][0]) &
    (results[query["accessibility"].replace(' ', '_')] == 1)
]

# formatted results to include details relevant to gcn usage
formatted_results = matched[[
    "id","name","cuisine","location","lat","lon","rating","review_count",
    "wheelchair_accessible","accessible_restroom","step-free_entrance","accessible_parking"
]].reset_index(drop=True)

print(formatted_results)
