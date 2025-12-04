import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from dotenv import load_dotenv
from src.api.yelp_api import fetch_places
from src.api.community_json import get_flags, record_flag

load_dotenv()
pd.set_option("display.max_columns", None)

# load models
nlp = spacy.load("en_core_web_lg")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# example accessibility labels
LABELS = [
    "wheelchair accessible",
    "accessible restroom",
    "step-free entrance",
    "accessible parking",
    "automatic door",
    "elevator access"
]

# label_thresholds = {}

cuisines_list = ["thai","italian","mexican","vegan","japanese","pizza","burger","ramen","sushi","indian"]  # can add more


def parse_query(query):
    """
    Return cuisine, location (city), and accessibility filter from user query
    """
    # determine if user wants to find or review place
    query_lower = query.lower()
    is_review = any(word in query_lower for word in ["review", "rate", "my experience", "i went to", "visited"])
    is_search = any(word in query_lower for word in ["find", "show", "looking for", "restaurant", "place", "search", "near me"])

    if is_review:
        intent = "review"
    elif is_search:
        intent = "search"
    else:
        intent = "search"
    
    matched_labels = []
    doc = nlp(query)
    # extract location (GPE) and possible cuisine words
    location = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    cuisines = [token.text.lower() for token in doc if token.text.lower() in cuisines_list]
    
    if intent == "search":
        # detect "near me" or similar phrases
        if any(p in query_lower for p in ["near me", "around here", "close by", "nearby", "close to me"]):
            location = ["Long Beach"]   # long beach as our default city

        matched = set()
        # rule-based flag detection for explicit phrases
        for lbl in LABELS:
            if lbl in query_lower:
                matched.add(lbl)

        # find closest accessibility intent
        query_embedding = st_model.encode(query, normalize_embeddings=True)
        label_embedding = st_model.encode(LABELS, normalize_embeddings=True)
        similarities = util.cos_sim(query_embedding, label_embedding)[0].tolist()
        
        threshold = 0.40 # set for now
        for label, score in zip(LABELS, similarities):
            if score >= threshold:
                matched.add(label)

        # if nothing passes threshold, still keep  highest one
        if not matched:
            top_idx = int(similarities.index(max(similarities)))
            matched = {LABELS[top_idx]}
        matched_labels = list(matched)

    return {
        "intent": intent,
        "cuisine": cuisines or ["unspecified"],
        "location": location or ["unspecified"],
        "accessibility": matched_labels
    }


def get_accessibility_flags(reviews):
    """
    Return dict of label->0/1 by checking semantic similarity in review snippets
    """
    # if no reviews
    if not reviews:
        return {lbl: 0 for lbl in LABELS}
    
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
        return {lbl: 0 for lbl in LABELS}

    # create embeddings to find parts related to accessibility (flags)
    label_embeddings = st_model.encode(LABELS, normalize_embeddings=True)
    snippet_embeddings = st_model.encode(snippets, normalize_embeddings=True)
    similarities = util.cos_sim(label_embeddings, snippet_embeddings)
    
    threshold = 0.20 # set for now, can be refined later
    result = {}
    # if snippet similar enough to accessibility flags -> mark as review mentioning flag
    for i, lbl in enumerate(LABELS):
        result[lbl] = 1 if float(similarities[i].max().item()) >= threshold else 0
    return result


def build_results(city="Long Beach", term="restaurants"):
    """
    Return a DataFrame with columns: name, cuisine, location, and accessibility flags
    """
    # get places from yelp labeled as wheelchair accessible
    places_labeled = fetch_places(city=city, 
                                  term=term, 
                                  categories="restaurants",
                                  limit=10, 
                                  max_results=20, 
                                  include_reviews=True,
                                  attributes="wheelchair_accessible")
    # get places from yelp, not checking for label
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
        for lbl in LABELS:
            if lbl == "wheelchair accessible":
                row[lbl.replace(" ", "_")] = wheelchair_final
            else:
                row[lbl.replace(" ", "_")] = flags[lbl]

        community = get_flags(getattr(p, "id", ""))
        for lbl in LABELS:
            key = lbl.replace(" ", "_")
            if key in community:
                # community rating overrides model inference
                row[key] = community[key]
        rows.append(row)
    return pd.DataFrame(rows)



# query = parse_query("I'm looking for Thai food in Long Beach")

# city = query["location"][0] if query["location"][0] != "unspecified" else "Long Beach, CA"
# term = query["cuisine"][0] if query["cuisine"][0] != "unspecified" else "restaurants"

# results = build_results(city=city, term=term)
# required_cols = [lbl.replace(" ", "_") for lbl in query["accessibility"]]
# all_flags = results[required_cols].all(axis=1)

# matched = results[
#     (results["cuisine"] == query["cuisine"][0]) &
#     (results["location"] == query["location"][0]) &
#     all_flags &
#     (results["wheelchair_accessible"] == 1)
# ]

# # formatted results to include details relevant to gcn usage
# formatted_results = matched[[
#     "id","name","cuisine","location","lat","lon","rating","review_count",
#     "wheelchair_accessible","accessible_restroom","step-free_entrance","accessible_parking"
# ]].reset_index(drop=True)

# print(formatted_results)



record_flag("business_id_3", "step-free_entrance", 1)
