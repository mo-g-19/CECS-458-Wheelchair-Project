import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd

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

# test output
# print(parse_query("Find a Thai restaurant in Long Beach with a ramp or no stairs"))
# print(parse_query("I'm looking for Italian food near me with a ramp or no stairs"))


results = pd.read_csv("./data/places.csv")

query = parse_query("I'm looking for Thai food in Long Beach with an accessible restroom")
matched = results[
    (results["cuisine"] == query["cuisine"][0]) &
    (results["location"] == query["location"][0]) &
    (results[query["accessibility"].replace(' ', '_')] == 1)
]

print(matched)
