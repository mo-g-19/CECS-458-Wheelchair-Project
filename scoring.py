import re
import numpy as np


PATTERNS = {
    "entrance": ["ramp", "no steps", "step-free", "curb cut"],
    "restroom": ["accessible restroom", "grab bar", "wide stall", "ADA restroom"],
    "parking": ["accessible parking", "handicap spot", "disabled parking"],
    "seating": ["accessible seating", "wide aisle", "movable chair"]
}

def keyword_hit(sentence, slot):
    for kw in PATTERNS[slot]:
        if re.search(rf"\b{kw}\b", sentence, re.I):
            return True
    return False

def score_place(place, query_emb, sent_embs, sentences, rating, count):
    sims = np.dot(sent_embs, query_emb) / (np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(query_emb))
    top_k = np.argsort(sims)[-5:]
    cover = {}
    for slot in PATTERNS:
        cover[slot] = any(keyword_hit(sentences[i], slot) for i in top_k)
    soft = np.mean(sims[top_k])
    prior = (rating/5) * (np.log1p(count)/np.log1p(1000))
    accessibility_score = round(5*(0.6*sum(cover.values())/len(cover) + 0.2*soft + 0.2*prior), 1)
    return accessibility_score, cover
