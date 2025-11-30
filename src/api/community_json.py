import json, os

STORE = os.path.join("src", "data", "community.json")

def load_store():
    if not os.path.exists(STORE):
        return {}
    with open(STORE, "r") as f:
        return json.load(f)

def save_store(d):
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w") as f:
        json.dump(d, f, indent=2)

def record_flag(business_id: str, flag: str, value: int):
    d = load_store()
    biz = d.setdefault(business_id, {})
    flags = biz.setdefault("flags", {})
    flags[flag] = max(flags.get(flag, 0), value)
    save_store(d)

def add_text_review(business_id: str, text: str):
    if not text.strip():
        return

    d = load_store()
    biz = d.setdefault(business_id, {})
    reviews = biz.setdefault("reviews", [])
    reviews.append(text.strip())
    save_store(d)

def get_flags(business_id: str):
    d = load_store()
    return d.get(business_id, {}).get("flags", {})

def get_text_reviews(business_id: str):
    d = load_store()
    return d.get(business_id, {}).get("reviews", [])