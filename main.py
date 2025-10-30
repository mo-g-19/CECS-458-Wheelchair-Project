
def parse_query(user_query: str):
    user_query = user_query.lower()
    # default values
    cuisine = None
    location = "Long Beach, CA"
    accessibility_phrase = "wheelchair accessible"

    # simple keyword-based extraction
    for cuisine_name in ["italian", "mexican", "thai", "japanese", "vegan", "burger", "pizza"]:
        if cuisine_name in user_query:
            cuisine = cuisine_name
            break

    # detect if user says "near me" or a city
    if "near me" in user_query:
        location = "Long Beach, CA"   # default or user location
    else:
        for city in ["long beach", "los angeles", "cerritos"]:
            if city in user_query:
                location = city.title()
                break

    # accessibility keywords
    if "accessible" in user_query or "wheelchair" in user_query:
        accessibility_phrase = "wheelchair accessible"

    return cuisine, location, accessibility_phrase


if __name__ == "__main__":
    query = "Find Italian restaurants near me that are wheelchair accessible"
    cuisine, location, acc_phrase = parse_query(query)
    # next: get results for query
    # then transfer info to gcn
