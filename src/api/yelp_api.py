# src/api/yelp_api.py
import os, time, requests
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.api.models import Place

YELP_API_KEY = os.getenv("YELP_API_KEY", "")
if not YELP_API_KEY:
    raise ValueError("Please set YELP_API_KEY in your environment or .env file.")

class YelpApiError(Exception): pass

class YelpClient:
    def __init__(self, api_key: str = YELP_API_KEY, base_url: str = "https://api.yelp.com/v3"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _url(self, path: str): return f"{self.base_url}/{path.lstrip('/')}"

    @retry(reraise=True, stop=stop_after_attempt(3),
           wait=wait_exponential(0.5, 0.5, 4),
           retry=retry_if_exception_type((requests.RequestException, YelpApiError)))
    def _get(self, path: str, params: Dict):
        r = self.session.get(self._url(path), params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "2")))
            raise YelpApiError("Rate limited")
        if not r.ok:
            raise YelpApiError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def search(self, **params): return self._get("/businesses/search", params)
    def reviews(self, biz_id: str): return self._get(f"/businesses/{biz_id}/reviews", {})

class Business(BaseModel):
    id: str
    name: str
    rating: float | None = None
    review_count: int | None = None
    url: str | None = None
    categories: list
    location: dict
    coordinates: dict | None = None

def fetch_places(city: str = "Long Beach, CA",
                 term: str = "restaurants",
                 categories: str = "restaurants",
                 limit: int = 50,
                 max_results: int = 150,
                 include_reviews: bool = True) -> List[Place]:
    client = YelpClient()
    offset, results = 0, []

    while offset < max_results:
        data = client.search(term=term, location=city, categories=categories, limit=limit, offset=offset)
        businesses = data.get("businesses", [])
        if not businesses: break

        for b in businesses:
            biz = Business.model_validate(b)
            addr = ", ".join(filter(None, [
                biz.location.get("address1"),
                biz.location.get("city"),
                biz.location.get("state"),
                biz.location.get("zip_code")
            ]))

            reviews = []
            if include_reviews:
                try:
                    for r in client.reviews(biz.id).get("reviews", []):
                        reviews.append(r.get("text", ""))
                except Exception:
                    pass

            results.append(Place(
                id=biz.id,
                name=biz.name,
                city=biz.location.get("city", ""),
                address=addr,
                lat=(biz.coordinates or {}).get("latitude", 0.0),
                lon=(biz.coordinates or {}).get("longitude", 0.0),
                rating=biz.rating,
                review_count=biz.review_count,
                url=biz.url,
                categories=[c["title"] for c in biz.categories],
                reviews=reviews
            ))
        offset += limit
        if offset >= data.get("total", 0): break

    return results

if __name__ == "__main__":
    print("Fetching sample Yelp data...")
    places = fetch_places(city="Long Beach, CA", term="vegan", max_results=10)
    print(f"Retrieved {len(places)} places")
    if places:
        print(places[0].model_dump(exclude={'reviews'}))
