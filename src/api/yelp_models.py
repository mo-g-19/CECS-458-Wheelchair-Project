# src/api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Place(BaseModel):
    id: str
    name: str
    city: str
    address: str
    lat: float
    lon: float
    rating: Optional[float] = None
    review_count: Optional[int] = None
    url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    reviews: List[str] = Field(default_factory=list)
    features: Dict[str, int] = Field(default_factory=dict)
    score: Optional[float] = None