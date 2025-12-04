# Draft - just testing model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


model = SentenceTransformer("all-MiniLM-L6-v2")

# Example data
sentences = [
    "The restaurant has a wheelchair ramp.",
    "There is a ramp for wheelchairs.",
    "The cafe is not accessible because of stairs.",
    "The food was delicious!"
]
embeddings = model.encode(sentences)

print("Embeddings shape:", embeddings.shape)

# Find which sentence is most similar to user query
# query = "Does the place have wheelchair access?"
query = "Is there a ramp and accessible restroom?"
query_emb = model.encode([query])

scores = cosine_similarity(query_emb, embeddings)[0]
best_idx = np.argmax(scores)

print("\nQuery:", query)
print("Most similar sentence:", sentences[best_idx])
print("Similarity score:", scores[best_idx])
