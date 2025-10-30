import re, nltk
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize


nltk.download('punkt', quiet=True)

def clean_sentence(s):
    s = re.sub(r'[^A-Za-z0-9 .,!?]', '', s)
    return s.strip()

def split_sentences(review):
    return [clean_sentence(s) for s in sent_tokenize(review) if len(s) > 5]

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(sentences):
    return model.encode(sentences)

