import json
import os
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "../catalog/shl_catalog.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "../catalog/shl_index.faiss")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "../catalog/vectorizer.pkl")

index = None
catalog = None
vectorizer = None

def load_catalog():
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)

def build_index():
    global index, catalog, vectorizer
    catalog = load_catalog()
    # Repeat name 3x to weight exact-name matches much higher than
    # description-only overlap, and include a category hint so terms
    # like "personality" / "cognitive ability" / "coding" match better.
    type_hint = {
        "A": "ability aptitude cognitive reasoning",
        "B": "biodata situational judgment",
        "C": "competency",
        "D": "development 360 feedback",
        "E": "assessment exercise simulation",
        "K": "knowledge skills technical coding programming",
        "P": "personality behavior traits",
        "S": "simulation practical exercise",
    }
    texts = [
        f"{item['name']} {item['name']} {item['name']}. "
        f"{type_hint.get(item.get('test_type', ''), '')}. "
        f"{item['description']}"
        for item in catalog
    ]

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"Index built with {len(catalog)} assessments")

def load_index():
    global index, catalog, vectorizer
    catalog = load_catalog()
    if os.path.exists(INDEX_PATH) and os.path.exists(VECTORIZER_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        build_index()

def search(query: str, top_k: int = 10) -> list:
    global index, catalog, vectorizer
    if index is None:
        load_index()
    
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(catalog):
            item = catalog[idx].copy()
            item["score"] = float(score)
            results.append(item)
    return results

if __name__ == "__main__":
    build_index()