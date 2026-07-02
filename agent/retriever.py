import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "../catalog/shl_catalog.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "../catalog/shl_index.faiss")

model = None
index = None
catalog = None

def load_catalog():
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def build_index():
    global index, catalog
    catalog = load_catalog()
    m = get_model()
    texts = [f"{item['name']}. {item['description']}" for item in catalog]
    embeddings = m.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"Index built with {len(catalog)} assessments")
    return index

def load_index():
    global index, catalog
    catalog = load_catalog()
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        build_index()
    return index

def search(query: str, top_k: int = 10) -> list:
    global index, catalog
    if index is None:
        load_index()
    m = get_model()
    q_vec = m.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    scores, indices = index.search(q_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(catalog):
            item = catalog[idx].copy()
            item["score"] = float(score)
            results.append(item)
    return results

if __name__ == "__main__":
    build_index()