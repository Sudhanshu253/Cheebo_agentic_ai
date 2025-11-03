# src/indexer.py
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast

def build_index(chunks: List[str], index_path: str = "index/faiss.index", meta_path: str = "index/meta.json"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Built index with {len(chunks)} chunks; saved to {index_path}")

def query_index(query: str, k: int = 5, index_path: str = "index/faiss.index", meta_path: str = "index/meta.json") -> List[str]:
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    index = faiss.read_index(index_path)
    D, I = index.search(q_emb, k)
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    results = []
    for idx in I[0]:
        if idx < len(metas):
            results.append(metas[idx])
    return results

if __name__ == "__main__":
    # quick demo if you want to build from a local file "index/meta.json" chunks
    if os.path.exists("index/meta.json"):
        with open("index/meta.json","r",encoding="utf-8") as f:
            chunks = json.load(f)
        build_index(chunks)
