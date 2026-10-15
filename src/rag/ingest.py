"""Embeddings + chunking + ingest. CPU-friendly MiniLM."""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def load_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def build_corpus(raw_docs: list[dict], chunk_size=512, overlap=80) -> list[dict]:
    docs, idx = [], 0
    for r in raw_docs:
        for ch in chunk_text(r["text"], chunk_size, overlap):
            docs.append({"id": idx, "text": ch, "source": r.get("source", "unknown"),
                         "title": r.get("title", "")})
            idx += 1
    return docs

def embed_docs(docs: list[dict], model_name: str, batch_size: int = 64) -> np.ndarray:
    enc = load_encoder(model_name)
    texts = [d["text"] for d in docs]
    emb = enc.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=True)
    return np.array(emb, dtype="float32")

def save_jsonl(docs: list[dict], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
