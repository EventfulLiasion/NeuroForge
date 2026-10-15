"""Hybrid retriever: FAISS dense + BM25 sparse with RRF fusion."""
import pickle
from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    return scores

class HybridRetriever:
    def __init__(self, docs: list[dict], embeddings: np.ndarray):
        """
        docs: [{id, text, source, ...}]
        embeddings: (N, d) float32 normalized
        """
        self.docs = docs
        self.embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query_emb: np.ndarray, query_text: str,
               top_k_dense=20, top_k_sparse=20, top_k_final=6) -> list[dict]:
        q = query_emb.astype("float32").reshape(1, -1)
        _, dense_ids = self.faiss_index.search(q, min(top_k_dense, len(self.docs)))
        dense_ids = dense_ids[0].tolist()
        sparse_scores = self.bm25.get_scores(query_text.lower().split())
        sparse_ids = list(np.argsort(sparse_scores)[::-1][:top_k_sparse])
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k_final]
        out = []
        for doc_idx, score in ranked:
            d = dict(self.docs[doc_idx])
            d["fusion_score"] = float(score)
            out.append(d)
        return out

    def save(self, faiss_path: str | Path, bm25_path: str | Path, docs_path: str | Path):
        import json
        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(docs_path, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, faiss_path, bm25_path, docs_path):
        import json
        index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        docs = [json.loads(l) for l in open(docs_path, encoding="utf-8")]
        obj = cls.__new__(cls)
        obj.docs = docs
        obj.faiss_index = index
        obj.bm25 = bm25
        # reconstruct embeddings placeholder (not needed for search except shape)
        n, d = index.ntotal, index.d
        obj.embeddings = np.zeros((n, d), dtype="float32")
        return obj
