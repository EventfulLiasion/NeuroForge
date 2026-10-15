"""Cross-encoder reranker with graceful CPU fallback."""
from ..common.logging import get_logger
log = get_logger(__name__)

class Reranker:
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        if enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
                log.info(f"Reranker loaded: {model_name}")
            except Exception as e:
                log.warning(f"Reranker disabled, fallback to fusion order: {e}")
                self.enabled = False

    def rerank(self, query: str, docs: list[dict], top_k: int = 6) -> list[dict]:
        if not self.enabled or self.model is None or not docs:
            return docs[:top_k]
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
