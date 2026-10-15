"""Full QA chain: rewrite -> hybrid retrieve -> rerank -> generate."""
import re
from . import llm_client

def rewrite_query(q: str) -> str:
    q = q.strip()
    # HyDE-lite: expand acronyms / strip fluff for better BM25+dense hit rate
    q = re.sub(r"\b(what is|explain|tell me about)\b", "", q, flags=re.I).strip(" ?")
    return q or "general overview"

class QAChain:
    def __init__(self, encoder, retriever, reranker):
        self.encoder = encoder
        self.retriever = retriever
        self.reranker = reranker

    def ask(self, query: str, top_k_final: int = 6) -> dict:
        rq = rewrite_query(query)
        q_emb = self.encoder.encode([rq], normalize_embeddings=True)[0]
        hits = self.retriever.search(q_emb, rq, top_k_final=top_k_final * 3)
        hits = self.reranker.rerank(rq, hits, top_k=top_k_final)
        gen = llm_client.generate(rq, hits)
        return {"query": query, "rewritten": rq, "contexts": hits, **gen}
