"""Lightweight RAG eval: MRR, hit@k, faithfulness heuristic, latency.
No API needed — runs fully offline on CPU.
"""
import time

def hit_at_k(retrieved_ids: list[int], gold_id: int, k: int = 5) -> int:
    return int(gold_id in retrieved_ids[:k])

def mrr(retrieved_ids: list[int], gold_id: int) -> float:
    try:
        rank = retrieved_ids.index(gold_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def faithfulness_heuristic(answer: str, contexts: list[dict]) -> float:
    """Fraction of answer words covered by retrieved context (cheap proxy)."""
    ctx = " ".join(c["text"].lower() for c in contexts)
    words = [w for w in answer.lower().split() if len(w) > 3]
    if not words:
        return 0.0
    hit = sum(1 for w in words if w in ctx)
    return round(hit / len(words), 3)

def evaluate_qa(chain, eval_pairs: list[dict], k: int = 5) -> dict:
    mrrs, hits, faiths, lats = [], [], [], []
    for ex in eval_pairs:
        t0 = time.time()
        out = chain.ask(ex["question"])
        lats.append(time.time() - t0)
        ids = [c["id"] for c in out["contexts"]]
        mrrs.append(mrr(ids, ex["gold_id"]))
        hits.append(hit_at_k(ids, ex["gold_id"], k))
        faiths.append(faithfulness_heuristic(out["answer"], out["contexts"]))
    n = max(len(eval_pairs), 1)
    return {
        "n": n, f"hit@{k}": round(sum(hits) / n, 3),
        "mrr": round(sum(mrrs) / n, 3),
        "faithfulness_proxy": round(sum(faiths) / n, 3),
        "p50_latency_s": round(sorted(lats)[len(lats)//2] if lats else 0, 3),
    }
