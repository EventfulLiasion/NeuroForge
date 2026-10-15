"""Evaluate RAG + ML, print report."""
import sys, json
sys.path.insert(0, ".")
from src.common.config import CFG
from src.rag.ingest import load_encoder
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.qa_chain import QAChain
from src.rag.eval import evaluate_qa
import numpy as np

# build quick in-memory QA for eval
docs = [json.loads(l) for l in open(CFG["paths"]["rag_corpus"], encoding="utf-8")]
enc = load_encoder(CFG["rag"]["embedding_model"])
embs = enc.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=False)
ret = HybridRetriever(docs, np.array(embs, dtype="float32"))
rer = Reranker(CFG["rag"]["reranker_model"], enabled=CFG["rag"]["use_reranker"])
chain = QAChain(enc, ret, rer)

# auto-generate eval pairs from corpus (first 20 docs -> cloze questions)
pairs = [{"question": f"What does this say: {d.get('title') or d['text'][:60]}?",
          "gold_id": d["id"]} for d in docs[:20]]
print(json.dumps(evaluate_qa(chain, pairs), indent=2))
print("ML:", open("data/ml_metrics.json").read() if __import__("pathlib").Path("data/ml_metrics.json").exists() else "train first")
