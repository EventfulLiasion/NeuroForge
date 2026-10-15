"""Build FAISS + BM25 index from data/rag_raw.jsonl."""
import json
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.common.config import CFG
from src.rag.ingest import build_corpus, embed_docs, save_jsonl
from src.rag.hybrid_retriever import HybridRetriever

raw = [json.loads(l) for l in open("data/rag_raw.jsonl", encoding="utf-8")]
corpus = build_corpus(raw, CFG["rag"]["chunk_size"], CFG["rag"]["chunk_overlap"])
embs = embed_docs(corpus, CFG["rag"]["embedding_model"])
save_jsonl(corpus, CFG["paths"]["rag_corpus"])
ret = HybridRetriever(corpus, embs)
Path("data").mkdir(exist_ok=True)
ret.save(CFG["paths"]["faiss_index"], CFG["paths"]["bm25_index"], CFG["paths"]["rag_corpus"])
print(f"Indexed {len(corpus)} chunks -> {CFG['paths']['faiss_index']}")
