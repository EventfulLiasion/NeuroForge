"""NeuroForge production API: RAG + ML + drift + eval."""
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="NeuroForge AI Platform",
    version="1.0.0",
    description="Two working AI demos: 1) Ask company docs (RAG) 2) Predict churn (ML). "
                "Start with GET /health, then POST /ml/predict, then POST /rag/query.")

class AskReq(BaseModel):
    query: str
    top_k: int = 6

class PredictReq(BaseModel):
    rows: list[dict]
    model_path: str = "data/churn_model.pkl"

@lru_cache(maxsize=1)
def get_qa():
    """Full RAG if torch/faiss installed, else TF-IDF fallback (minimal deps)."""
    from pathlib import Path as _P
    from src.common.config import CFG
    import json
    corpus_p = _P(CFG["paths"]["rag_corpus"])
    raw_p = _P("data/rag_raw.jsonl")
    src_p = corpus_p if corpus_p.exists() else raw_p
    if src_p.exists():
        docs = [json.loads(l) for l in open(src_p, encoding="utf-8")]
        # ensure ids
        for i, d in enumerate(docs):
            d.setdefault("id", i)
    else:
        docs = [
            {"id": 0, "title": "Churn playbook", "source": "internal-wiki",
             "text": "Customers with month-to-month contracts, high monthly charges and short tenure churn at 3x rate."},
            {"id": 1, "title": "Support SLA", "source": "internal-wiki",
             "text": "Priority tickets must be acknowledged in 1 hour and resolved in 24 hours."},
            {"id": 2, "title": "Photosynthesis primer", "source": "textbook",
             "text": "Photosynthesis converts carbon dioxide and water into glucose using chlorophyll and sunlight."},
        ]
    # try full stack
    try:
        from src.rag.ingest import load_encoder
        from src.rag.hybrid_retriever import HybridRetriever
        from src.rag.reranker import Reranker
        from src.rag.qa_chain import QAChain
        import numpy as np
        enc = load_encoder(CFG["rag"]["embedding_model"])
        embs = enc.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=False)
        ret = HybridRetriever(docs, np.array(embs, dtype="float32"))
        rer = Reranker(CFG["rag"]["reranker_model"], enabled=CFG["rag"]["use_reranker"])
        qa = QAChain(enc, ret, rer)
        qa.mode = "hybrid-faiss-bm25-rerank"
        return qa
    except Exception as e:
        # TF-IDF fallback — zero heavy deps, uses sklearn (already installed)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from src.rag import llm_client
        vec = TfidfVectorizer().fit([d["text"] for d in docs])
        class FallbackQA:
            mode = f"tfidf-fallback (full RAG missing: {type(e).__name__}: {str(e)[:120]})"
            def ask(self, query: str, top_k_final: int = 6):
                import numpy as _np
                qv = vec.transform([query])
                dv = vec.transform([d["text"] for d in docs])
                sims = cosine_similarity(qv, dv)[0]
                idx = _np.argsort(sims)[::-1][:top_k_final]
                hits = [dict(docs[i]) for i in idx]
                gen = llm_client.generate(query, hits)
                return {"query": query, "rewritten": query, "contexts": hits,
                        "mode": self.mode, **gen}
        return FallbackQA()

@app.get("/health", summary="Health — start here", description="Returns ok if API is running.")
def health():
    return {"status": "ok", "service": "neuroforge"}

@app.post("/rag/query", summary="1. Ask company docs",
          description="Ask a question. Returns answer + source docs. Example: {'query':'What causes churn?','top_k':3}")
def rag_query(req: AskReq):
    qa = get_qa()
    return qa.ask(req.query, top_k_final=req.top_k)

@app.post("/ml/predict", summary="2. Predict who leaves",
          description="Send customer rows, get Yes/No churn. Example row: tenure=5, MonthlyCharges=80, Contract=Month-to-month.")
def ml_predict(req: PredictReq):
    from src.ml.predict import predict_rows
    return {"results": predict_rows(req.rows, req.model_path)}

class AddDocReq(BaseModel):
    title: str = "My note"
    text: str = ""
    source: str = "my-upload"

@app.get("/rag/docs", summary="List docs the AI can read",
         description="Returns title + preview of every doc. No input needed.")
def rag_docs():
    import json
    from pathlib import Path as _P
    from src.common.config import CFG
    for p in [CFG["paths"]["rag_corpus"], "data/rag_raw.jsonl"]:
        pp = _P(p)
        if pp.exists():
            docs = [json.loads(l) for l in open(pp, encoding="utf-8")]
            return {"count": len(docs),
                    "docs": [{"title": d.get("title", "?"), "source": d.get("source", "?"),
                              "preview": d.get("text", "")[:200]} for d in docs]}
    return {"count": 0, "docs": []}

@app.post("/rag/add_doc", summary="Teach AI a new doc (no files)",
          description="Send title+text. AI can answer it instantly. Example: {'title':'Fees','text':'Late fee is 50 rupees after 10th.'}")
def rag_add_doc(req: AddDocReq):
    import json
    p = "data/rag_raw.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"title": req.title, "source": req.source, "text": req.text}, ensure_ascii=False) + "\n")
    get_qa.cache_clear()
    # also append to built corpus so hybrid picks it next restart
    try:
        cp = "data/rag_corpus.jsonl"
        import pathlib
        if pathlib.Path(cp).exists():
            with open(cp, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": 999999, "title": req.title, "source": req.source, "text": req.text}, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return {"ok": True, "message": f"Learned '{req.title}'. Ask about it now."}

class AgentReq(BaseModel):
    question: str = "Will a tenure=5 bill=80 customer churn and why?"

@app.post("/agent/react", summary="4. ReAct agent (docs+predict+calc)",
          description="Multi-step agent. Example: {'question':'tenure 5 bill 80 churn? and 12*8?'}")
def agent_react(req: AgentReq):
    from src.agents.react_agent import react
    return react(req.question)

@app.post("/ml/drift", summary="3. Check data drift",
          description="Compare reference vs current numeric columns, returns PSI per column. PSI>0.25 = drifted.")
def ml_drift(payload: dict):
    import pandas as pd
    from src.ml.drift import drift_report
    ref = pd.DataFrame(payload["reference"])
    cur = pd.DataFrame(payload["current"])
    return drift_report(ref, cur)
