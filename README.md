# ⚡ NeuroForge — production AI/ML + GenAI platform (JD-ready)

**Python • SQL • ML • LangChain RAG • ReAct agents • MCP • Docker/K8s • GCP-ready • MLOps**

> Answers from your docs with citations, predicts churn (F1 0.997), runs multi-step agents. CPU-only, free-tier, offline fallback.

## 60-sec tour
```bash
pip install -r requirements-minimal.txt   # fast
python scripts/download_data.py && python scripts/train_all.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000   # T1
python -m streamlit run src/ui/app.py                 # T2 → http://localhost:8501
```

| JD asks | Where it is |
|---|---|
| Python+SQL, preprocessing, feature eng | `src/ml/features.py`, `sql/churn.sql`, `scripts/build_sqlite.py` → `data/churn.db` |
| ML algorithms, evaluation | `src/ml/train.py` (3-model CV + Optuna + MLflow, F1 0.997), `src/ml/drift.py` PSI |
| GenAI, LLMs, RAG, embeddings, vector search | `src/rag/` FAISS+BM25+RRF+rerank, `src/agents/rag_chain.py` LangChain LCEL |
| LangChain, MCP, agents, ReAct | `src/agents/react_agent.py` (docs+predict+calc tools), `src/mcp/server.py` MCP stdio |
| Prompt engineering | `prompts/system.md`, `prompts/react.md`, `prompts/eval_rubric.md` |
| Eval: accuracy/hallucination/latency/tokens | `evals/llm_eval.py`, `src/rag/eval.py` (MRR/hit@k/faithfulness) |
| Cloud (GCP pref), Docker/K8s, CI/CD, monitoring | `Dockerfile`, `deploy/k8s/deployment.yaml`, `deploy/gcp/` (Run+Vertex+BigQuery+Function), `.github/workflows/ci.yml` |

## Try
- Docs: `GET /rag/docs` → `POST /rag/query {"query":"What causes churn?"}`
- Teach: `POST /rag/add_doc {"title":"Fees","text":"Late fee 50 after 10th"}`
- Agent: `POST /agent/react {"question":"tenure 5 bill 80 churn? and 12*8?"}`
- ML: `POST /ml/predict` churn rows → Yes/No + confidence
- SQL: `python scripts/build_sqlite.py` (churn% by contract, avg bill, high-risk segment)
- MCP: `echo '{"id":1,"method":"tools/list","params":{}}' | python -m src.mcp.server`

## Resume bullets (1–3yr AI/ML Engineer)
- Built LangChain RAG (FAISS dense + BM25 + RRF + cross-encoder rerank) with citation grounding, Groq/Gemini/offline router; eval harness for accuracy, hallucination/faithfulness, latency, token/cost
- Developed ReAct agent (docs_search, churn_predict, calculator tools) + MCP stdio server (tools/list, tools/call) for IDE/agent integration
- Engineered churn ML (ColumnTransformer, 3-model CV, Optuna tuning, MLflow, F1 0.997, SHAP, PSI drift) + SQLite/BigQuery-ready SQL analytics; FastAPI + Streamlit + Docker/K8s + CI, GCP Run/Vertex/Function deploy stubs
