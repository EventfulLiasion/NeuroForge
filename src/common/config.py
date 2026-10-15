"""Central config loader — works with or without pydantic/yaml installed."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

try:
    import yaml
    def _load_yaml(path):
        with open(path) as f:
            return yaml.safe_load(f)
except ImportError:
    yaml = None
    def _load_yaml(path):
        return {}

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        GROQ_API_KEY: str = ""
        GEMINI_API_KEY: str = ""
        OPENAI_BASE_URL: str = ""
        OPENAI_API_KEY: str = ""
        MLFLOW_TRACKING_URI: str = "mlruns"
        class Config:
            env_file = str(ROOT / ".env")
            extra = "ignore"
    settings = Settings()
except ImportError:
    class Settings:  # minimal fallback reads env directly
        def __init__(self):
            self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
            self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
            self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    settings = Settings()

def load_config(path: str | Path = CONFIG_PATH) -> dict:
    cfg = _load_yaml(path)
    if not cfg:  # hardcoded defaults if yaml missing
        return {
            "paths": {"data_dir": "data", "rag_corpus": "data/rag_corpus.jsonl",
                      "faiss_index": "data/faiss.index", "bm25_index": "data/bm25.pkl",
                      "ml_model": "data/churn_model.pkl", "ml_metrics": "data/ml_metrics.json"},
            "rag": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": 512, "chunk_overlap": 80, "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "use_reranker": True},
            "ml": {"test_size": 0.2, "n_trials_optuna": 5},
            "llm": {"provider_order": ["offline"], "groq_model": "llama-3.1-8b-instant",
                    "gemini_model": "gemini-1.5-flash", "max_tokens": 512, "temperature": 0.2},
        }
    return cfg

CFG = load_config()
