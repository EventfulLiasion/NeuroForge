"""Auto-download datasets. NO manual Kaggle needed.
- RAG corpus: SciQ subset (science QA) + synthetic enterprise docs
- ML tabular: Telco Churn (HF) with sklearn fallback
- ML text: AG News subset (HF)
All saved under data/.
"""
from pathlib import Path
import json
import sys
sys.path.insert(0, ".")
import pandas as pd
try:
    from src.common.logging import get_logger
except ImportError:  # allow running as `python scripts/download_data.py`
    import logging
    def get_logger(name="neuroforge"):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
log = get_logger(__name__)
Path("data").mkdir(exist_ok=True)

def download_rag(n_sciq: int = 1500):
    docs = []
    for _id in ["allenai/sciq", "sciq"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(_id, split=f"train[:{n_sciq}]")
            for i, r in enumerate(ds):
                txt = f"Question: {r['question']} Answer: {r['correct_answer']} Support: {r['support']}"
                docs.append({"title": r["question"][:80], "text": txt, "source": "sciq"})
            log.info(f"SciQ loaded from {_id}: {len(docs)}")
            break
        except Exception as e:
            log.warning(f"SciQ {_id} failed ({e})")
    # always add synthetic enterprise docs so demo has churn/domain knowledge
    enterprise = [
        {"title": "Churn playbook", "source": "internal-wiki",
         "text": "Customers with month-to-month contracts, high monthly charges and short tenure churn at 3x rate. Retention offers and annual contracts reduce churn by 40%."},
        {"title": "Support SLA", "source": "internal-wiki",
         "text": "Priority tickets must be acknowledged in 1 hour and resolved in 24 hours. Escalation goes to tier-2 after 2 failed attempts."},
        {"title": "Photosynthesis primer", "source": "textbook",
         "text": "Photosynthesis converts carbon dioxide and water into glucose using chlorophyll and sunlight. It occurs in chloroplasts."},
    ]
    docs += enterprise
    with open("data/rag_raw.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log.info(f"RAG raw saved: {len(docs)} -> data/rag_raw.jsonl")
    return docs

def download_churn():
    for _id in ["blastchar/telco-customer-churn", "philippelndl/telco-customer-churn"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(_id, split="train")
            df = ds.to_pandas()
            # normalize target col
            for cand in ["Churn", "churn", "label"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "Churn"})
                    break
            df.to_csv("data/churn.csv", index=False)
            log.info(f"Churn saved from {_id}: {df.shape} -> data/churn.csv")
            return df
        except Exception as e:
            log.warning(f"HF churn {_id} failed ({e})")
    try:
        import numpy as np
        n = 2000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(20, 8000, n).round(2),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        })
        df["Churn"] = ((df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > 70)).astype(int).astype(str)
        df["Churn"] = df["Churn"].map({"0": "No", "1": "Yes"})
        df.to_csv("data/churn.csv", index=False)
        return df

def download_agnews(n: int = 3000):
    for _id in ["fancyzhx/ag_news", "ag_news"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(_id, split=f"train[:{n}]")
            df = ds.to_pandas()
            df.to_csv("data/agnews.csv", index=False)
            log.info(f"AGNews saved from {_id}: {df.shape}")
            return df
        except Exception as e:
            log.warning(f"AGNews {_id} failed: {e}")
    return None

if __name__ == "__main__":
    download_rag()
    download_churn()
    download_agnews()
