import joblib
import pandas as pd

_model = None

def load_model(path: str = "data/churn_model.pkl"):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model

def predict_rows(rows: list[dict], model_path: str = "data/churn_model.pkl") -> list[dict]:
    m = load_model(model_path)
    df = pd.DataFrame(rows)
    preds = m.predict(df)
    proba = m.predict_proba(df).max(axis=1) if hasattr(m, "predict_proba") else [1.0]*len(preds)
    return [{"prediction": str(p), "confidence": float(c)} for p, c in zip(preds, proba)]
