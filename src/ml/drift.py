"""PSI drift detection — zero extra deps, pure pandas/numpy."""
import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    e = expected.dropna(); a = actual.dropna()
    if e.nunique() < 2:
        return 0.0
    qs = np.linspace(0, 1, buckets + 1)
    breaks = [e.quantile(q) for q in qs]
    e_perc = np.histogram(e, bins=breaks)[0] / max(len(e), 1)
    a_perc = np.histogram(a, bins=breaks)[0] / max(len(a), 1)
    e_perc = np.where(e_perc == 0, 0.0001, e_perc)
    a_perc = np.where(a_perc == 0, 0.0001, a_perc)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    out = {}
    for col in ref.select_dtypes(include=["number"]).columns:
        if col in cur.columns:
            v = psi(ref[col], cur[col])
            out[col] = {"psi": round(v, 4),
                        "drift": bool(v > 0.25)}
    return out
