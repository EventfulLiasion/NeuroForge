"""SHAP explainability (sampled, CPU-safe)."""
import pandas as pd

def explain(model, rows: pd.DataFrame, nsamples: int = 50) -> dict:
    try:
        import shap
        # Use first pipeline step output if preprocessor exists
        background = rows.head(min(nsamples, len(rows)))
        explainer = shap.Explainer(model.predict, background)
        sv = explainer(background.head(5))
        return {"values_shape": list(sv.values.shape),
                "base_values": sv.base_values.tolist() if hasattr(sv.base_values, "tolist") else str(sv.base_values),
                "feature_names": list(background.columns)}
    except Exception as e:
        return {"error": str(e)}
