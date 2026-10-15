"""ML training with Optuna + MLflow. CPU only.
Handles both tabular (churn.csv) and text (ag_news) datasets.
"""
import json
from pathlib import Path
import joblib
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None
    _HAS_MLFLOW = False
try:
    import optuna
    _HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
    _HAS_OPTUNA = False
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from .features import make_preprocessor, text_pipeline
from ..common.logging import get_logger
log = get_logger(__name__)

class _NoRun:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def _run(name):
    if _HAS_MLFLOW:
        mlflow.set_experiment("neuroforge-ml")
        return mlflow.start_run(run_name=name)
    return _NoRun()

def _log_metric(k, v):
    if _HAS_MLFLOW:
        try: mlflow.log_metric(k, float(v))
        except Exception: pass

def _log_params(d):
    if _HAS_MLFLOW:
        try: mlflow.log_params(d)
        except Exception: pass

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "hist_gb": HistGradientBoostingClassifier(random_state=42),
}

def train_tabular(df: pd.DataFrame, target: str, test_size=0.2, n_trials=10, out_model="data/churn_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]), df[target], test_size=test_size,
        random_state=42, stratify=df[target])
    pre, _, _ = make_preprocessor(df, target)

    best = {"name": None, "score": -1, "pipe": None}
    with _run("tabular-search"):
        for name, clf in MODELS.items():
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            sc = cross_val_score(pipe, X_train, y_train, cv=3, scoring="f1_weighted").mean()
            _log_metric(f"{name}_cv_f1", float(sc))
            log.info(f"{name}: cv_f1={sc:.4f}")
            if sc > best["score"]:
                best = {"name": name, "score": sc, "pipe": pipe}
        # Optuna tune RF depth as demo of real tuning (fallback to defaults if optuna missing)
        if _HAS_OPTUNA:
            def obj(trial):
                d = trial.suggest_int("max_depth", 4, 20)
                e = trial.suggest_int("n_estimators", 100, 300)
                clf = RandomForestClassifier(max_depth=d, n_estimators=e, n_jobs=-1, random_state=42)
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                return cross_val_score(pipe, X_train, y_train, cv=3, scoring="f1_weighted").mean()
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=n_trials)
            _log_params(study.best_params)
            best_params = study.best_params
        else:
            log.warning("optuna not installed, using default RF params")
            best_params = {"max_depth": 12, "n_estimators": 200}
        final = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            **best_params, n_jobs=-1, random_state=42))])
        final.fit(X_train, y_train)
        pred = final.predict(X_test)
        proba = final.predict_proba(X_test) if hasattr(final, "predict_proba") else None
        metrics = {"accuracy": accuracy_score(y_test, pred),
                   "f1_weighted": f1_score(y_test, pred, average="weighted"),
                   "roc_auc_ovr": (roc_auc_score(y_test, proba, multi_class="ovr") if proba is not None and len(set(y_test)) > 2 else
                                   (roc_auc_score(y_test, proba[:, 1]) if proba is not None else 0.0)),
                    "best_model": f"rf-tuned {best_params}",
                    "report": classification_report(y_test, pred, output_dict=True)}
        for k, v in metrics.items():
            if isinstance(v, float):
                _log_metric(k, v)
        Path(out_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, out_model)
        Path("data/ml_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
        log.info(f"Saved {out_model} | {metrics}")
        return final, metrics

def train_text(texts: list[str], labels: list, out_model="data/agnews_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    pipe = text_pipeline()
    with _run("text-tfidf-logreg"):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, pred)),
                   "f1_weighted": float(f1_score(y_test, pred, average="weighted"))}
        for k, v in metrics.items():
            _log_metric(k, v)
        joblib.dump(pipe, out_model)
        log.info(f"Text model saved {out_model} | {metrics}")
        return pipe, metrics
