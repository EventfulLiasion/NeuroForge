"""Tabular feature pipeline: works on Telco-churn-style CSV or HF AG-News fallback."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def make_preprocessor(df: pd.DataFrame, target: str):
    num_cols = df.select_dtypes(include=["number"]).columns.drop(target, errors="ignore").tolist()
    cat_cols = [c for c in df.columns if c not in num_cols + [target]]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return pre, num_cols, cat_cols

def text_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
                     ("clf", LogisticRegression(max_iter=1000))])
