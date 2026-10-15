"""One-command training: churn tabular + agnews text."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.common.config import CFG
from src.ml.train import train_tabular, train_text

# tabular
df = pd.read_csv("data/churn.csv")
# find target
target = "Churn" if "Churn" in df.columns else df.columns[-1]
_, m1 = train_tabular(df, target, test_size=CFG["ml"]["test_size"],
                      n_trials=5, out_model=CFG["paths"]["ml_model"])
print("TABULAR:", m1)

# text (optional, small subset for CPU speed)
try:
    tdf = pd.read_csv("data/agnews.csv").head(2000)
    txt_col = "text" if "text" in tdf.columns else tdf.columns[0]
    lab_col = "label" if "label" in tdf.columns else tdf.columns[-1]
    _, m2 = train_text(tdf[txt_col].tolist(), tdf[lab_col].tolist())
    print("TEXT:", m2)
except Exception as e:
    print("text training skipped:", e)
