"""Build SQLite churn.db from churn.csv + run sql/churn.sql (JD: SQL, data pipelines)."""
import sqlite3, pathlib
import pandas as pd
db = "data/churn.db"
df = pd.read_csv("data/churn.csv")
con = sqlite3.connect(db)
df.to_sql("churn", con, if_exists="replace", index=False)
print(f"wrote {db}: {df.shape}")
for q in open("sql/churn.sql").read().split(";"):
    q = q.strip()
    if q and not q.startswith("-- JD"):
        print("---", q.splitlines()[0][:80])
        try:
            print(con.execute(q).fetchall()[:5])
        except Exception as e:
            print("skip:", e)
con.close()
