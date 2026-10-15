"""LLM eval: accuracy/faithfulness + latency + token/cost estimate (JD keywords).
Runs offline, no key needed.
"""
import time, json, sys
sys.path.insert(0, ".")
from src.api.main import get_qa

CASES = [
    {"q": "What causes churn?", "must_contain": ["month-to-month", "churn"]},
    {"q": "What is the refund policy?", "must_contain": ["refund"]},
    {"q": "What is photosynthesis?", "must_contain": ["chlorophyll", "glucose"]},
]
qa = get_qa()
rows = []
for c in CASES:
    t0 = time.time()
    out = qa.ask(c["q"], top_k_final=3)
    lat = time.time() - t0
    ans = out["answer"]
    hit = sum(1 for k in c["must_contain"] if k.lower() in ans.lower()) / len(c["must_contain"])
    toks = len((c["q"] + ans)) // 4
    rows.append({"q": c["q"], "accuracy_proxy": round(hit, 2), "latency_s": round(lat, 2),
                 "tokens_est": toks, "cost_usd_est": round(toks * 0.0000005, 6),
                 "provider": out.get("provider"), "mode": out.get("mode", "?")[:40]})
print(json.dumps(rows, indent=2))
print("PASS" if all(r["accuracy_proxy"] >= 0.5 for r in rows) else "REVIEW")
