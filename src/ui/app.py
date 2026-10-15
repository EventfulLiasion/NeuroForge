"""NeuroForge — so simple your non-tech friend gets it. No JSON, no jargon."""
import streamlit as st
import requests

API = "http://localhost:8000"
st.set_page_config(page_title="NeuroForge", page_icon="⚡", layout="wide")

# status line in plain words
try:
    h = requests.get(f"{API}/health", timeout=3).json()
    st.success("✅ AI is ON — ask anything below", icon="✅")
except Exception:
    st.error("⛔ AI is OFF — start Terminal 1: uvicorn src.api.main:app --host 0.0.0.0 --port 8000", icon="⛔")

st.title("⚡ NeuroForge")
st.markdown("### Your company AI. It reads your papers, answers questions, and spots customers who will leave.")
st.caption("You don't need to know AI. Just talk to it like WhatsApp.")

# show what it knows — no mystery
with st.expander("📚 What does the AI know right now? (click to see)", expanded=False):
    try:
        d = requests.get(f"{API}/rag/docs", timeout=5).json()
        st.write(f"It has read **{d.get('count',0)} papers**:")
        for doc in d.get("docs", []):
            st.markdown(f"- **{doc['title']}** — {doc['preview'][:120]}...")
    except Exception as e:
        st.write(f"Can't load list: {e}")

st.divider()
left, right = st.columns(2)

# ---- ASK ----
with left:
    st.header("💬 Ask a question")
    st.caption("Like: refund? churn? photosynthesis?")
    q = st.text_input("Type here", "What causes churn?", label_visibility="collapsed",
                      placeholder="e.g. What is the refund policy?")
    c1, c2, c3 = st.columns(3)
    if c1.button("Churn?"): q = "What causes churn?"
    if c2.button("Refund?"): q = "What is the refund policy?"
    if c3.button("Plants?"): q = "What is photosynthesis?"
    if st.button("Ask ➜", type="primary", use_container_width=True):
        with st.spinner("Reading papers..."):
            try:
                r = requests.post(f"{API}/rag/query", json={"query": q, "top_k": 3}, timeout=120).json()
                st.markdown("#### Answer")
                st.info(r.get("answer", "No answer"))
                st.caption("Proof — it copied from these lines:")
                for i, ctx in enumerate(r.get("contexts", [])[:3]):
                    st.markdown(f"**{i+1}. {ctx.get('title','')}**: {ctx.get('text','')[:250]}...")
            except Exception as e:
                st.error(f"Start the AI first (Terminal 1). {e}")

# ---- PREDICT ----
with right:
    st.header("🔮 Will they leave?")
    st.caption("Move sliders → press button. Green = stays, Red = leaves.")
    tenure = st.slider("Stayed how many months?", 0, 72, 5)
    bill = st.slider("Monthly bill ₹", 20, 120, 80)
    contract = st.radio("Plan", ["Month-to-month", "One year", "Two year"], horizontal=True)
    if st.button("Check ➜", type="primary", use_container_width=True):
        row = {"tenure": tenure, "MonthlyCharges": float(bill),
               "TotalCharges": float(bill * max(tenure, 1)),
               "Contract": contract, "InternetService": "Fiber optic"}
        try:
            r = requests.post(f"{API}/ml/predict", json={"rows": [row]}, timeout=30).json()
            res = r["results"][0]
            if str(res["prediction"]).lower().startswith("y"):
                st.error(f"🔴 YES, will leave — {res['confidence']:.0%} sure. Give discount!", icon="🔴")
            else:
                st.success(f"🟢 NO, will stay — {res['confidence']:.0%} sure.", icon="🟢")
        except Exception as e:
            st.error(f"Start the AI first. {e}")

st.divider()
st.header("➕ Teach it something new (no files)")
st.caption("Type anything — rules, notes, menu — it will answer it instantly.")
t = st.text_input("Title", "Hostel mess menu")
b = st.text_area("What should it remember?", "Monday dal, Tuesday paneer. Mess open 8am-9pm.")
if st.button("Teach ➜"):
    try:
        r = requests.post(f"{API}/rag/add_doc", json={"title": t, "text": b}, timeout=15).json()
        st.success(r.get("message", "Learned!") + " Now ask about it above ☝️")
    except Exception as e:
        st.error(str(e))
