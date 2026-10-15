"""Free-tier LLM router: Groq -> Gemini -> OpenAI-compatible -> offline extractive.
Works with ZERO keys so the demo never breaks on CPU.
"""
from ..common.config import settings, CFG
from ..common.logging import get_logger
log = get_logger(__name__)

SYSTEM = ("You are NeuroForge, a precise enterprise assistant. "
          "Answer ONLY from the provided context. Cite sources like [1], [2]. "
          "If the answer is not in context, say you don't know.")

def _format_ctx(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] ({d.get('source','?')}) {d['text'][:1200]}" for i, d in enumerate(docs))

def offline_answer(query: str, docs: list[dict]) -> dict:
    # extractive fallback: return top sentences overlapping query
    import re
    qwords = set(query.lower().split())
    sents = []
    for i, d in enumerate(docs):
        for s in re.split(r"(?<=[.!?]) +", d["text"]):
            overlap = len(set(s.lower().split()) & qwords)
            if overlap > 0:
                sents.append((overlap, f"{s.strip()} [{i+1}]"))
    sents.sort(reverse=True)
    answer = " ".join(s for _, s in sents[:3]) or (docs[0]["text"][:600] if docs else "I don't know.")
    return {"answer": answer, "provider": "offline-extractive",
            "citations": [d.get("source", "?") for d in docs[:3]]}

def generate(query: str, docs: list[dict]) -> dict:
    ctx = _format_ctx(docs)
    prompt = f"{SYSTEM}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\nANSWER:"
    order = CFG["llm"].get("provider_order", ["offline"])

    if "groq" in order and settings.GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            r = client.chat.completions.create(
                model=CFG["llm"]["groq_model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=CFG["llm"]["max_tokens"],
                temperature=CFG["llm"]["temperature"])
            return {"answer": r.choices[0].message.content, "provider": "groq",
                    "citations": [d.get("source", "?") for d in docs]}
        except Exception as e:
            log.warning(f"Groq failed: {e}")

    if "gemini" in order and settings.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            m = genai.GenerativeModel(CFG["llm"]["gemini_model"])
            r = m.generate_content(prompt)
            return {"answer": r.text, "provider": "gemini",
                    "citations": [d.get("source", "?") for d in docs]}
        except Exception as e:
            log.warning(f"Gemini failed: {e}")

    if "openai_compatible" in order and settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY,
                            base_url=settings.OPENAI_BASE_URL or None)
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=CFG["llm"]["max_tokens"])
            return {"answer": r.choices[0].message.content, "provider": "openai-compatible",
                    "citations": [d.get("source", "?") for d in docs]}
        except Exception as e:
            log.warning(f"OpenAI-compatible failed: {e}")

    return offline_answer(query, docs)
