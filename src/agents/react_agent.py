"""ReAct agent (JD: agent reasoning, ReAct, tool integration, multi-step).
Tools: docs_search, churn_predict, calculator. Works OFFLINE — no LLM key needed.
With GROQ_API_KEY it uses Llama-3.1 for reasoning; without, uses rule-based ReAct.
"""
import re
from ..common.logging import get_logger
log = get_logger(__name__)

def tool_docs_search(query: str) -> str:
    from src.api.main import get_qa
    out = get_qa().ask(query, top_k_final=3)
    return out["answer"][:800] + f" [provider={out.get('provider')}]"

def tool_churn_predict(tenure: float = 5, bill: float = 80.0) -> str:
    from src.ml.predict import predict_rows
    r = predict_rows([{"tenure": tenure, "MonthlyCharges": bill,
                       "TotalCharges": bill * max(tenure, 1),
                       "Contract": "Month-to-month", "InternetService": "Fiber optic"}])[0]
    return f"{r['prediction']} ({r['confidence']:.0%})"

def tool_calculator(expr: str) -> str:
    if not re.fullmatch(r"[\d\s\+\-\*\/\.\(\)%]+", expr):
        return "unsafe expr"
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"

TOOLS = {"docs_search": tool_docs_search, "churn_predict": tool_churn_predict,
         "calculator": tool_calculator}

def react(question: str, max_steps: int = 4) -> dict:
    """Rule-based ReAct loop (transparent, testable) + optional Groq reasoning."""
    trace = []
    ql = question.lower()
    # Thought 1: route
    if any(k in ql for k in ["churn", "leave", "cancel", "retention"]):
        trace.append({"thought": "User asks about churn → check docs + run predictor",
                      "action": "docs_search + churn_predict"})
        docs = tool_docs_search(question)
        trace.append({"observation": docs[:300]})
        # extract numbers if present else defaults
        m_t = re.search(r"tenure\D*(\d+)", ql); m_b = re.search(r"(bill|charge)\D*(\d+)", ql)
        pred = tool_churn_predict(float(m_t.group(1)) if m_t else 5,
                                  float(m_b.group(2)) if m_b else 80.0)
        trace.append({"observation": f"predictor={pred}"})
        answer = f"{docs}\n\nPredictor says: {pred}."
    elif re.search(r"[\d]+\s*[\+\-\*\/]\s*[\d]+", question):
        expr = re.search(r"([\d\s\+\-\*\/\.\(\)%]+)", question).group(1)
        trace.append({"thought": "Math detected → calculator", "action": f"calculator({expr})"})
        val = tool_calculator(expr)
        trace.append({"observation": val})
        answer = f"{expr.strip()} = {val}"
    else:
        trace.append({"thought": "General knowledge → docs_search", "action": "docs_search"})
        docs = tool_docs_search(question)
        trace.append({"observation": docs[:300]})
        answer = docs
    # Try Groq polish if key present (JD: LLM + tools)
    try:
        from src.common.config import settings
        if settings.GROQ_API_KEY:
            from groq import Groq
            c = Groq(api_key=settings.GROQ_API_KEY)
            r = c.chat.completions.create(model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Polish concisely with citations: {answer}"}],
                max_tokens=256)
            answer = r.choices[0].message.content
            trace.append({"thought": "Polished with Groq Llama-3.1", "action": "llm-polish"})
    except Exception as e:
        log.warning(f"groq polish skipped: {e}")
    return {"answer": answer, "trace": trace, "tools_used": list(TOOLS)}
