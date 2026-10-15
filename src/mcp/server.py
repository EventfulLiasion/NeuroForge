"""MCP server (JD: Model Context Protocol) — exposes NeuroForge as tools.
Stdio JSON-RPC so Claude/IDE agents can call: docs.search, ml.predict, agent.react.
Run: python -m src.mcp.server
"""
import sys, json

def handle(req: dict) -> dict:
    method, params, _id = req.get("method"), req.get("params", {}), req.get("id")
    try:
        if method == "tools/list":
            return {"id": _id, "result": {"tools": [
                {"name": "docs.search", "desc": "Search company docs"},
                {"name": "ml.predict", "desc": "Churn Yes/No"},
                {"name": "agent.react", "desc": "Multi-step ReAct agent"}]}}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            if name == "docs.search":
                from src.api.main import get_qa
                out = get_qa().ask(args.get("query", ""), top_k_final=3)
                return {"id": _id, "result": {"answer": out["answer"], "provider": out.get("provider")}}
            if name == "ml.predict":
                from src.ml.predict import predict_rows
                return {"id": _id, "result": {"results": predict_rows(args.get("rows", []))}}
            if name == "agent.react":
                from src.agents.react_agent import react
                return {"id": _id, "result": react(args.get("question", ""))}
            return {"id": _id, "error": f"unknown tool {name}"}
        return {"id": _id, "error": f"unknown method {method}"}
    except Exception as e:
        return {"id": _id, "error": str(e)}

def main():
    for line in sys.stdin:
        line = line.strip()
        if line:
            sys.stdout.write(json.dumps(handle(json.loads(line))) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
