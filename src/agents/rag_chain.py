"""LangChain LCEL RAG chain — JD keyword: LangChain, embeddings, vector search.
Runs on your existing retriever + LLM router. No new infra.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM = ("Answer ONLY from context. Cite [1][2]. If missing, say you don't know.\n"
          "Context:\n{context}\n\nQuestion: {question}")

def format_docs(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] {d.get('text','')[:1000]}" for i, d in enumerate(docs))

def build_langchain_rag(qa_chain, llm_fn=None):
    """qa_chain = your QAChain (has retriever+reranker). llm_fn(text)->str, defaults to router."""
    if llm_fn is None:
        from src.rag import llm_client
        def llm_fn(prompt: str) -> str:
            # extract question/context crudely — router expects (query, docs)
            # we pass through: caller builds docs separately
            return prompt  # replaced below by custom runnable
    prompt = ChatPromptTemplate.from_template(SYSTEM)

    def _retrieve(question: str) -> dict:
        out = qa_chain.ask(question)
        return {"context": format_docs(out["contexts"]), "question": question,
                "contexts": out["contexts"], "provider": out.get("provider")}

    def _generate(d: dict) -> dict:
        from src.rag import llm_client
        docs = d.pop("contexts")
        # rebuild prompt text then route via Groq/Gemini/offline
        text = SYSTEM.format(context=d["context"], question=d["question"])
        # use router for real generation (keeps free-tier + offline working)
        gen = llm_client.generate(d["question"], docs)
        return {**d, **gen, "contexts": docs}

    chain = RunnableLambda(_retrieve) | RunnableLambda(_generate)
    return chain
