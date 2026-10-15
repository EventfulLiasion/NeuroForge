def test_rrf():
    from src.rag.hybrid_retriever import reciprocal_rank_fusion
    s = reciprocal_rank_fusion([[0, 1], [1, 0]])
    assert s[0] > 0 and s[1] > 0

def test_rewrite():
    from src.rag.qa_chain import rewrite_query
    assert "photosynthesis" in rewrite_query("What is photosynthesis?").lower()

def test_offline_llm():
    from src.rag import llm_client
    out = llm_client.generate("test?", [{"text": "photosynthesis uses chlorophyll", "source": "t"}])
    assert "answer" in out and len(out["answer"]) > 0
