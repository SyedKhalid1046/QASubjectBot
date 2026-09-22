"""Unit and Integration Tests for QASubjectBot RAG Pipeline & Vector Search."""

import pytest
from embed_index import EmbeddingEngine, VectorStoreIndex
from rag_pipeline import RAGPipeline, answer_question


@pytest.fixture(scope="module")
def rag_pipeline():
    """Initializes and caches the default RAG pipeline for testing."""
    return RAGPipeline(
        index_dir="./data/faiss_index",
        embedding_model="all-MiniLM-L6-v2",
        score_threshold=0.25,
    )


def test_faiss_index_loading_and_vector_search(rag_pipeline):
    store = rag_pipeline.vector_store
    assert store.index is not None
    assert store.index.ntotal >= 50
    assert len(store.chunks) == store.index.ntotal

    results = store.search("Magnetic Resonance Imaging MRI brain scan", top_k=3)
    assert len(results) == 3
    top_chunk, score = results[0]
    assert score > 0.05
    assert "BIOPSYCHOLOGY" in top_chunk.source.upper()


def test_semantic_retrieval_across_topics(rag_pipeline):
    test_cases = [
        ("What is Magnetic Resonance Imaging (MRI) and how does it work?", "UNIT_1_BIOPSYCHOLOGY.pptx"),
        ("What is the structure of the nervous system and neurons?", "UNIT_2_BIOPSYCHOLOGY.pdf"),
        ("What is synaptic transmission and action potentials in the brain?", "UNIT_3_BIOPSYCHOLOGY.pdf"),
    ]

    for query, expected_source in test_cases:
        retrieved = rag_pipeline.retrieve(query, top_k=5, score_threshold=0.05)
        assert len(retrieved) > 0, f"Failed to retrieve chunks for query: '{query}'"
        assert any(expected_source.lower() in c.source.lower() for c, _ in retrieved), (
            f"Expected {expected_source} in results for '{query}', got {[c.source for c, _ in retrieved]}"
        )


def test_out_of_scope_fallback_behavior(rag_pipeline):
    out_of_scope_queries = [
        "What is the average surface temperature on Mars during winter?",
        "Who won the 1994 FIFA World Cup in soccer?",
    ]

    for q in out_of_scope_queries:
        res = rag_pipeline.answer_question(q, score_threshold=0.25)
        assert res["status"] == "insufficient_context", f"Expected fallback for '{q}', got status {res['status']}"
        assert "don't have enough information" in res["answer"].lower()
        assert len(res["citations"]) == 0
        assert res["retrieval_confidence"] < 0.25


def test_answer_question_api_contract(rag_pipeline):
    query = "What is Magnetic Resonance Imaging (MRI)?"
    res = rag_pipeline.answer_question(query, score_threshold=0.05)

    # Assert required contract fields
    assert "query" in res
    assert "answer" in res
    assert "citations" in res
    assert "retrieval_confidence" in res
    assert "status" in res
    assert res["status"] in ["success", "insufficient_context"]

    if res["status"] == "success":
        assert len(res["citations"]) >= 1
        cit = res["citations"][0]
        assert "source" in cit
        assert "page" in cit
        assert "chunk_id" in cit
        assert "similarity_score" in cit
        assert "snippet" in cit
        assert cit["source"].endswith((".pdf", ".txt", ".docx", ".pptx"))
        assert isinstance(cit["page"], int)
        assert isinstance(cit["similarity_score"], float)


def test_convenience_wrapper():
    res = answer_question("What is Magnetic Resonance Imaging?", score_threshold=0.05)
    assert res is not None
    assert "answer" in res
    assert res["status"] in ["success", "insufficient_context"]

