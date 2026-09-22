"""Test Suite for Hybrid Search (BM25 + FAISS), Cross-Encoder Reranker, and Conversation Memory."""

import pytest
from embed_index import EmbeddingEngine, VectorStoreIndex, tokenize_for_bm25
from ingestion import DocumentChunk
from rag_pipeline import ConversationMemory, CrossEncoderReranker, RAGPipeline


@pytest.fixture
def sample_chunks():
    chunks = [
        DocumentChunk(
            text="The blood-brain barrier (BBB) protects neural tissue from toxins and circulating pathogens.",
            metadata={"source": "neuro_unit1.pdf", "page": 4, "chunk_id": "c1"},
        ),
        DocumentChunk(
            text="Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k)) * V where d_k is the dimension of the keys.",
            metadata={"source": "transformer_spec.pdf", "page": 2, "chunk_id": "c2"},
        ),
        DocumentChunk(
            text="Break-even point in units is calculated using the formula: Fixed Costs / (Selling Price per unit - Variable Cost per unit).",
            metadata={"source": "marginal_costing.docx", "page": 1, "chunk_id": "c3"},
        ),
        DocumentChunk(
            text="Action potentials in unmyelinated axons propagate continuously along the entire axolemma at slower speeds.",
            metadata={"source": "neuro_unit1.pdf", "page": 8, "chunk_id": "c4"},
        ),
    ]
    return chunks


def test_bm25_tokenization():
    text = "Break-even point is $50,000 / (₹100 - 20%)!"
    tokens = tokenize_for_bm25(text)
    assert "break-even" in tokens or "break" in tokens
    assert "50" in tokens or "50000" in tokens
    assert len(tokens) >= 4


def test_bm25_and_hybrid_search(sample_chunks):
    engine = EmbeddingEngine()
    store = VectorStoreIndex(engine)
    store.build_from_chunks(sample_chunks)

    # 1. Test exact keyword BM25 retrieval
    bm25_res = store.search_bm25("Break-even Fixed Costs", top_k=2)
    assert len(bm25_res) >= 1
    assert bm25_res[0][0].chunk_id == "c3"

    # 2. Test exact formula BM25 retrieval
    math_res = store.search_bm25("softmax(QK^T / sqrt(d_k))", top_k=2)
    assert len(math_res) >= 1
    assert math_res[0][0].chunk_id == "c2"

    # 3. Test Hybrid Search (combines dense and sparse)
    hybrid_res = store.hybrid_search("blood-brain barrier pathogens", top_k=2)
    assert len(hybrid_res) >= 1
    assert hybrid_res[0][0].chunk_id == "c1"


def test_cross_encoder_reranker(sample_chunks):
    reranker = CrossEncoderReranker()
    candidates = [(c, 0.5) for c in sample_chunks]
    
    query = "What is the mathematical equation for break even point in marginal costing?"
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    # The marginal costing chunk should be ranked #1
    assert reranked[0][0].chunk_id == "c3"
    assert reranked[0][1] >= 0.0


def test_conversation_memory_rewriting():
    memory = ConversationMemory()
    
    # 1. Independent query should not be modified
    q1 = "What is an action potential in neuroscience?"
    res1 = memory.rewrite_query(q1, conversation_history=[])
    assert res1 == q1

    # 2. Follow-up query with pronoun should be rewritten
    history = [
        {"role": "user", "content": "What is the blood-brain barrier?"},
        {"role": "assistant", "content": "The blood-brain barrier protects the brain from toxins."},
    ]
    followup = "What are its main functions and limitations?"
    res2 = memory.rewrite_query(followup, conversation_history=history)
    assert "blood-brain" in res2.lower() or "barrier" in res2.lower()
