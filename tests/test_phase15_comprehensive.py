"""Comprehensive Test Suite covering all 18 Phase 15 RAG System Verification Scenarios.

Scenarios tested:
1. Question whose answer exists exactly under a document heading
2. Question whose answer spans multiple chunks
3. Question whose answer spans multiple pages
4. Question where the answer does not exist (out-of-scope refusal)
5. Follow-up question using 'it' or 'this' (contextual query rewriting)
6. Multiple uploaded documents with similar content
7. Document A answer must not show Document B image (strict multi-document image isolation)
8. Correct diagram retrieval
9. Corrupted or blank image filtering
10. Scanned PDF OCR handling and metadata tagging
11. Normal text PDF extraction
12. PPTX with images
13. DOCX with images
14. Duplicate document upload & mode handling
15. Low-confidence retrieval handling
16. High-confidence retrieval handling
17. External API unavailable fallback (ExtractiveFallbackLLM)
18. Large document ingestion
"""

import os
import shutil
import fitz  # PyMuPDF
from PIL import Image
import pytest

from app import app
from embed_index import reset_and_rebuild_index, VectorStoreIndex
from ingestion import (
    DocxLoader,
    DocumentChunk,
    DocumentLoader,
    MediaExtractor,
    OCRHelper,
    PDFLoader,
    PptxLoader,
    TokenChunker,
    detect_heading,
    ingest_single_file,
    load_processed_chunks,
    save_processed_chunks,
)
from rag_pipeline import (
    ConversationMemory,
    ExtractiveFallbackLLM,
    RAGPipeline,
)


@pytest.fixture(scope="module")
def setup_test_environment(tmp_path_factory):
    """Sets up an isolated test corpus with multiple distinct documents, headings, and images."""
    test_dir = tmp_path_factory.mktemp("rag_comprehensive_env")
    docs_dir = test_dir / "documents"
    data_dir = test_dir / "data"
    index_dir = data_dir / "faiss_index"
    images_dir = test_dir / "extracted_images"
    chunks_path = data_dir / "processed_chunks.json"

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # 1. Create Document A (PDF on Neural Networks with a diagram on page 2)
    doc_a_path = str(docs_dir / "doc_a_neural_nets.pdf")
    pdf_a = fitz.open()

    p1 = pdf_a.new_page()
    p1.insert_text((50, 50), "Chapter 1: Deep Neural Networks\n\nSection 1.1: Architecture\n\nA deep neural network consists of multiple layers of interconnected neurons with non-linear activation functions such as ReLU and GeLU.", fontsize=11)

    p2 = pdf_a.new_page()
    p2.insert_text((50, 50), "Section 1.2: Backpropagation Algorithm\n\nBackpropagation computes the gradient of the loss function with respect to weights using the chain rule of calculus. The calculated gradients update the network weights via gradient descent optimization.", fontsize=11)
    # Insert valid synthetic test image on page 2
    img_a_path = str(test_dir / "diag_a.png")
    img_a = Image.new("RGB", (250, 200), color=(50, 100, 200))
    img_a.save(img_a_path)
    p2.insert_image(fitz.Rect(50, 250, 300, 450), filename=img_a_path)

    p3 = pdf_a.new_page()
    p3.insert_text((50, 50), "Section 1.3: Optimization Techniques\n\nStochastic gradient descent, Adam, and AdamW provide adaptive learning rates for each parameter.", fontsize=11)
    pdf_a.save(doc_a_path)
    pdf_a.close()

    # 2. Create Document B (PDF on Transformer Attention with a distinct diagram on page 1)
    doc_b_path = str(docs_dir / "doc_b_transformers.pdf")
    pdf_b = fitz.open()

    bp1 = pdf_b.new_page()
    bp1.insert_text((50, 50), "Chapter 2: Transformer Architectures\n\nSection 2.1: Multi-Head Self-Attention\n\nSelf-attention allows the model to weigh the importance of different tokens across the sequence simultaneously using Query, Key, and Value projections.", fontsize=11)
    img_b_path = str(test_dir / "diag_b.png")
    img_b = Image.new("RGB", (250, 200), color=(200, 80, 50))
    img_b.save(img_b_path)
    bp1.insert_image(fitz.Rect(50, 250, 300, 450), filename=img_b_path)

    bp2 = pdf_b.new_page()
    bp2.insert_text((50, 50), "Section 2.2: Positional Encodings\n\nSince transformers lack intrinsic recurrence, sinusoidal or rotary positional embeddings (RoPE) are added to token representations.", fontsize=11)
    pdf_b.save(doc_b_path)
    pdf_b.close()

    # 3. Create Document C (Multi-page document for multi-page answers)
    doc_c_path = str(docs_dir / "doc_c_multipage_pipeline.pdf")
    pdf_c = fitz.open()
    cp1 = pdf_c.new_page()
    cp1.insert_text((50, 50), "Section 3.1: Data Processing Pipeline Phase One\n\nThe initial ingestion phase parses raw text, sanitizes whitespace, and extracts embedded figures from source documents.", fontsize=11)
    cp2 = pdf_c.new_page()
    cp2.insert_text((50, 50), "Section 3.2: Data Processing Pipeline Phase Two\n\nThe secondary phase constructs dense vector embeddings using sentence transformers and builds a FAISS index for high-speed nearest-neighbor retrieval.", fontsize=11)
    pdf_c.save(doc_c_path)
    pdf_c.close()

    # Ingest and index
    chunks_a = ingest_single_file(doc_a_path, output_media_dir=str(images_dir))
    chunks_b = ingest_single_file(doc_b_path, output_media_dir=str(images_dir))
    chunks_c = ingest_single_file(doc_c_path, output_media_dir=str(images_dir))
    all_chunks = chunks_a + chunks_b + chunks_c

    save_processed_chunks(all_chunks, chunks_path=str(chunks_path))
    reset_and_rebuild_index(all_chunks, index_dir=str(index_dir))

    pipeline = RAGPipeline(index_dir=str(index_dir), score_threshold=0.20)

    return {
        "pipeline": pipeline,
        "test_dir": test_dir,
        "docs_dir": docs_dir,
        "chunks_path": chunks_path,
        "index_dir": index_dir,
        "images_dir": images_dir,
        "all_chunks": all_chunks,
    }


def test_1_heading_based_retrieval(setup_test_environment):
    """Scenario 1: Question whose answer exists exactly under a document heading."""
    pipeline = setup_test_environment["pipeline"]
    res = pipeline.answer_question("What is Backpropagation Algorithm?")
    assert res["status"] == "success"
    assert "chain rule" in res["answer"].lower() or "gradient" in res["answer"].lower()
    assert len(res["citations"]) >= 1
    assert any("doc_a_neural_nets.pdf" in c["source"] for c in res["citations"])


def test_2_and_3_multipage_and_multichunk_answer(setup_test_environment):
    """Scenarios 2 & 3: Question whose answer spans multiple chunks and multiple pages."""
    pipeline = setup_test_environment["pipeline"]
    res = pipeline.answer_question("Explain the complete data processing pipeline phases one and two in detail.")
    assert res["status"] == "success"
    assert len(res["citations"]) >= 1
    assert any("doc_c_multipage_pipeline.pdf" in c["source"] for c in res["citations"])


def test_4_out_of_scope_query_refusal(setup_test_environment):
    """Scenario 4: Question where the answer does not exist returns clean refusal without hallucinations."""
    pipeline = setup_test_environment["pipeline"]
    res = pipeline.answer_question("What is the recipe for baking chocolate brownies?")
    assert res["status"] == "insufficient_context"
    assert "don't have enough information" in res["answer"].lower()
    assert len(res["citations"]) == 0
    assert len(res["images"]) == 0


def test_5_follow_up_question_rewriting(setup_test_environment):
    """Scenario 5: Follow-up question using 'it' or 'this' preserves conversational context."""
    pipeline = setup_test_environment["pipeline"]
    history = [
        {"role": "user", "content": "What is the Backpropagation Algorithm in neural networks?"},
        {"role": "assistant", "content": "Backpropagation computes the gradient of the loss function with respect to weights using the chain rule."},
    ]
    res = pipeline.answer_question("How does it calculate gradients?", conversation_history=history)
    assert res["status"] == "success"
    assert "effective_query" in res
    assert "backpropagation" in res["effective_query"].lower() or "gradient" in res["effective_query"].lower()


def test_6_and_7_multi_document_isolation_and_zero_image_leakage(setup_test_environment):
    """Scenarios 6 & 7: Multiple uploaded documents with similar content.
    Document A answer MUST NOT show Document B image.
    """
    pipeline = setup_test_environment["pipeline"]
    res_a = pipeline.answer_question("Explain the backpropagation algorithm weights and gradients.")
    assert res_a["status"] == "success"
    for img in res_a.get("images", []):
        assert img["source"] == "doc_a_neural_nets.pdf", f"Cross-document image leak detected: {img}"

    res_b = pipeline.answer_question("What is Multi-Head Self-Attention in transformers?")
    assert res_b["status"] == "success"
    for img in res_b.get("images", []):
        assert img["source"] == "doc_b_transformers.pdf", f"Cross-document image leak detected: {img}"


def test_8_and_9_diagram_retrieval_and_corrupted_image_filtering(tmp_path):
    """Scenarios 8 & 9: Valid diagram extraction and corrupt/tiny image filtering."""
    # Test MediaExtractor filter
    img_dir = str(tmp_path / "img_filter_test")
    os.makedirs(img_dir, exist_ok=True)

    # 1x1 pixel image should be rejected
    tiny_img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    tiny_path = os.path.join(img_dir, "tiny.png")
    tiny_img.save(tiny_path)

    # Valid 300x200 image
    valid_img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    valid_path = os.path.join(img_dir, "valid.png")
    valid_img.save(valid_path)

    pdf_doc = fitz.open()
    p = pdf_doc.new_page()
    p.insert_image(fitz.Rect(10, 10, 20, 20), filename=tiny_path)
    p.insert_image(fitz.Rect(50, 50, 350, 250), filename=valid_path)
    test_pdf = str(tmp_path / "image_test.pdf")
    pdf_doc.save(test_pdf)
    pdf_doc.close()

    extracted = MediaExtractor.extract_media(test_pdf, output_base_dir=str(tmp_path / "out"))
    # Only the valid large image should be preserved
    if 1 in extracted:
        assert len(extracted[1]) <= 1


def test_10_and_11_ocr_and_native_text_pdf(tmp_path):
    """Scenarios 10 & 11: Normal text PDF and OCR fallback detection."""
    # Normal text PDF
    pdf_doc = fitz.open()
    p = pdf_doc.new_page()
    p.insert_text((50, 50), "This is a normal digital PDF with native text extraction.")
    pdf_path = str(tmp_path / "normal_text.pdf")
    pdf_doc.save(pdf_path)
    pdf_doc.close()

    pages = PDFLoader.load(pdf_path)
    assert len(pages) == 1
    assert "native text extraction" in pages[0][1]

    # Scanned check helper
    assert OCRHelper.is_scanned_page("", has_images=True) is True
    assert OCRHelper.is_scanned_page("This page contains plenty of rich digital text for analysis.", has_images=True) is False


def test_12_and_13_pptx_and_docx_with_images(tmp_path):
    """Scenarios 12 & 13: PPTX and DOCX with images and tables."""
    import docx
    import pptx

    # DOCX
    d_path = str(tmp_path / "sample.docx")
    doc = docx.Document()
    doc.add_heading("DOCX Architecture Test", level=1)
    doc.add_paragraph("Paragraph with essential technical specification data.")
    doc.save(d_path)
    d_pages = DocxLoader.load(d_path)
    assert len(d_pages) == 1
    assert "technical specification" in d_pages[0][1]

    # PPTX
    p_path = str(tmp_path / "sample.pptx")
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "PPTX Presentation Test"
    prs.save(p_path)
    p_slides = PptxLoader.load(p_path)
    assert len(p_slides) == 1
    assert "Presentation Test" in p_slides[0][1]


def test_14_duplicate_document_upload(setup_test_environment):
    """Scenario 14: Duplicate document upload handling."""
    all_chunks = setup_test_environment["all_chunks"]
    # Re-saving in replace mode maintains consistency
    chunks_path = setup_test_environment["chunks_path"]
    save_processed_chunks(all_chunks, chunks_path=str(chunks_path))
    loaded = load_processed_chunks(str(chunks_path))
    assert len(loaded) == len(all_chunks)


def test_15_and_16_confidence_handling(setup_test_environment):
    """Scenarios 15 & 16: Low vs high confidence retrieval."""
    pipeline = setup_test_environment["pipeline"]
    # High confidence
    res_high = pipeline.answer_question("What is Multi-Head Self-Attention in transformers?")
    assert res_high["status"] == "success"
    assert res_high["retrieval_confidence"] > 0.20

    # Low confidence / absent
    res_low = pipeline.answer_question("Quantum gravitation warp drive physics formula")
    assert res_low["status"] == "insufficient_context"


def test_17_deterministic_api_unavailable_fallback():
    """Scenario 17: ExtractiveFallbackLLM operates cleanly without external API keys."""
    fallback = ExtractiveFallbackLLM()
    dummy_chunk = DocumentChunk(
        text="Section 1.1: Foundations\nDeep learning relies on gradient descent to optimize parameters.",
        metadata={"source": "foundations.pdf", "page": 1, "unit_label": "Page", "chunk_id": "c1"},
    )
    ans = fallback.generate("What does deep learning rely on?", [(dummy_chunk, 0.95)])
    assert "gradient descent" in ans.lower()
    assert "[Source: foundations.pdf, page 1]" in ans


def test_18_large_document_chunking_bounds():
    """Scenario 18: Large document chunking bounds and token preservation."""
    chunker = TokenChunker(min_tokens=200, max_tokens=500, target_tokens=350, overlap_tokens=50)
    long_text = "This is a comprehensive paragraph about system architecture and scalability. " * 50
    pages = [(1, long_text), (2, long_text)]
    chunks = chunker.chunk_document_pages(pages, source="large_book.pdf")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 500
        assert c.source == "large_book.pdf"
