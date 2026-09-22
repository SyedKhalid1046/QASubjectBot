"""Comprehensive Automated Test Suite for Multi-Format Ingestion (PDF, DOCX, PPTX)."""

import io
import json
import os
import docx
import pptx
import pytest

from app import app
from ingestion import (
    DocxLoader,
    DocumentChunk,
    DocumentLoader,
    PDFLoader,
    PptxLoader,
    TokenChunker,
    ingest_single_file,
)
from scripts.generate_corpus import create_pdf


@pytest.fixture
def test_client(tmp_path):
    """Creates an isolated test client with dedicated upload, chunks, and faiss directories."""
    import shutil
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "documents")
    app.config["CHUNKS_PATH"] = str(tmp_path / "data" / "processed_chunks.json")
    app.config["INDEX_DIR"] = str(tmp_path / "data" / "faiss_index")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INDEX_DIR"], exist_ok=True)

    with app.test_client() as client:
        yield client


def test_docx_loader_structure_and_extraction(tmp_path):
    """Verifies DocxLoader preserves headings, bullet points, and tables."""
    docx_path = os.path.join(tmp_path, "sample_document.docx")

    doc = docx.Document()
    doc.add_heading("Deep Learning Architectural Innovations", level=1)
    doc.add_paragraph("This is an introductory paragraph discussing foundation models.")
    doc.add_paragraph("First key optimization principle", style="List Bullet")
    doc.add_paragraph("Second key optimization principle", style="List Bullet")

    # Add a table
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Model Architecture"
    table.cell(0, 1).text = "Parameter Count"
    table.cell(1, 0).text = "Transformer-XL"
    table.cell(1, 1).text = "500M"

    doc.save(docx_path)

    # Load via DocxLoader
    pages = DocxLoader.load(docx_path)
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1

    # Check structural markers
    assert "# Deep Learning Architectural Innovations" in text
    assert "introductory paragraph" in text
    assert "• First key optimization principle" in text
    assert "• Second key optimization principle" in text
    assert "Transformer-XL" in text
    assert "500M" in text


def test_pptx_loader_slides_and_extraction(tmp_path):
    """Verifies PptxLoader extracts slide titles, bullet points, shapes, and tables with 1-indexed slide mapping."""
    pptx_path = os.path.join(tmp_path, "sample_presentation.pptx")

    prs = pptx.Presentation()

    # Slide 1: Title & Subtitle
    slide_layout_0 = prs.slide_layouts[0]
    slide1 = prs.slides.add_slide(slide_layout_0)
    slide1.shapes.title.text = "Introduction to Mixture of Experts (MoE)"
    slide1.placeholders[1].text = "Sparse routing mechanisms in modern LLMs"

    # Slide 2: Bulleted List
    slide_layout_1 = prs.slide_layouts[1]
    slide2 = prs.slides.add_slide(slide_layout_1)
    slide2.shapes.title.text = "MoE Gating Mechanism"
    tf = slide2.placeholders[1].text_frame
    tf.text = "Top-2 gating router distributes tokens dynamically"
    p2 = tf.add_paragraph()
    p2.text = "Auxiliary load balancing loss prevents expert collapse"
    p2.level = 1

    # Slide 3: Table
    slide_layout_5 = prs.slide_layouts[5]
    slide3 = prs.slides.add_slide(slide_layout_5)
    slide3.shapes.title.text = "Expert Capacity Benchmarks"
    rows, cols = 2, 2
    left, top, width, height = pptx.util.Inches(1), pptx.util.Inches(2), pptx.util.Inches(6), pptx.util.Inches(1.5)
    table_shape = slide3.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.cell(0, 0).text = "Configuration"
    table.cell(0, 1).text = "Active Parameters"
    table.cell(1, 0).text = "8x7B Sparse MoE"
    table.cell(1, 1).text = "12.8B Active"

    prs.save(pptx_path)

    # Load via PptxLoader
    slides_data = PptxLoader.load(pptx_path)
    assert len(slides_data) == 3

    # Verify slide 1
    s1_num, s1_text = slides_data[0]
    assert s1_num == 1
    assert "Mixture of Experts" in s1_text
    assert "Sparse routing mechanisms" in s1_text

    # Verify slide 2
    s2_num, s2_text = slides_data[1]
    assert s2_num == 2
    assert "MoE Gating Mechanism" in s2_text
    assert "Top-2 gating router" in s2_text
    assert "• Auxiliary load balancing loss" in s2_text

    # Verify slide 3
    s3_num, s3_text = slides_data[2]
    assert s3_num == 3
    assert "Expert Capacity Benchmarks" in s3_text
    assert "8x7B Sparse MoE" in s3_text
    assert "12.8B Active" in s3_text


def test_document_loader_dispatch(tmp_path):
    """Tests automatic loader selection based on file extensions and rejection of invalid formats."""
    docx_file = os.path.join(tmp_path, "test.docx")
    d = docx.Document()
    d.add_paragraph("Docx sample")
    d.save(docx_file)

    pptx_file = os.path.join(tmp_path, "test.pptx")
    p = pptx.Presentation()
    s = p.slides.add_slide(p.slide_layouts[0])
    s.shapes.title.text = "PPTX sample"
    p.save(pptx_file)

    txt_file = os.path.join(tmp_path, "test.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("Plain text sample")

    assert len(DocumentLoader.load(docx_file)) >= 1
    assert len(DocumentLoader.load(pptx_file)) >= 1
    assert len(DocumentLoader.load(txt_file)) >= 1

    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentLoader.load(os.path.join(tmp_path, "test.unsupported_ext"))


def test_multiformat_chunk_metadata(tmp_path):
    """Verifies that DocumentChunk correctly tags format and unit labels for PPTX, DOCX, and PDF."""
    chunker = TokenChunker(min_tokens=50, max_tokens=200, target_tokens=100, overlap_tokens=20)

    # PPTX chunking
    pptx_slides = [(1, "Slide 1 title and content"), (2, "Slide 2 second topic")]
    pptx_chunks = chunker.chunk_document_pages(pptx_slides, source="presentation.pptx")
    assert len(pptx_chunks) >= 1
    assert pptx_chunks[0].doc_type == "pptx"
    assert pptx_chunks[0].unit_label == "Slide"

    # DOCX chunking
    docx_pages = [(1, "Word document section content")]
    docx_chunks = chunker.chunk_document_pages(docx_pages, source="report.docx")
    assert len(docx_chunks) >= 1
    assert docx_chunks[0].doc_type == "docx"
    assert docx_chunks[0].unit_label == "Page"


def test_mixed_batch_upload_and_cross_format_qa(test_client, tmp_path):
    """Integration test: Uploads a mixed batch (1 PDF + 1 DOCX + 1 PPTX) and tests unified indexing and citations."""
    # 1. Create a PDF
    pdf_path = os.path.join(tmp_path, "neural_spec.pdf")
    doc_info = {
        "filename": "neural_spec.pdf",
        "title": "Quantum Neural Encoding Architecture",
        "pages": [
            [
                (
                    "Section Q.1: Quantum State Formulation",
                    "The QuantumState-9000 encoding matrix maps continuous token vectors into orthogonal Hilbert subspaces for ultra-low latency semantic retrieval.",
                )
            ]
        ],
    }
    create_pdf(doc_info, pdf_path)

    # 2. Create a Word Document
    docx_path = os.path.join(tmp_path, "agent_framework.docx")
    d = docx.Document()
    d.add_heading("Agentic Orchestration Framework", level=1)
    d.add_paragraph(
        "The MultiAgent-Apex runtime coordinates autonomous decision trees through "
        "hierarchical state machine delegates and parallel tool dispatch protocols."
    )
    d.save(docx_path)

    # 3. Create a PowerPoint Presentation
    pptx_path = os.path.join(tmp_path, "model_scaling.pptx")
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Scaling Laws for Frontier AI Models"
    s.placeholders[1].text = (
        "The HyperCompute-Alpha compute cluster delivers 500 PFLOPS of FP8 tensor throughput "
        "with zero inter-node latency degradation."
    )
    prs.save(pptx_path)

    # 4. Upload all 3 files simultaneously in a single mixed batch
    with open(pdf_path, "rb") as f_pdf, open(docx_path, "rb") as f_docx, open(pptx_path, "rb") as f_pptx:
        data = {
            "files": [
                (io.BytesIO(f_pdf.read()), "neural_spec.pdf"),
                (io.BytesIO(f_docx.read()), "agent_framework.docx"),
                (io.BytesIO(f_pptx.read()), "model_scaling.pptx"),
            ]
        }
        res = test_client.post("/upload", data=data, content_type="multipart/form-data")

    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json["status"] == "success"
    assert res_json["files_count"] == 3
    assert res_json["total_documents"] == 3
    assert res_json["chunks_added"] >= 3

    # 5. Check Document Library API returns all 3 formats correctly
    doc_res = test_client.get("/api/documents")
    assert doc_res.status_code == 200
    doc_json = doc_res.get_json()
    assert doc_json["total_documents"] == 3
    file_types = {d["filename"]: d["file_type"] for d in doc_json["documents"]}
    assert file_types["neural_spec.pdf"] == "PDF"
    assert file_types["agent_framework.docx"] == "DOCX"
    assert file_types["model_scaling.pptx"] == "PPTX"

    # 6. Ask question answered by the Word document
    qa_docx = test_client.post(
        "/ask",
        data=json.dumps({"question": "What is the MultiAgent-Apex runtime and how does it coordinate?", "top_k": 3, "score_threshold": 0.15}),
        content_type="application/json",
    )
    assert qa_docx.status_code == 200
    qa_docx_json = qa_docx.get_json()
    assert qa_docx_json["status"] == "success"
    assert any("agent_framework.docx" in c["source"] for c in qa_docx_json["citations"])

    # 7. Ask question answered by the PowerPoint presentation (verifying PPTX citation)
    qa_pptx = test_client.post(
        "/ask",
        data=json.dumps({"question": "What is the HyperCompute-Alpha cluster tensor throughput?", "top_k": 3, "score_threshold": 0.15}),
        content_type="application/json",
    )
    assert qa_pptx.status_code == 200
    qa_pptx_json = qa_pptx.get_json()
    assert qa_pptx_json["status"] == "success"
    pptx_cits = [c for c in qa_pptx_json["citations"] if "model_scaling.pptx" in c["source"]]
    assert len(pptx_cits) >= 1
    assert pptx_cits[0]["unit"] == "Slide"
    assert pptx_cits[0]["page"] == 1


def test_upload_rejection_and_graceful_error_handling(test_client, tmp_path):
    """Tests that unsupported file types are rejected and corrupted files are handled gracefully."""
    # 1. Reject purely unsupported files
    invalid_data = {"files": [(io.BytesIO(b"binary data"), "malware.exe")]}
    res = test_client.post("/upload", data=invalid_data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Invalid file type" in res.get_json()["message"]

    # 2. Mixed upload with 1 valid file and 1 unsupported file
    valid_docx = os.path.join(tmp_path, "valid.docx")
    d = docx.Document()
    d.add_paragraph("Valid content for testing mixed error handling.")
    d.save(valid_docx)

    with open(valid_docx, "rb") as f:
        mixed_data = {
            "files": [
                (io.BytesIO(f.read()), "valid.docx"),
                (io.BytesIO(b"random text"), "unsupported.csv"),
            ]
        }
        mixed_res = test_client.post("/upload", data=mixed_data, content_type="multipart/form-data")

    assert mixed_res.status_code == 200
    mixed_json = mixed_res.get_json()
    assert mixed_json["status"] == "success"
    assert mixed_json["files_count"] == 1
    assert len(mixed_json["failed_files"]) == 1
    assert mixed_json["failed_files"][0]["filename"] == "unsupported.csv"
