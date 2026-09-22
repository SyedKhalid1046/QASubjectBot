"""Automated Test Suite for QASubjectBot Flask Web Application."""

import io
import json
import os
import pytest

from app import app
from scripts.generate_corpus import create_pdf


@pytest.fixture
def client(tmp_path):
    """Creates a Flask test client with isolated upload, chunks, and index directories."""
    import shutil
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "documents")
    app.config["CHUNKS_PATH"] = str(tmp_path / "data" / "processed_chunks.json")
    app.config["INDEX_DIR"] = str(tmp_path / "data" / "faiss_index")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INDEX_DIR"], exist_ok=True)

    # Initialize isolated test environment with baseline corpus
    base_chunks = os.path.abspath("./data/processed_chunks.json")
    base_index = os.path.abspath("./data/faiss_index")
    if os.path.exists(base_chunks):
        shutil.copy2(base_chunks, app.config["CHUNKS_PATH"])
    if os.path.exists(base_index):
        for f in os.listdir(base_index):
            shutil.copy2(os.path.join(base_index, f), os.path.join(app.config["INDEX_DIR"], f))

    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"QASubjectBot" in response.data
    assert b"Submit Document" in response.data
    assert b"Knowledge Base" in response.data


def test_ask_route_valid_question(client):
    payload = {
        "question": "What is Magnetic Resonance Imaging (MRI) and brain imaging?",
        "top_k": 3,
        "score_threshold": 0.10,
    }
    response = client.post(
        "/ask",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "answer" in data
    assert len(data["citations"]) >= 1
    assert data["retrieval_confidence"] > 0.10


def test_ask_route_out_of_scope(client):
    payload = {
        "question": "What is the average surface temperature on Mars during winter?",
        "top_k": 3,
        "score_threshold": 0.25,
    }
    response = client.post(
        "/ask",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "insufficient_context"
    assert "don't have enough information" in data["answer"].lower()
    assert len(data["citations"]) == 0


def test_ask_route_empty_question(client):
    payload = {"question": "   "}
    response = client.post(
        "/ask",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_documents_api(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["total_documents"] >= 3
    assert data["total_chunks"] >= 50
    assert len(data["documents"]) == data["total_documents"]


def test_upload_route_invalid_extension(client):
    data = {"file": (io.BytesIO(b"dummy text"), "test.exe")}
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    result = response.get_json()
    assert result["status"] == "error"
    assert "Invalid file type" in result["message"]


def test_upload_route_replaces_single_pdf(client, tmp_path):
    """Tests uploading a single PDF replaces the previous corpus completely."""
    # Generate a small valid PDF in temp path
    test_pdf_path = os.path.join(tmp_path, "upload_test_sample.pdf")
    doc_info = {
        "filename": "upload_test_sample.pdf",
        "title": "Upload Verification Document on QuantumX789",
        "pages": [
            [
                (
                    "Section U.1: QuantumX789 Protocol",
                    "This is an uploaded verification document covering special synthetic test terms: QuantumX789 tokenization protocol. It guarantees dynamic FAISS vector index re-indexing without restarting the Flask web server runtime.",
                )
            ]
        ],
    }
    create_pdf(doc_info, test_pdf_path)

    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    data = {"file": (io.BytesIO(pdf_bytes), "upload_test_sample.pdf")}
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "success"
    assert "Replaced knowledge base with 1 new document(s)" in res_data["message"]
    assert res_data["total_documents"] == 1
    assert res_data["chunks_added"] >= 1

    # Verify that the Indexed Corpus Library now reflects ONLY the single newly uploaded document
    doc_api_res = client.get("/api/documents")
    doc_api_data = doc_api_res.get_json()
    assert doc_api_data["total_documents"] == 1
    assert doc_api_data["documents"][0]["filename"] == "upload_test_sample.pdf"

    # Verify that the new document is queryable immediately!
    ask_payload = {
        "question": "What is the QuantumX789 tokenization protocol in uploaded document?",
        "top_k": 3,
        "score_threshold": 0.20,
    }
    ask_response = client.post(
        "/ask",
        data=json.dumps(ask_payload),
        content_type="application/json",
    )
    assert ask_response.status_code == 200
    ask_data = ask_response.get_json()
    assert ask_data["status"] == "success"
    assert any("upload_test_sample.pdf" in c["source"] for c in ask_data["citations"])

    # Verify that previous corpus questions now return insufficient context (refusal)
    old_topic_payload = {
        "question": "What is Magnetic Resonance Imaging (MRI) and brain imaging?",
        "top_k": 3,
        "score_threshold": 0.25,
    }
    old_topic_res = client.post(
        "/ask",
        data=json.dumps(old_topic_payload),
        content_type="application/json",
    )
    assert old_topic_res.status_code == 200
    old_topic_data = old_topic_res.get_json()
    assert old_topic_data["status"] == "insufficient_context"


def test_upload_route_batch_multiple_pdfs_replaces_corpus(client, tmp_path):
    """Tests uploading a batch of 10 PDFs simultaneously replaces whatever was previously in the corpus."""
    num_batch_files = 10
    file_tuples = []

    for i in range(1, num_batch_files + 1):
        fname = f"batch_doc_{i:02d}.pdf"
        fpath = os.path.join(tmp_path, fname)
        doc_info = {
            "filename": fname,
            "title": f"Batch Volume {i}: Distributed AI Training Cluster Node {i}",
            "pages": [
                [
                    (
                        f"Section B.{i}: SuperCluster Protocol #{i}",
                        f"Node identifier #{i} executes the HyperBatch-{i*100} synchronization algorithm for distributed deep learning models with zero communication bottlenecks.",
                    )
                ]
            ],
        }
        create_pdf(doc_info, fpath)

        with open(fpath, "rb") as f:
            pdf_bytes = f.read()
        file_tuples.append((io.BytesIO(pdf_bytes), fname))

    # Send multi-file upload payload
    data = {"files": file_tuples}
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data["status"] == "success"
    assert "Replaced knowledge base with 10 new document(s)" in res_data["message"]
    assert res_data["files_count"] == num_batch_files
    assert res_data["total_documents"] == num_batch_files
    assert res_data["chunks_added"] >= num_batch_files

    # Verify document library API reflects exactly 10 documents
    doc_api_res = client.get("/api/documents")
    doc_api_data = doc_api_res.get_json()
    assert doc_api_data["total_documents"] == 10
    assert len(doc_api_data["documents"]) == 10

    # Verify querying across the new batch
    ask_payload = {
        "question": "What is the HyperBatch-500 synchronization algorithm?",
        "top_k": 3,
        "score_threshold": 0.20,
    }
    ask_response = client.post(
        "/ask",
        data=json.dumps(ask_payload),
        content_type="application/json",
    )
    assert ask_response.status_code == 200
    ask_data = ask_response.get_json()
    assert ask_data["status"] == "success"
    assert any("batch_doc_05.pdf" in c["source"] for c in ask_data["citations"])

