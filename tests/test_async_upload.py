"""Test Suite for Asynchronous Upload, Status Polling, and 200MB Config."""

import io
import json
import os
import time
import pytest

from app import app
from scripts.generate_corpus import create_pdf


@pytest.fixture
def client(tmp_path):
    import shutil
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path / "documents")
    app.config["CHUNKS_PATH"] = str(tmp_path / "data" / "processed_chunks.json")
    app.config["INDEX_DIR"] = str(tmp_path / "data" / "faiss_index")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INDEX_DIR"], exist_ok=True)

    base_chunks = os.path.abspath("./data/processed_chunks.json")
    base_index = os.path.abspath("./data/faiss_index")
    if os.path.exists(base_chunks):
        shutil.copy2(base_chunks, app.config["CHUNKS_PATH"])
    if os.path.exists(base_index):
        for f in os.listdir(base_index):
            shutil.copy2(os.path.join(base_index, f), os.path.join(app.config["INDEX_DIR"], f))

    with app.test_client() as client:
        yield client


def test_max_content_length_configuration():
    assert app.config["MAX_CONTENT_LENGTH"] >= 200 * 1024 * 1024


def test_supported_extensions():
    assert "pdf" in app.config["ALLOWED_EXTENSIONS"]
    assert "docx" in app.config["ALLOWED_EXTENSIONS"]
    assert "pptx" in app.config["ALLOWED_EXTENSIONS"]
    assert "txt" in app.config["ALLOWED_EXTENSIONS"]


def test_async_upload_flow(client, tmp_path):
    """Tests initiating an async upload and polling the status endpoint until completion."""
    test_pdf_path = os.path.join(tmp_path, "async_test_sample.pdf")
    doc_info = {
        "filename": "async_test_sample.pdf",
        "title": "Async Task Verification Document",
        "pages": [
            [
                (
                    "Section A.1: Real-time Progress Tracking",
                    "This is an async test document verifying that task_id generation, background worker execution, and real-time status polling eliminate frontend network timeouts for large document ingestion.",
                )
            ]
        ],
    }
    create_pdf(doc_info, test_pdf_path)

    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    data = {
        "file": (io.BytesIO(pdf_bytes), "async_test_sample.pdf"),
        "async": "true",
    }
    response = client.post(
        "/upload",
        data=data,
        headers={"X-Async-Upload": "true"},
    )

    assert response.status_code == 202
    res_data = response.get_json()
    assert res_data["status"] == "processing"
    assert "task_id" in res_data
    task_id = res_data["task_id"]

    # Poll status until completed or timeout
    max_wait_seconds = 30
    start_time = time.time()
    completed = False

    while time.time() - start_time < max_wait_seconds:
        status_res = client.get(f"/api/upload-status/{task_id}")
        assert status_res.status_code == 200
        status_data = status_res.get_json()
        assert status_data["task_id"] == task_id
        assert "progress_pct" in status_data
        assert "stage" in status_data

        if status_data["status"] == "completed":
            completed = True
            assert status_data["progress_pct"] == 100
            assert status_data["result"] is not None
            assert status_data["result"]["total_chunks"] >= 1
            break
        elif status_data["status"] == "error":
            pytest.fail(f"Async task failed: {status_data.get('message')}")

        time.sleep(0.5)

    assert completed, "Async upload task did not complete within timeout."
