"""Flask Web Application for QASubjectBot.

Provides:
- Standalone Flask web server with custom responsive HTML/CSS/JavaScript UI.
- Asynchronous chunked document uploading (up to 200MB, PDF, DOCX, PPTX, TXT) with real-time status polling & progress tracking.
- Memory-efficient streaming of large PDF books with PyMuPDF and OCR fallback.
- Hybrid Search (FAISS dense vectors + BM25 sparse keywords) and Cross-Encoder reranking.
- Multi-turn conversation memory with contextual follow-up query rewriting.
- Live corpus explorer and document listing API.
"""

import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from embed_index import (
    add_chunks_to_index,
    reset_and_rebuild_index,
    VectorStoreIndex,
)
from ingestion import (
    append_chunks_to_processed_file,
    clear_directory,
    ingest_single_file,
    load_processed_chunks,
    save_processed_chunks,
    DocumentChunk,
)
from rag_pipeline import get_pipeline

app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "documents")
app.config["CHUNKS_PATH"] = os.path.join(PROJECT_ROOT, "data", "processed_chunks.json")
app.config["INDEX_DIR"] = os.path.join(PROJECT_ROOT, "data", "faiss_index")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload size
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "docx", "pptx", "txt"}
app.config["MAX_UPLOAD_FILES"] = 15

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["INDEX_DIR"], exist_ok=True)

# In-memory thread-safe task registry for async uploads
UPLOAD_TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


def allowed_file(filename: str) -> bool:
    """Validates that uploaded file has an allowed extension (.pdf, .docx, .pptx, .txt)."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def update_task_progress(
    task_id: str,
    status: str,
    progress_pct: int,
    stage: str,
    message: str,
    current_file: str = "",
    current_page: int = 0,
    total_pages: int = 0,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    with TASKS_LOCK:
        if task_id in UPLOAD_TASKS:
            UPLOAD_TASKS[task_id].update({
                "status": status,
                "progress_pct": max(0, min(100, progress_pct)),
                "stage": stage,
                "message": message,
                "current_file": current_file or UPLOAD_TASKS[task_id].get("current_file", ""),
                "current_page": current_page,
                "total_pages": total_pages,
                "updated_at": time.time(),
            })
            if result:
                UPLOAD_TASKS[task_id]["result"] = result
            if error:
                UPLOAD_TASKS[task_id]["error"] = error


def _run_async_ingestion_task(
    task_id: str,
    saved_file_infos: List[Dict[str, str]],
    replace_mode: bool = True,
    initial_failed_files: Optional[List[Dict[str, str]]] = None,
):
    """Background worker thread that processes files, generates embeddings, and rebuilds FAISS & BM25 indexes."""
    try:
        total_files = len(saved_file_infos)
        processed_files = []
        failed_files = list(initial_failed_files or [])
        all_new_chunks: List[DocumentChunk] = []

        update_task_progress(
            task_id=task_id,
            status="processing",
            progress_pct=10,
            stage="extracting",
            message=f"Starting extraction for {total_files} file(s)...",
        )

        for f_idx, finfo in enumerate(saved_file_infos, 1):
            filepath = finfo["filepath"]
            filename = finfo["filename"]
            orig_name = finfo["orig_name"]

            def page_progress(stage: str, cur_p: int, tot_p: int, msg: str):
                # Scale progress between 10% and 65% across files
                file_weight = 55.0 / total_files
                base_pct = 10.0 + (f_idx - 1) * file_weight
                fraction = cur_p / max(tot_p, 1)
                curr_pct = int(base_pct + fraction * file_weight)
                update_task_progress(
                    task_id=task_id,
                    status="processing",
                    progress_pct=curr_pct,
                    stage=stage,
                    message=f"[{f_idx}/{total_files}] {orig_name}: {msg}",
                    current_file=orig_name,
                    current_page=cur_p,
                    total_pages=tot_p,
                )

            try:
                chunks = ingest_single_file(filepath, progress_callback=page_progress)
                if not chunks:
                    failed_files.append({
                        "filename": orig_name,
                        "reason": "No readable text chunks could be extracted.",
                    })
                    continue

                all_new_chunks.extend(chunks)
                processed_files.append({
                    "filename": orig_name,
                    "saved_name": filename,
                    "chunks": len(chunks),
                    "file_type": os.path.splitext(orig_name)[1].lstrip(".").upper(),
                })
            except Exception as e:
                failed_files.append({"filename": orig_name, "reason": str(e)})

        if not all_new_chunks:
            update_task_progress(
                task_id=task_id,
                status="error",
                progress_pct=100,
                stage="error",
                message="Failed to extract readable text chunks from the uploaded files.",
                error="No readable text chunks extracted.",
                result={"failed_files": failed_files},
            )
            return

        # Stage: Embedding & Indexing
        update_task_progress(
            task_id=task_id,
            status="processing",
            progress_pct=70,
            stage="embedding",
            message=f"Generating dense vectors & BM25 keywords for {len(all_new_chunks)} chunks...",
        )

        if replace_mode:
            save_processed_chunks(all_new_chunks, chunks_path=app.config["CHUNKS_PATH"])
            reset_and_rebuild_index(all_new_chunks, index_dir=app.config["INDEX_DIR"])
        else:
            all_chunks = append_chunks_to_processed_file(all_new_chunks, chunks_path=app.config["CHUNKS_PATH"])
            add_chunks_to_index(all_new_chunks, index_dir=app.config["INDEX_DIR"])

        update_task_progress(
            task_id=task_id,
            status="processing",
            progress_pct=95,
            stage="reloading",
            message="Reloading vector store into RAG pipeline...",
        )

        pipeline = get_pipeline()
        pipeline.reload_vector_store(index_dir=app.config["INDEX_DIR"])

        total_chunks = len(load_processed_chunks(app.config["CHUNKS_PATH"]))
        total_docs = len(set(c.source for c in load_processed_chunks(app.config["CHUNKS_PATH"])))

        if replace_mode:
            success_msg = (
                f"Replaced knowledge base with {len(processed_files)} new document(s) "
                f"({len(all_new_chunks)} chunks indexed)."
            )
        else:
            success_msg = (
                f"Appended {len(processed_files)} new document(s) to knowledge base "
                f"({len(all_new_chunks)} chunks indexed)."
            )
        if failed_files:
            success_msg += f" {len(failed_files)} file(s) could not be processed."

        final_result = {
            "status": "success",
            "message": success_msg,
            "filename": processed_files[0]["filename"] if len(processed_files) == 1 else f"{len(processed_files)} documents",
            "uploaded_files": processed_files,
            "failed_files": failed_files,
            "files_count": len(processed_files),
            "chunks_added": len(all_new_chunks),
            "total_chunks": total_chunks,
            "total_documents": total_docs,
        }

        update_task_progress(
            task_id=task_id,
            status="completed",
            progress_pct=100,
            stage="completed",
            message="Document indexing completed successfully!",
            result=final_result,
        )

    except Exception as ex:
        print(f"[ASYNC TASK ERROR] Task {task_id} failed: {ex}")
        update_task_progress(
            task_id=task_id,
            status="error",
            progress_pct=100,
            stage="error",
            message=f"Processing failed: {str(ex)}",
            error=str(ex),
        )


@app.route("/")
def index():
    """Renders the main QASubjectBot web application page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """Handles single or multi-file upload (PDF, DOCX, PPTX, TXT), supporting asynchronous background processing
    and real-time status reporting to eliminate frontend timeouts on large PDF books.
    """
    uploaded_file_objects = request.files.getlist("files") or request.files.getlist("file")
    valid_file_objects = [f for f in uploaded_file_objects if f and f.filename.strip() != ""]

    if not valid_file_objects:
        return jsonify({"status": "error", "message": "No files selected for upload."}), 400

    if len(valid_file_objects) > app.config["MAX_UPLOAD_FILES"]:
        return (
            jsonify({
                "status": "error",
                "message": (
                    f"Too many files selected ({len(valid_file_objects)}). "
                    f"Please upload up to {app.config['MAX_UPLOAD_FILES']} files at a time."
                ),
            }),
            400,
        )

    supported_files = [f for f in valid_file_objects if allowed_file(f.filename)]
    unsupported_files = [f for f in valid_file_objects if not allowed_file(f.filename)]

    if not supported_files:
        allowed_str = ", ".join(f".{ext}" for ext in sorted(app.config["ALLOWED_EXTENSIONS"]))
        failed_files = [
            {
                "filename": f.filename,
                "reason": f"Invalid file type. Supported formats are: {allowed_str}",
            }
            for f in unsupported_files
        ]
        return (
            jsonify({
                "status": "error",
                "message": f"Invalid file type. Supported formats are: {allowed_str}",
                "failed_files": failed_files,
            }),
            400,
        )

    # Determine mode: default is replace
    mode = request.form.get("mode", "replace").lower()
    replace_mode = mode != "append"

    upload_dir = app.config["UPLOAD_FOLDER"]
    if replace_mode:
        clear_directory(upload_dir)

    saved_file_infos = []
    failed_files = [
        {
            "filename": f.filename,
            "reason": f"Unsupported format. Accepted formats: {', '.join(sorted(app.config['ALLOWED_EXTENSIONS']))}",
        }
        for f in unsupported_files
    ]

    for file in supported_files:
        orig_name = file.filename
        filename = secure_filename(orig_name)
        ext = os.path.splitext(orig_name)[1].lower()
        if not filename:
            filename = f"uploaded_{int(time.time())}_{len(saved_file_infos)}{ext}"

        filepath = os.path.join(upload_dir, filename)
        try:
            file.save(filepath)
            saved_file_infos.append({
                "filepath": filepath,
                "filename": filename,
                "orig_name": orig_name,
            })
        except Exception as e:
            failed_files.append({"filename": orig_name, "reason": str(e)})

    if not saved_file_infos:
        return jsonify({"status": "error", "message": "Failed to save uploaded files.", "failed_files": failed_files}), 500

    # Check if client explicitly requests synchronous vs asynchronous processing
    is_async_req = (
        request.values.get("async", "").lower() in ("true", "1")
        or request.headers.get("X-Async-Upload") == "true"
        or request.args.get("async", "").lower() in ("true", "1")
    )
    is_sync_req = (
        request.values.get("sync", "").lower() in ("true", "1")
        or request.headers.get("X-Sync-Upload") == "true"
        or request.args.get("sync", "").lower() in ("true", "1")
    )
    is_sync = is_sync_req or (app.config.get("TESTING", False) and not is_async_req)

    task_id = f"task_{uuid.uuid4().hex[:12]}"
    with TASKS_LOCK:
        UPLOAD_TASKS[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress_pct": 0,
            "stage": "uploading",
            "message": "Upload received. Initializing processing...",
            "created_at": time.time(),
            "updated_at": time.time(),
            "files_count": len(saved_file_infos),
            "result": None,
            "error": None,
        }

    if is_sync:
        # Run synchronously for test clients
        _run_async_ingestion_task(
            task_id,
            saved_file_infos,
            replace_mode=replace_mode,
            initial_failed_files=failed_files,
        )
        task_data = UPLOAD_TASKS.get(task_id, {})
        if task_data.get("status") == "completed":
            res = task_data.get("result", {})
            return jsonify(res), 200
        else:
            return jsonify({
                "status": "error",
                "message": task_data.get("message", "Processing failed."),
                "failed_files": task_data.get("result", {}).get("failed_files", []),
            }), 422

    # Launch asynchronous background ingestion thread
    thread = threading.Thread(
        target=_run_async_ingestion_task,
        args=(task_id, saved_file_infos, replace_mode, failed_files),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "processing",
        "task_id": task_id,
        "message": f"Upload accepted ({len(saved_file_infos)} file(s)). Processing started.",
        "failed_files": failed_files,
    }), 202


@app.route("/api/upload-status/<task_id>", methods=["GET"])
def get_upload_status(task_id: str):
    """Returns real-time progress for an asynchronous document upload/indexing task."""
    with TASKS_LOCK:
        task = UPLOAD_TASKS.get(task_id)

    if not task:
        return jsonify({"status": "error", "message": "Task not found."}), 404

    return jsonify({
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress_pct": task.get("progress_pct", 0),
        "stage": task.get("stage", "idle"),
        "message": task.get("message", ""),
        "current_file": task.get("current_file", ""),
        "current_page": task.get("current_page", 0),
        "total_pages": task.get("total_pages", 0),
        "result": task.get("result"),
        "error": task.get("error"),
    })


@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles question submission with conversation history, hybrid search, reranking, and citation generation."""
    data = request.get_json(silent=True) or request.form

    query = data.get("question", "").strip() or data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Question cannot be empty."}), 400

    top_k = int(data.get("top_k", 3))
    score_threshold = float(data.get("score_threshold", 0.25))
    conversation_history = data.get("conversation_history", [])

    start_time = time.time()
    try:
        pipeline = get_pipeline()
        configured_index_dir = os.path.abspath(app.config.get("INDEX_DIR", "./data/faiss_index"))
        if pipeline.index_dir != configured_index_dir:
            pipeline.reload_vector_store(index_dir=configured_index_dir)

        result = pipeline.answer_question(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            conversation_history=conversation_history,
        )
        latency_ms = round((time.time() - start_time) * 1000, 1)
        result["latency_ms"] = latency_ms

        return jsonify(result)

    except Exception as e:
        print(f"[ASK ERROR] {e}")
        return (
            jsonify({
                "status": "error",
                "message": f"An error occurred while answering the question: {str(e)}",
            }),
            500,
        )


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """Returns metadata and statistics for all currently indexed documents."""
    try:
        chunks_file = app.config.get("CHUNKS_PATH", "./data/processed_chunks.json")
        if not os.path.exists(chunks_file):
            return jsonify({
                "status": "success",
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_model": "all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "llm_provider": "local",
                "documents": [],
            })

        chunks = load_processed_chunks(chunks_file)
        sources_map: Dict[str, Dict[str, Any]] = {}

        for c in chunks:
            src = c.source
            if src not in sources_map:
                ext = src.rsplit(".", 1)[-1].lower() if "." in src else ""
                file_type = ext.upper() if ext in {"pdf", "docx", "pptx", "txt", "md"} else "UNKNOWN"
                unit_label = "slides" if ext == "pptx" else "pages"
                sources_map[src] = {
                    "filename": src,
                    "pages": set(),
                    "chunk_count": 0,
                    "total_tokens": 0,
                    "images": set(),
                    "has_tables": False,
                    "has_ocr": False,
                    "file_type": file_type,
                    "unit_label": unit_label,
                }
            sources_map[src]["pages"].add(c.page)
            sources_map[src]["chunk_count"] += 1
            sources_map[src]["total_tokens"] += c.token_count
            for img in getattr(c, "images", []):
                if isinstance(img, dict) and "url" in img:
                    sources_map[src]["images"].add(img["url"])
            if getattr(c, "has_tables", False):
                sources_map[src]["has_tables"] = True
            if getattr(c, "is_ocr", False):
                sources_map[src]["has_ocr"] = True

        docs_list = [
            {
                "filename": v["filename"],
                "file_type": v["file_type"],
                "unit_label": v["unit_label"],
                "page_count": len(v["pages"]),
                "chunk_count": v["chunk_count"],
                "total_tokens": v["total_tokens"],
                "image_count": len(v["images"]),
                "has_tables": v["has_tables"],
                "has_ocr": v["has_ocr"],
            }
            for v in sorted(sources_map.values(), key=lambda x: x["filename"])
        ]

        pipeline = get_pipeline()

        return jsonify({
            "status": "success",
            "total_documents": len(docs_list),
            "total_chunks": len(chunks),
            "embedding_model": pipeline.embedding_engine.model_name,
            "reranker_model": getattr(pipeline.reranker, "model_name", "ms-marco-MiniLM-L-6-v2"),
            "llm_provider": pipeline.llm_provider,
            "documents": docs_list,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("\n========================================================")
    print("      LAUNCHING QASUBJECTBOT FLASK WEB APPLICATION      ")
    print("      Access web UI at: http://127.0.0.1:5000           ")
    print("========================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
