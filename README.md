# QASubjectBot: RAG Question-Answering Agent with Verified Source Citations

**QASubjectBot** is a Retrieval-Augmented Generation (RAG) system built with **Python, Flask, FAISS, BM25, and a cross-encoder reranker**. It answers questions over uploaded documents with inline source citations, document-grounded responses, and refusal fallbacks when the uploaded corpus does not contain enough evidence.

The system features a responsive Flask web application with asynchronous multi-format uploads, dynamic FAISS/BM25 re-indexing, citation inspectors, extracted diagrams and tables, and AJAX-based question answering.

---

## 📖 Subject Domain & Multi-Format Ingestion Support

The current sample corpus is **Biopsychology**, covering brain imaging, neurons and glia, neural communication, and related biological processes. The application is domain-agnostic: users can replace the corpus by uploading their own supported documents.

### Actual Indexed Corpus Documents (`documents/`)
The active workspace corpus currently contains **3 documents** and **57 indexed chunks**:

1. `UNIT_1_BIOPSYCHOLOGY.pptx`: Biopsychology topics including brain-imaging techniques and MRI.
2. `UNIT_2_BIOPSYCHOLOGY.pdf`: Neurons, glial cells, and the blood-brain barrier.
3. `UNIT_3_BIOPSYCHOLOGY.pdf`: Neural communication and related biopsychology topics.

### Supported Document Formats
QASubjectBot ingests and indexes four document formats:
1. **Portable Document Format (`.pdf`)**: Extracted page by page with PyMuPDF and a `pypdf` fallback, with OCR support for scanned pages. Cited as `[Source: filename.pdf, page X]`.
2. **Microsoft Word (`.docx`)**: Extracted with `python-docx`, preserving headings, lists, tables, and embedded images. Cited as `[Source: filename.docx, page X]`.
3. **PowerPoint Presentations (`.pptx`)**: Extracted slide by slide with `python-pptx`, including titles, text, tables, and embedded images. Cited as `[Source: presentation.pptx, slide X]`.
4. **Plain text and Markdown (`.txt`, `.md`)**: Indexed as text documents and cited by page-style chunk units.

*Users can upload one document or a mixed batch of up to 15 `.pdf`, `.docx`, `.pptx`, or `.txt` files through the web UI.*

### Grounded heading and visual retrieval

- Dense FAISS search, BM25 keyword search, and cross-encoder reranking work together for retrieval.
- Before returning an insufficient-context response, the system searches the complete indexed corpus. Exact document terms can recover content even when the user's wording differs from the source.
- When a relevant heading is found, the answer context expands from that heading through the content before the next heading.
- Answers use only uploaded-document evidence and state when requested information is absent.
- Embedded figures and diagrams are extracted to `static/extracted_images/` and can be attached from nearby pages or slides.

---

## 🏗️ Technical Architecture & Pipeline

```
[User Interface (HTML/CSS/JS via Flask)]
      │
      ├── POST /upload ─────────────► [Multi-Format Extraction: PDF, DOCX, PPTX, TXT]
      │                                       │
      │                                       ▼
      │                               [Token Chunking & Metadata Tagging]
      │                                       │
      │                                       ▼
      │                               [FAISS Index Rebuild & Persistence]
      │                                       │
      │                                       ▼
      │                               [Vector Store In-Memory Reload]
      │
      └── POST /ask ────────────────► [Query Embedding: all-MiniLM-L6-v2]
                                              │
                                              ▼
                                      [FAISS Vector Search (IndexFlatIP)]
                                              │
                                              ├── Initial hybrid retrieval and reranking
                                              │         │
                                              │         ▼
                                              │   [Heading-aware section expansion]
                                              │         │
                                              │         ▼
                                              │   [Grounded context and citations]
                                              │         │
                                              │         ▼
                                              │   [Answer + citations + relevant diagrams]
                                              │
                                              └── No semantic or lexical evidence after full-corpus search
                                                        │
                                                        ▼
                                                  [Graceful Refusal: "I don't have enough information..."]
```

---

## 🌐 Flask Web API Endpoints

| Endpoint | Method | Description | Request / Parameters | Response |
| :--- | :---: | :--- | :--- | :--- |
| `/` | `GET` | Main Web Application UI | None | HTML Web Page |
| `/upload` | `POST` | Upload up to 15 files and dynamically re-index | `multipart/form-data` with `files` and optional `mode=replace|append` | JSON `{status, message, uploaded_files, failed_files, chunks_added, total_chunks, total_documents}` |
| `/ask` | `POST` | Query the grounded RAG agent | JSON `{question, top_k, score_threshold, conversation_history}` | JSON `{query, answer, citations, images, has_images, retrieval_confidence, status, latency_ms}` |
| `/api/documents` | `GET` | List Indexed Corpus Files, Formats & Stats | None | JSON `{status, total_documents, total_chunks, documents: [...]}` |

---

## 📊 Benchmark Evaluation Results

The pipeline was evaluated against a domain-specific evaluation benchmark covering easy factual, comparative/computational, and near-topic/out-of-scope negative control queries.

*See full details in the [Evaluation Report](eval_report.md).*

---

## 🚀 How to Run the Project Locally

### 1. Prerequisites & Installation
Clone or navigate to the project workspace and install dependencies:
```bash
pip install -r requirements.txt
```

*(Optional)* Configure API keys in `.env` if using OpenAI or Claude:
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY
```
*(Note: QASubjectBot runs 100% locally out of the box with the local embedding model and deterministic generator even without an external API key).*

### 2. Launch the Flask Web Application (Single Command)
```bash
python S:/Project/QASubjectBot/app.py
```
The application resolves `documents/`, `data/`, and the FAISS index relative to the project files, so it can be launched from another working directory.
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 4. Run Evaluation Benchmark
```bash
python evaluate.py
```

---

## 📁 Repository Structure

```
QASubjectBot/
├── app.py                      # Flask web server with /upload, /ask, /api/documents routes
├── templates/
│   └── index.html              # Custom semantic HTML5 web frontend
├── static/
│   ├── style.css               # Modern CSS styling (cards, badges, animations)
│   └── script.js               # Client-side AJAX script for upload & question answering
├── documents/                  # Uploaded PDF, DOCX, PPTX, TXT, and Markdown documents
├── data/
│   ├── processed_chunks.json   # Ingested chunks with citations, tables, and image metadata
│   └── faiss_index/            # Persistent FAISS index binary and metadata mapping
├── scripts/
│   └── generate_corpus.py      # Multi-page corpus generator
├── tests/
│   ├── test_flask_app.py       # Flask routes and upload/ask test suite
│   ├── test_ingestion.py       # Ingestion & cleaning unit tests
│   ├── test_rag_pipeline.py    # Vector search, citation & fallback tests
│   └── test_multiformat_ingestion.py # Multi-format DOCX/PPTX ingestion & mixed upload test suite
├── ingestion.py                # Multi-format document loader, cleaner, and token chunker
├── embed_index.py              # Embedding generation & FAISS index builder
├── rag_pipeline.py             # Hybrid retrieval, section expansion, grounded answers, and citations
├── evaluate.py                 # Evaluation benchmark runner
├── eval_report.md              # Detailed evaluation report with metrics
├── requirements.txt            # Python dependencies (Flask, LangChain, FAISS, PyPDF, python-docx, python-pptx, etc.)
├── .env.example                # Environment variables template
└── README.md                   # Project documentation
```
