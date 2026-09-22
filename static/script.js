/**
 * QASubjectBot Frontend Client-Side Script
 * Handles asynchronous chunked uploads with live progress bar polling,
 * Hybrid Search (BM25 + FAISS + Reranker), multi-turn conversation memory,
 * KaTeX formula rendering, Markdown tables, and verified citations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Upload & Sidebar
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const fileSelectedInfo = document.getElementById("file-selected-info");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadSpinner = document.getElementById("upload-spinner");
    const uploadMessage = document.getElementById("upload-message");
    const selectedFilesCount = document.getElementById("selected-files-count");
    const selectedTotalSize = document.getElementById("selected-total-size");
    const selectedFilesList = document.getElementById("selected-files-list");
    const uploadBtnText = document.getElementById("upload-btn-text");

    // Progress Tracker Elements
    const uploadProgressContainer = document.getElementById("upload-progress-container");
    const progressStageBadge = document.getElementById("progress-stage-badge");
    const progressPercent = document.getElementById("progress-percent");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressMessage = document.getElementById("progress-message");
    const stepUpload = document.getElementById("step-upload");
    const stepExtract = document.getElementById("step-extract");
    const stepEmbed = document.getElementById("step-embed");
    const stepDone = document.getElementById("step-done");

    // Sliders & Corpus Stats
    const topKSlider = document.getElementById("top-k-slider");
    const topKVal = document.getElementById("top-k-val");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    const libraryList = document.getElementById("library-list");
    const libraryCountBadge = document.getElementById("library-count-badge");
    const statDocsCount = document.getElementById("stat-docs-count");
    const statChunksCount = document.getElementById("stat-chunks-count");
    const statModelName = document.getElementById("stat-model-name");

    // QA Main Workspace
    const qaForm = document.getElementById("qa-form");
    const questionInput = document.getElementById("question-input");
    const askBtn = document.getElementById("ask-btn");
    const askSpinner = document.getElementById("ask-spinner");
    const qaLoading = document.getElementById("qa-loading");

    // Conversation History Container
    const conversationArea = document.getElementById("conversation-area");
    const conversationList = document.getElementById("conversation-list");
    const conversationCountBadge = document.getElementById("conversation-count-badge");
    const clearConversationBtn = document.getElementById("clear-conversation-btn");

    let totalTurns = 0;
    let isSubmitting = false;
    let isUploading = false;
    let progressPollInterval = null;

    // Multi-turn conversation memory store
    const conversationHistory = [];

    // ------------------------------------------------------------------------
    // 1. Initialize and Fetch Corpus Metadata
    // ------------------------------------------------------------------------
    const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".txt", ".md"];

    function getFileTypeInfo(filename) {
        const ext = (filename.split(".").pop() || "").toLowerCase();
        if (ext === "pdf") {
            return { type: "PDF", icon: "📄", badgeClass: "badge-pdf" };
        } else if (ext === "docx" || ext === "doc") {
            return { type: "DOCX", icon: "📝", badgeClass: "badge-docx" };
        } else if (ext === "pptx" || ext === "ppt") {
            return { type: "PPTX", icon: "📊", badgeClass: "badge-pptx" };
        } else if (ext === "txt" || ext === "md") {
            return { type: "TXT", icon: "📄", badgeClass: "badge-txt" };
        }
        return { type: ext.toUpperCase() || "FILE", icon: "📄", badgeClass: "badge-file" };
    }

    // async function fetchCorpusMetadata() {
    //     try {
    //         const res = await fetch("/api/documents");
    //         const data = await res.json();
    //         if (data.status === "success") {
    //             statDocsCount.textContent = `${data.total_documents} Documents`;
    //             statChunksCount.textContent = `${data.total_chunks} Chunks`;
    //             statModelName.textContent = data.embedding_model;
    //             libraryCountBadge.textContent = `${data.total_documents} Files`;

    //             renderLibraryList(data.documents);
    //         }
    //     } catch (err) {
    //         console.error("Failed to load corpus metadata:", err);
    //         libraryList.innerHTML = `<div class="alert alert-error">Failed to load corpus list.</div>`;
    //     }
    // }
    async function fetchCorpusMetadata() {
        try {
            const res = await fetch("/api/documents");

            const data = await res.json();

            if (!res.ok || data.status !== "success") {
                throw new Error(
                    data.message || `Server returned HTTP ${res.status}`
                );
            }

            if (statDocsCount) {
                statDocsCount.textContent = `${data.total_documents} Documents`;
            }

            if (statChunksCount) {
                statChunksCount.textContent = `${data.total_chunks} Chunks`;
            }

            if (statModelName) {
                statModelName.textContent = data.embedding_model || "Embedding Model";
            }

            if (libraryCountBadge) {
                libraryCountBadge.textContent = `${data.total_documents} Files`;
            }

            renderLibraryList(data.documents || []);

        } catch (err) {
            console.error("Failed to load corpus metadata:", err);

            libraryList.innerHTML = `
            <div class="alert alert-error">
                Failed to load corpus list: ${escapeHtml(err.message)}
            </div>
        `;
        }
    }

    function renderLibraryList(documents) {
        if (!documents || documents.length === 0) {
            libraryList.innerHTML = `<div class="loading-placeholder">No documents indexed yet.</div>`;
            return;
        }

        libraryList.innerHTML = documents
            .map((doc) => {
                const info = getFileTypeInfo(doc.filename);
                const unit = doc.unit_label || (info.type === "PPTX" ? "slides" : "pages");
                const imgBadge = doc.image_count > 0 ? `<span class="badge-mini badge-media" title="${doc.image_count} extracted figures/diagrams">🖼️ ${doc.image_count}</span>` : "";
                const tableBadge = doc.has_tables ? `<span class="badge-mini badge-table" title="Contains structured Markdown tables">📊 Table</span>` : "";
                const ocrBadge = doc.has_ocr ? `<span class="badge-mini badge-ocr" title="Scanned page extracted via OCR">🔍 OCR</span>` : "";
                return `
            <div class="library-item" title="${escapeHtml(doc.filename)}">
                <div class="library-item-left">
                    <span class="file-type-badge ${info.badgeClass}">${info.type}</span>
                    <span class="library-item-name">${info.icon} ${escapeHtml(doc.filename)}</span>
                </div>
                <div class="library-item-right">
                    ${imgBadge}
                    ${tableBadge}
                    ${ocrBadge}
                    <span class="library-item-badge">${doc.chunk_count} chunks • ${doc.page_count} ${unit}</span>
                </div>
            </div>
        `;
            })
            .join("");
    }

    // ------------------------------------------------------------------------
    // 2. Settings Slider Event Listeners
    // ------------------------------------------------------------------------
    topKSlider.addEventListener("input", (e) => {
        topKVal.textContent = e.target.value;
    });

    // Threshold is controlled by the backend
    const threshold = 0.20;

    // ------------------------------------------------------------------------
    // 3. File Upload & Drag-and-Drop Handling (up to 200MB, PDF/DOCX/PPTX/TXT)
    // ------------------------------------------------------------------------
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFilesSelection(fileInput.files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFilesSelection(e.target.files);
        }
    });

    function handleFilesSelection(files) {
        const fileArray = Array.from(files);
        const unsupportedFiles = fileArray.filter(
            (f) => !ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
        );

        if (unsupportedFiles.length > 0) {
            showUploadMessage(
                `Unsupported format(s): ${unsupportedFiles.map(f => f.name).join(", ")}. Supported: PDF, Word (.docx), PowerPoint (.pptx), Text (.txt).`,
                "error"
            );
            fileInput.value = "";
            fileSelectedInfo.classList.add("hidden");
            return;
        }

        if (fileArray.length > 20) {
            showUploadMessage(
                `Please select up to 15 files at a time (you selected ${fileArray.length}).`,
                "error"
            );
            fileInput.value = "";
            fileSelectedInfo.classList.add("hidden");
            return;
        }

        const totalBytes = fileArray.reduce((acc, f) => acc + f.size, 0);
        selectedFilesCount.textContent = `${fileArray.length} File${fileArray.length > 1 ? "s" : ""} Selected`;
        selectedTotalSize.textContent = `Total: ${formatBytes(totalBytes)}`;
        uploadBtnText.textContent = `Index ${fileArray.length} Document${fileArray.length > 1 ? "s" : ""}`;

        // Render file list preview
        selectedFilesList.innerHTML = fileArray
            .map((f) => {
                const info = getFileTypeInfo(f.name);
                return `
            <div class="selected-file-item">
                <div class="selected-file-item-left">
                    <span class="file-type-badge ${info.badgeClass}">${info.type}</span>
                    <span class="selected-file-item-name">${info.icon} ${escapeHtml(f.name)}</span>
                </div>
                <span class="file-size">${formatBytes(f.size)}</span>
            </div>
        `;
            })
            .join("");

        fileSelectedInfo.classList.remove("hidden");
        uploadMessage.classList.add("hidden");
    }

    // ------------------------------------------------------------------------
    // 4. Asynchronous Upload & Real-Time Progress Polling
    // ------------------------------------------------------------------------
    function updateProgressUI(pct, stageName, message, stepIdx) {
        uploadProgressContainer.classList.remove("hidden");
        const safePct = Math.max(0, Math.min(100, Math.round(pct)));
        progressBarFill.style.width = `${safePct}%`;
        progressPercent.textContent = `${safePct}%`;
        progressStageBadge.textContent = stageName || "Processing";
        progressMessage.textContent = message || "Processing documents...";

        // Step highlights
        [stepUpload, stepExtract, stepEmbed, stepDone].forEach((stepEl, idx) => {
            if (!stepEl) return;
            stepEl.classList.remove("active", "completed");
            if (idx + 1 < stepIdx) {
                stepEl.classList.add("completed");
            } else if (idx + 1 === stepIdx) {
                stepEl.classList.add("active");
            }
        });
    }

    function pollUploadProgress(taskId, isReplaceMode) {
        if (progressPollInterval) clearInterval(progressPollInterval);

        progressPollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/upload-status/${taskId}`);
                if (!res.ok) return;

                const data = await res.json();
                const pct = data.progress_pct || 0;
                const stage = data.stage || "extracting";
                const msg = data.message || "Processing...";

                let stepIdx = 2;
                if (pct < 10) stepIdx = 1;
                else if (pct < 70) stepIdx = 2;
                else if (pct < 99) stepIdx = 3;
                else stepIdx = 4;

                let stageLabel = "PyMuPDF & OCR";
                if (stage === "uploading") stageLabel = "Uploading";
                else if (stage === "embedding") stageLabel = "FAISS & BM25";
                else if (stage === "reloading") stageLabel = "Reloading Vector Store";
                else if (stage === "completed") stageLabel = "Complete";
                else if (stage === "error") stageLabel = "Error";

                updateProgressUI(pct, stageLabel, msg, stepIdx);

                if (data.status === "completed") {
                    clearInterval(progressPollInterval);
                    progressPollInterval = null;
                    isUploading = false;
                    uploadBtn.disabled = false;
                    uploadSpinner.classList.add("hidden");

                    updateProgressUI(100, "Complete", "Indexing complete!", 4);
                    showUploadMessage(`✅ ${data.result ? data.result.message : "Document indexing completed successfully!"}`, "success");

                    fileInput.value = "";
                    fileSelectedInfo.classList.add("hidden");
                    uploadBtnText.textContent = "Save & Index";

                    if (isReplaceMode) {
                        clearConversation();
                    }
                    fetchCorpusMetadata();

                    // Fade out progress container after 4 seconds
                    setTimeout(() => {
                        uploadProgressContainer.classList.add("hidden");
                    }, 4500);
                } else if (data.status === "error") {
                    clearInterval(progressPollInterval);
                    progressPollInterval = null;
                    isUploading = false;
                    uploadBtn.disabled = false;
                    uploadSpinner.classList.add("hidden");

                    showUploadMessage(`❌ ${data.message || data.error || "Upload processing failed."}`, "error");
                }
            } catch (err) {
                console.error("Progress poll error:", err);
            }
        }, 500);
    }

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (isUploading) return;
        if (!fileInput.files || fileInput.files.length === 0) {
            showUploadMessage("Please select at least one document (.pdf, .docx, .pptx, .txt).", "error");
            return;
        }

        const modeRadios = document.getElementsByName("upload-mode");
        let mode = "replace";
        for (const r of modeRadios) {
            if (r.checked) mode = r.value;
        }
        const isReplaceMode = mode === "replace";

        const fileArray = Array.from(fileInput.files);
        const formData = new FormData();
        fileArray.forEach((file) => {
            formData.append("files", file);
        });
        formData.append("mode", mode);

        // UI Loading State
        isUploading = true;
        uploadBtn.disabled = true;
        uploadSpinner.classList.remove("hidden");
        updateProgressUI(5, "Uploading", `Uploading ${fileArray.length} file(s)...`, 1);
        showUploadMessage(`Uploading ${fileArray.length} file(s)... Starting background worker...`, "info");

        try {
            const res = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            if (res.status === 202 && data.task_id) {
                // Background async processing initiated
                updateProgressUI(10, "PyMuPDF & OCR", `Uploaded. Extracting ${fileArray.length} document(s)...`, 2);
                pollUploadProgress(data.task_id, isReplaceMode);
            } else if (res.ok && data.status === "success") {
                // Synchronous return
                updateProgressUI(100, "Complete", "Indexing complete!", 4);
                showUploadMessage(`✅ ${data.message}`, "success");
                fileInput.value = "";
                fileSelectedInfo.classList.add("hidden");
                uploadBtnText.textContent = "Save & Index";
                if (isReplaceMode) clearConversation();
                fetchCorpusMetadata();
                isUploading = false;
                uploadBtn.disabled = false;
                uploadSpinner.classList.add("hidden");
            } else {
                isUploading = false;
                uploadBtn.disabled = false;
                uploadSpinner.classList.add("hidden");
                showUploadMessage(`❌ ${data.message || "Upload failed."}`, "error");
            }
        } catch (err) {
            isUploading = false;
            uploadBtn.disabled = false;
            uploadSpinner.classList.add("hidden");
            showUploadMessage(`❌ Network error: ${err.message}`, "error");
        }
    });

    function showUploadMessage(msg, type) {
        uploadMessage.textContent = msg;
        uploadMessage.className = `alert ${type === "success" ? "alert-success" : type === "error" ? "alert-error" : "alert-info"}`;
        uploadMessage.classList.remove("hidden");
    }

    // ------------------------------------------------------------------------
    // 5. Question Answering Submission & Multi-Turn History
    // ------------------------------------------------------------------------
    qaForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = questionInput.value.trim();
        if (!query || isSubmitting) return;
        submitQuestion(query, { sourceInput: questionInput });
    });

    // Enter key triggers submit for main textarea, Shift+Enter adds newline
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            qaForm.requestSubmit();
        }
    });

    // Clear Conversation Button
    clearConversationBtn.addEventListener("click", () => {
        clearConversation();
    });

    function clearConversation() {
        conversationList.innerHTML = "";
        conversationHistory.length = 0; // Clear memory
        totalTurns = 0;
        updateConversationBadge();
        conversationArea.classList.add("hidden");
        questionInput.value = "";
    }

    function updateConversationBadge() {
        conversationCountBadge.textContent = `${totalTurns} ${totalTurns === 1 ? "Answer" : "Answers"}`;
    }

    /**
     * Submits a question from either top search box or follow-up box, passing conversation history.
     */
    async function submitQuestion(query, options = {}) {
        if (isSubmitting) return;
        isSubmitting = true;

        const topK = parseInt(topKSlider.value, 10);
        const threshold = 0.20;

        // UI Loading State
        conversationArea.classList.remove("hidden");
        qaLoading.classList.remove("hidden");
        disableAllAskButtons(true);

        if (options.submitBtn && options.spinner) {
            options.spinner.classList.remove("hidden");
        } else if (askSpinner) {
            askSpinner.classList.remove("hidden");
        }

        qaLoading.scrollIntoView({ behavior: "smooth", block: "nearest" });

        // Record user question in conversation memory
        conversationHistory.push({ role: "user", content: query });

        try {
            const res = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: query,
                    top_k: topK,
                    score_threshold: threshold,
                    conversation_history: conversationHistory.slice(-8), // Send last 8 turns
                }),
            });

            const data = await res.json();

            totalTurns += 1;
            updateConversationBadge();

            if (res.ok) {
                // Record assistant response
                conversationHistory.push({ role: "assistant", content: data.answer });
                renderNewAnswerCard(query, data, totalTurns);
            } else {
                renderErrorAnswerCard(query, data.message || "Failed to generate answer.", totalTurns);
            }

            if (options.sourceInput) {
                options.sourceInput.value = "";
            }
        } catch (err) {
            totalTurns += 1;
            updateConversationBadge();
            renderErrorAnswerCard(query, `Network error: ${err.message}`, totalTurns);
        } finally {
            qaLoading.classList.add("hidden");
            disableAllAskButtons(false);
            if (options.spinner) {
                options.spinner.classList.add("hidden");
            }
            if (askSpinner) {
                askSpinner.classList.add("hidden");
            }
            isSubmitting = false;
        }
    }

    function disableAllAskButtons(disabled) {
        if (askBtn) askBtn.disabled = disabled;
    }

    function getConfidenceRating(confidenceScore, isSuccess = true) {
        if (!isSuccess) {
            return { label: "Bad", className: "conf-bad" };
        }
        const score = parseFloat(confidenceScore || 0);
        if (score >= 0.40) {
            return { label: "Good", className: "conf-good" };
        } else if (score >= 0.20) {
            return { label: "Intermediate", className: "conf-intermediate" };
        } else {
            return { label: "Bad", className: "conf-bad" };
        }
    }

    /**
     * Renders a new dynamic Answer Card with citations, diagrams/visuals, tables, and follow-up box.
     */
    function renderNewAnswerCard(query, data, turnNumber) {
        const isSuccess = data.status === "success";
        const statusBadgeClass = isSuccess ? "status-success" : "status-fallback";
        const statusBadgeText = isSuccess
            ? "✅ Grounded Answer with Verified Source Citations"
            : "⚠️ Refusal: Insufficient In-Scope Context";

        const conf = getConfidenceRating(data.retrieval_confidence, isSuccess);
        const latency = `${data.latency_ms || 0} ms`;
        const highlightedAnswer = formatAnswerMarkdownAndCitations(data.answer);
        const citations = data.citations || [];
        const images = data.images || [];

        // Build Visual Evidence / Diagrams Gallery HTML
        let visualsHtml = "";
        if (images.length > 0) {
            visualsHtml = `
            <div class="visuals-section">
                <div class="visuals-header">
                    <h3 class="visuals-title">🖼️ Diagrams & Visual Evidence (${images.length})</h3>
                    <span class="visuals-hint">Click any image to view full size</span>
                </div>
                <div class="media-gallery">
                    ${images.map((img) => {
                const info = getFileTypeInfo(img.source);
                const unitLabel = img.unit_label || (info.type === "PPTX" ? "Slide" : "Page");
                return `
                        <div class="media-card" data-img-url="${escapeHtml(img.url)}" data-caption="${escapeHtml(img.caption || img.source)}" title="Click to view full diagram">
                            <div class="media-thumb-wrapper">
                                <img src="${escapeHtml(img.url)}" alt="${escapeHtml(img.caption || 'Diagram')}" class="media-thumb" loading="lazy" />
                                <div class="media-overlay"><span class="zoom-icon">🔍 View High-Res</span></div>
                            </div>
                            <div class="media-info">
                                <span class="media-caption">${escapeHtml(img.caption || img.filename)}</span>
                                <span class="media-source-tag">${info.icon} ${escapeHtml(img.source)} (${unitLabel} ${img.page})</span>
                            </div>
                        </div>`;
            }).join("")}
                </div>
            </div>`;
        }

        // Build Citations HTML
        let citationsHtml = "";
        if (citations.length > 0) {
            citationsHtml = citations
                .map((c) => {
                    const info = getFileTypeInfo(c.source);
                    const unitName = c.unit || (info.type === "PPTX" ? "Slide" : "Page");
                    return `
                <div class="citation-card">
                    <div class="citation-header">
                        <div class="citation-source">
                            <span class="file-type-badge ${info.badgeClass}">${info.type}</span>
                            <span>${info.icon} ${escapeHtml(c.source)}</span>
                        </div>
                        <div class="citation-meta-badges">
                            <span class="badge-page">${unitName} ${c.page}</span>
                        </div>
                    </div>
                    ${c.snippet ? `<div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>` : ""}
                </div>
            `;
                })
                .join("");
        } else {
            citationsHtml = `
                <p class="card-desc" style="margin-bottom:0;">
                    No citations attached because the question scored below the required confidence threshold or lacked factual context in the indexed corpus.
                </p>
            `;
        }

        // Create Card Element
        const card = document.createElement("article");
        card.className = "card answer-card conversation-turn-card";
        card.setAttribute("data-turn", turnNumber);

        const rewrittenBadge = data.effective_query && data.effective_query !== query
            ? `<div class="rewritten-query-banner">🧠 Context Expanded: <em>"${escapeHtml(data.effective_query)}"</em></div>`
            : "";

        card.innerHTML = `
            <!-- Question Banner -->
            <div class="question-banner">
                <div class="question-meta-row">
                    <span class="turn-pill">Turn #${turnNumber}</span>
                    <span class="question-timestamp">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                </div>
                <div class="question-query-row">
                    <span class="question-icon">❓</span>
                    <span class="question-query-text">${escapeHtml(query)}</span>
                </div>
                ${rewrittenBadge}
            </div>

            <!-- Status & Metrics Header -->
            <div class="answer-header">
                <div class="status-badge ${statusBadgeClass}">${statusBadgeText}</div>
                <div class="metrics-row">
                    <span class="badge-mini badge-hybrid" title="BM25 + FAISS Hybrid Retrieval">⚡ Hybrid RRF</span>
                    <span class="badge-mini badge-rerank" title="Cross-Encoder Precision Reranker">🎯 Cross-Encoder</span>
                    ${data.has_calculations ? '<span class="badge-mini badge-calc" title="Contains calculations / formulas">🧮 Math & Formulas</span>' : ''}
                    ${data.has_tables ? '<span class="badge-mini badge-table" title="Contains structured table data">📊 Data Table</span>' : ''}
                    <span class="metric-item">Confidence: <strong class="${conf.className}">${conf.label}</strong></span>
                    <span class="metric-item">Latency: <strong>${latency}</strong></span>
                </div>
            </div>

            <!-- Synthesized Answer Text with Tables and Calculations -->
            <div class="answer-body">
                <h3 class="answer-title">💡 Synthesized Answer</h3>
                <div class="answer-content markdown-body">${highlightedAnswer}</div>
            </div>

            <!-- Diagrams and Visual Evidence Section -->
            ${visualsHtml}

            <!-- Verified Citations Area -->
            <div class="citations-section">
                <h3 class="citations-title">📑 Verified Source Citations (${citations.length})</h3>
                <div class="citations-list">${citationsHtml}</div>
            </div>
        `;

        // Prepend to conversation list (newest on top)
        conversationList.prepend(card);

        // Render KaTeX math formulas if available
        if (typeof renderMathInElement !== "undefined") {
            try {
                const ansElem = card.querySelector(".answer-content");
                if (ansElem) {
                    renderMathInElement(ansElem, {
                        delimiters: [
                            { left: "$$", right: "$$", display: true },
                            { left: "$", right: "$", display: false },
                            { left: "\\[", right: "\\]", display: true },
                            { left: "\\(", right: "\\)", display: false }
                        ],
                        throwOnError: false
                    });
                }
            } catch (katexErr) {
                console.warn("KaTeX render notice:", katexErr);
            }
        }


        // Scroll to card
        card.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderErrorAnswerCard(query, errorMsg, turnNumber) {
        const card = document.createElement("article");
        card.className = "card answer-card conversation-turn-card";
        card.setAttribute("data-turn", turnNumber);

        card.innerHTML = `
            <div class="question-banner">
                <div class="question-meta-row">
                    <span class="turn-pill">Turn #${turnNumber}</span>
                    <span class="question-timestamp">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                </div>
                <div class="question-query-row">
                    <span class="question-icon">❓</span>
                    <span class="question-query-text">${escapeHtml(query)}</span>
                </div>
            </div>

            <div class="answer-header">
                <div class="status-badge status-fallback">❌ Error Processing Question</div>
            </div>

            <div class="answer-body">
                <h3 class="answer-title">💡 Synthesized Answer</h3>
                <div class="answer-content">
                    <div class="alert alert-error">${escapeHtml(errorMsg)}</div>
                </div>
            </div>

            <div class="followup-section">
                <div class="followup-header">
                    <span class="followup-title">💬 Ask a follow-up question / retry:</span>
                </div>
                <form class="followup-form">
                    <div class="input-wrapper">
                        <textarea class="followup-input" rows="2" placeholder="Ask a follow-up question or rephrase query..." required></textarea>
                        <button type="submit" class="btn btn-primary btn-ask followup-btn">
                            <span class="btn-text">Ask AI</span>
                            <span class="spinner hidden"></span>
                        </button>
                    </div>
                </form>
            </div>
        `;

        conversationList.prepend(card);

        const followupForm = card.querySelector(".followup-form");
        const followupInput = card.querySelector(".followup-input");
        const followupBtn = card.querySelector(".followup-btn");
        const followupSpinner = followupBtn.querySelector(".spinner");

        followupForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const followupQuery = followupInput.value.trim();
            if (!followupQuery || isSubmitting) return;
            submitQuestion(followupQuery, {
                sourceInput: followupInput,
                submitBtn: followupBtn,
                spinner: followupSpinner,
            });
        });

        followupInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                followupForm.requestSubmit();
            }
        });

        card.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function formatAnswerMarkdownAndCitations(rawText) {
        if (!rawText) return "";

        const citationMap = [];
        const textWithPlaceholders = rawText.replace(
            /\[Source:\s*([^,\]]+),\s*(page|slide|section)\s*(\d+)\]/gi,
            (match, source, unit, num) => {
                const id = `__CIT_${citationMap.length}__`;
                const info = getFileTypeInfo(source.trim());
                const shortUnit = unit.toLowerCase() === "slide" ? "Slide" : "Page";
                citationMap.push({
                    placeholder: id,
                    html: `<span class="inline-citation" title="${escapeHtml(source)} (${shortUnit} ${num})">${info.icon} ${escapeHtml(source)} (${shortUnit} ${num})</span>`
                });
                return id;
            }
        );

        let parsedHtml = "";
        if (typeof marked !== "undefined" && marked.parse) {
            try {
                parsedHtml = marked.parse(textWithPlaceholders, {
                    gfm: true,
                    breaks: true,
                });
            } catch (err) {
                parsedHtml = escapeHtml(textWithPlaceholders).replace(/\n\n/g, "<br><br>");
            }
        } else {
            parsedHtml = escapeHtml(textWithPlaceholders).replace(/\n\n/g, "<br><br>");
        }

        citationMap.forEach((item) => {
            parsedHtml = parsedHtml.split(item.placeholder).join(item.html);
        });

        return parsedHtml;
    }

    // ------------------------------------------------------------------------
    // 6. Image Lightbox Modal Handlers
    // ------------------------------------------------------------------------
    const lightbox = document.getElementById("image-lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    const lightboxCaption = document.getElementById("lightbox-caption-text");
    const lightboxDownloadBtn = document.getElementById("lightbox-download-btn");
    const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
    const lightboxOverlay = document.getElementById("lightbox-overlay");

    function openLightbox(url, caption) {
        if (!lightbox || !lightboxImg) return;
        lightboxImg.src = url;
        if (lightboxCaption) lightboxCaption.textContent = caption || "Document Diagram";
        if (lightboxDownloadBtn) {
            lightboxDownloadBtn.href = url;
            lightboxDownloadBtn.setAttribute("download", url.split("/").pop() || "diagram.png");
        }
        lightbox.classList.remove("hidden");
        lightbox.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
    }

    function closeLightbox() {
        if (!lightbox) return;
        lightbox.classList.add("hidden");
        lightbox.setAttribute("aria-hidden", "true");
        if (lightboxImg) lightboxImg.src = "";
        document.body.style.overflow = "";
    }

    if (lightboxCloseBtn) lightboxCloseBtn.addEventListener("click", closeLightbox);
    if (lightboxOverlay) lightboxOverlay.addEventListener("click", closeLightbox);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && lightbox && !lightbox.classList.contains("hidden")) {
            closeLightbox();
        }
    });

    conversationList.addEventListener("click", (e) => {
        const mediaCard = e.target.closest(".media-card");
        if (mediaCard) {
            const url = mediaCard.getAttribute("data-img-url");
            const caption = mediaCard.getAttribute("data-caption");
            if (url) openLightbox(url, caption);
        }
    });

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    }

    fetchCorpusMetadata();
});
