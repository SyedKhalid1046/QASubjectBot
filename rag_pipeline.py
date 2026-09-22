"""RAG Pipeline Module for QASubjectBot.

Implements:
1. Hybrid chunk retrieval (Dense FAISS Vector Search + Sparse BM25 Keyword Search).
2. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) for precision ranking of top-10 candidate chunks down to top-3.
3. Multi-turn conversation memory and contextual follow-up query rewriting.
4. Similarity thresholding fallback for out-of-scope queries.
5. Citation-enforced prompt construction with [Source: filename, page X] inline tags.
6. Flexible LLM support (OpenAI, Claude, and local deterministic fallback).
7. Unified function: answer_question(query, conversation_history) -> response dictionary.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from embed_index import EmbeddingEngine, VectorStoreIndex
from ingestion import DocumentChunk

# Load environment variables from .env
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "faiss_index")


class CrossEncoderReranker:
    """Reranks retrieved candidate chunks using a Transformer Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._disabled = False

        try:
            from sentence_transformers import CrossEncoder
            # Set HF warning suppression
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self.model = CrossEncoder(model_name)
            print(f"[RERANKER] Initialized Cross-Encoder ({model_name})")
        except Exception as e:
            print(f"[RERANKER WARNING] Could not load CrossEncoder ({e}). Using similarity scores.")
            self._disabled = True

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[DocumentChunk, float]],
        top_k: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Scores (query, chunk_text) pairs and reranks candidates descending by cross-encoder score."""
        if not candidate_chunks:
            return []

        if self._disabled or self.model is None:
            return candidate_chunks[:top_k]

        try:
            pairs = [[query, chunk.text] for chunk, _ in candidate_chunks]
            raw_scores = self.model.predict(pairs)

            # Sigmoid normalization: 1 / (1 + exp(-x))
            normalized_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

            scored_candidates = [
                (chunk, round(norm_s, 4))
                for (chunk, _), norm_s in zip(candidate_chunks, normalized_scores)
            ]

            # Sort descending by cross-encoder score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[:top_k]

        except Exception as e:
            print(f"[RERANKER ERROR] Reranking failed: {e}. Falling back to input ranking.")
            return candidate_chunks[:top_k]


class ConversationMemory:
    """Manages multi-turn conversation context and rewrites follow-up questions into standalone queries."""

    PRONOUNS_AND_REFERENCES = {
        "it", "its", "they", "them", "their", "this", "that", "these", "those",
        "he", "she", "him", "her", "the formula", "the equation", "the technique",
        "the method", "the concept", "the process", "the approach", "advantages",
        "disadvantages", "limitations", "examples", "applications", "why is that",
    }

    @classmethod
    def is_followup_query(cls, query: str) -> bool:
        """Determines if a query looks like a context-dependent follow-up question."""
        q_lower = query.lower().strip()
        words = set(re.findall(r"\b\w+\b", q_lower))

        # Check for ambiguous pronouns or short referential phrases
        if len(words) <= 5 and any(p in words for p in ["it", "this", "that", "why", "how", "what"]):
            return True
        if any(ref in q_lower for ref in ["what about", "how about", "explain more", "and its", "its main", "their"]):
            return True
        return False

    @classmethod
    def rewrite_query(
        cls,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        llm_client: Optional[Any] = None,
        llm_provider: str = "local",
    ) -> str:
        """Rewrites a contextual follow-up query into a standalone search query."""
        if not conversation_history:
            return query.strip()

        # Extract last user and assistant turn
        last_user_turn = ""
        last_asst_turn = ""
        for msg in reversed(conversation_history):
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()
            if role == "user" and not last_user_turn:
                last_user_turn = content
            elif role in ("assistant", "bot") and not last_asst_turn:
                last_asst_turn = content
            if last_user_turn and last_asst_turn:
                break

        if not last_user_turn:
            return query.strip()

        # If external LLM is available, use fast contextual reformulation
        if llm_provider in ["openai", "anthropic"] and llm_client is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                rewrite_prompt = (
                    "Given the conversation history and a follow-up question, rewrite the follow-up question into a "
                    "standalone, specific search query that incorporates necessary context and entities from earlier turns. "
                    "Return ONLY the rewritten query without any commentary or quotation marks."
                )
                conv_snippet = f"Previous User Query: {last_user_turn}\nPrevious Answer Summary: {last_asst_turn[:250]}\nFollow-up Question: {query}"
                messages = [
                    SystemMessage(content=rewrite_prompt),
                    HumanMessage(content=conv_snippet),
                ]
                resp = llm_client.invoke(messages)
                rewritten = resp.content.strip().strip('"').strip("'")
                if rewritten and len(rewritten) > 3:
                    return rewritten
            except Exception as e:
                print(f"[MEMORY REWRITE ERROR] LLM query rewriting failed: {e}")

        # Deterministic Keyword-Augmentation Fallback
        if cls.is_followup_query(query):
            # Extract key nouns/terms from previous user turn
            prev_words = re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", last_user_turn)
            prev_stopwords = {
                "what", "when", "where", "which", "explain", "describe", "about",
                "does", "have", "been", "with", "from", "that", "this", "these",
            }
            salient_terms = [w for w in prev_words if w.lower() not in prev_stopwords]
            if salient_terms:
                augmented = f"{query} {' '.join(salient_terms[:3])}"
                return augmented.strip()

        return query.strip()


class ExtractiveFallbackLLM:
    """Local deterministic generator used when no external API key is configured.
    Synthesizes complete, grounded answers strictly from retrieved context passages with verified citations,
    preserving Markdown tables and calculation formulas.
    """

    def generate(self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        if not context_chunks:
            return "I don't have enough information on that based on the provided documents."

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
            "from", "with", "have", "been", "does", "explain", "describe", "about", "tell", "show", "give",
        }
        meaningful_query_words = query_words - stopwords or query_words

        sections = []
        seen_texts = set()
        complete_section_context = any(
            chunk.metadata.get("section_expanded") for chunk, _ in context_chunks
        )

        for chunk, score in context_chunks:
            text = chunk.text.strip()
            if not text:
                continue

            # Split by double newline or logical paragraph boundaries
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            chunk_paras = []
            for p in paragraphs:
                if len(p) < 20:
                    continue
                p_norm = re.sub(r"\s+", " ", p).lower()
                if p_norm in seen_texts:
                    continue

                # Preserve Markdown tables and calculations
                is_table_or_calc = ("|" in p and "---" in p) or bool(re.search(r"(=|\+|\-|\*|/|%|₹|\$|\bformula\b)", p))

                p_words = set(re.findall(r"\b\w{3,}\b", p.lower()))
                overlap = len(meaningful_query_words.intersection(p_words))
                if chunk.metadata.get("section_expanded") or overlap > 0 or is_table_or_calc or len(chunk_paras) == 0:
                    seen_texts.add(p_norm)
                    priority = overlap + (5 if is_table_or_calc else 0)
                    chunk_paras.append((priority, p))

            unit = getattr(chunk, "unit_label", "Page").lower()
            if chunk_paras:
                chunk_paras.sort(key=lambda x: x[0], reverse=True)
                selected_paras = [item[1] for item in chunk_paras] if chunk.metadata.get("section_expanded") else [item[1] for item in chunk_paras[:2]]
                combined_text = "\n\n".join(selected_paras)
                citation = f"[Source: {chunk.source}, {unit} {chunk.page}]"
                sections.append(f"{combined_text} {citation}")
            else:
                t_norm = re.sub(r"\s+", " ", text[:150]).lower()
                if t_norm not in seen_texts:
                    seen_texts.add(t_norm)
                    citation = f"[Source: {chunk.source}, {unit} {chunk.page}]"
                    sections.append(f"{text} {citation}")

            if not complete_section_context and len(sections) >= 2:
                break

        if not sections:
            top_chunk, _ = context_chunks[0]
            unit = getattr(top_chunk, "unit_label", "Page").lower()
            citation = f"[Source: {top_chunk.source}, {unit} {top_chunk.page}]"
            return f"{top_chunk.text.strip()} {citation}"

        return "\n\n".join(sections)


class RAGPipeline:
    """Production RAG Pipeline with Hybrid Search, Cross-Encoder Reranking, Conversation Memory,
    Citation Enforcement, Table Formatting, and Calculation Support.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise, comprehensive, and factual AI Question-Answering Assistant for the QASubjectBot system.\n"
        "Your goal is to provide a complete, thorough, and detailed answer to the user's question using ONLY the retrieved context chunks provided below. When a heading is present, use the complete content from that heading until the next heading.\n\n"
        "STRICT CITATION AND ANSWERING RULES:\n"
        "1. Comprehensive Depth: Give a full and complete explanation. Include definitions, key principles, step-by-step mechanisms, and details present in the text.\n"
        "2. Tables & Structured Data: If the retrieved context contains tables or structured data relevant to the question, present them in clean GitHub Markdown Table format (| Col 1 | Col 2 |).\n"
        "3. Calculations & Mathematical Formulas: If the question involves calculations, formulas, or arithmetic, write the formulas clearly (e.g. using LaTeX/KaTeX math syntax like $$...$$ or bold formula callouts) and show the step-by-step calculation.\n"
        "4. Visuals & Diagrams: If relevant diagrams or figures exist in the context, refer to them and explain what they illustrate.\n"
        "5. Grounding: Answer strictly and solely based on the facts provided in the context chunks. Do not add outside speculation, assumptions, or facts from general knowledge. If a requested detail is absent, say that it is not available in the uploaded document.\n"
        "6. Verified Citations: For every factual statement, paragraph, or key section in your answer, append an inline citation referencing the exact source document and page/slide number, formatted strictly as: [Source: <filename>, page <page>] or [Source: <filename>, slide <slide>]\n"
        "   Example: The blood-brain barrier minimizes the risk of irreparable brain damage [Source: UNIT_2_BIOPSYCHOLOGY.pdf, page 11] or [Source: presentation.pptx, slide 4].\n"
        "7. Fallback on Missing Information: If the provided context does NOT contain sufficient facts to answer the question, do NOT guess. Respond strictly with:\n"
        "\"I don't have enough information on that based on the provided documents.\""
    )

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: Optional[str] = None,
        score_threshold: float = 0.25,
    ):
        self.score_threshold = score_threshold
        self.index_dir = index_dir

        # 1. Initialize Embedding Engine and load FAISS + BM25 Index
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = VectorStoreIndex.load(
            index_dir=index_dir,
            embedding_engine=self.embedding_engine,
        )

        # 2. Initialize Cross-Encoder Reranker
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # 3. Initialize Conversation Memory
        self.memory = ConversationMemory()

        # 4. Initialize LLM Provider (OpenAI, Anthropic, or Local Fallback)
        self.llm_provider, self.llm_client = self._init_llm(llm_model)

    def reload_vector_store(self, index_dir: Optional[str] = None):
        """Reloads the FAISS index, BM25, and chunk metadata from disk into the running pipeline."""
        if index_dir:
            self.index_dir = os.path.abspath(index_dir)
        print(f"[RAG] Reloading FAISS + BM25 vector index from: {self.index_dir}")
        self.vector_store = VectorStoreIndex.load(
            index_dir=self.index_dir,
            embedding_engine=self.embedding_engine,
        )
        print(f"[RAG] Reload complete. Total active vectors: {self.vector_store.index.ntotal}")

    def _init_llm(self, llm_model: Optional[str]) -> Tuple[str, Any]:
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if openai_key and not openai_key.startswith("your_"):
            try:
                from langchain_openai import ChatOpenAI
                model_name = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                llm = ChatOpenAI(model=model_name, temperature=0.0, api_key=openai_key)
                print(f"[RAG] Initialized OpenAI LLM ({model_name})")
                return ("openai", llm)
            except Exception as e:
                print(f"[RAG WARNING] Failed to initialize ChatOpenAI: {e}")

        if anthropic_key and not anthropic_key.startswith("your_"):
            try:
                from langchain_anthropic import ChatAnthropic
                model_name = llm_model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                llm = ChatAnthropic(model=model_name, temperature=0.0, api_key=anthropic_key)
                print(f"[RAG] Initialized Anthropic LLM ({model_name})")
                return ("anthropic", llm)
            except Exception as e:
                print(f"[RAG WARNING] Failed to initialize ChatAnthropic: {e}")

        print("[RAG] No external LLM API key detected. Using local deterministic citation generator.")
        return ("local", ExtractiveFallbackLLM())

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_reranker: bool = True,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Hybrid retrieval: Rewrites query with memory, fetches top 10 candidates via Hybrid BM25+FAISS,
        and reranks down to top_k with Cross-Encoder.
        Returns List of (chunk, score).
        """
        threshold = self.score_threshold if score_threshold is None else score_threshold

        # 1. Conversation Memory Query Rewriting
        effective_query = self.memory.rewrite_query(
            query=query,
            conversation_history=conversation_history,
            llm_client=self.llm_client,
            llm_provider=self.llm_provider,
        )
        self.last_effective_query = effective_query

        # 2. Hybrid Candidate Retrieval (top 10 candidates)
        candidate_count = max(10, top_k * 3)
        candidate_chunks = self.vector_store.hybrid_search(
            query=effective_query,
            top_k=candidate_count,
            alpha=0.7,
        )

        # 3. Cross-Encoder Reranking
        if use_reranker and candidate_chunks:
            reranked_chunks = self.reranker.rerank(
                query=effective_query,
                candidate_chunks=candidate_chunks,
                top_k=top_k,
            )
        else:
            reranked_chunks = candidate_chunks[:top_k]

        # 4. Threshold filtering
        filtered_results = [(chunk, score) for chunk, score in reranked_chunks if score >= threshold]

        # A refusal must only happen after checking the complete uploaded corpus.
        # The normal path stays small for latency; this recovery path prevents a
        # relevant chunk ranked outside the first candidate window from being missed.
        if not filtered_results and len(candidate_chunks) < len(self.vector_store.chunks):
            full_candidates = self.vector_store.hybrid_search(
                query=effective_query,
                top_k=len(self.vector_store.chunks),
                alpha=0.7,
            )
            full_reranked = self.reranker.rerank(
                query=effective_query,
                candidate_chunks=full_candidates,
                top_k=min(len(full_candidates), max(top_k * 4, 12)),
            )
            filtered_results = [
                (chunk, score) for chunk, score in full_reranked if score >= threshold
            ][:top_k]

            # Exact terms from the uploaded document are stronger evidence than
            # a low cross-encoder score caused by different question wording.
            if not filtered_results:
                stopwords = {
                    "what", "when", "where", "which", "who", "how", "why", "does",
                    "are", "is", "the", "a", "an", "and", "or", "of", "to", "in",
                    "on", "for", "from", "about", "explain", "describe", "tell", "give",
                }
                query_terms = {
                    term for term in re.findall(r"[a-z0-9]+", effective_query.lower())
                    if term not in stopwords and len(term) >= 3
                }
                lexical_matches = []
                for chunk, score in full_candidates:
                    chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
                    overlap = len(query_terms & chunk_terms)
                    if overlap >= 2 or any(len(term) >= 8 and term in chunk_terms for term in query_terms):
                        lexical_matches.append((overlap, score, chunk))
                lexical_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
                filtered_results = [
                    (chunk, max(score, threshold))
                    for _, score, chunk in lexical_matches[:top_k]
                ]

        return filtered_results

    def format_context_prompt(
        self,
        query: str,
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
    ) -> str:
        """Constructs structured context prompt with chunk metadata."""
        context_blocks = []
        for rank, (chunk, score) in enumerate(chunks_with_scores, 1):
            unit = getattr(chunk, "unit_label", "Page")
            section_heading = chunk.metadata.get("section_heading")
            heading_label = f" | Heading: {section_heading}" if section_heading else ""
            header = f"[Chunk {rank} | Source: {chunk.source} | {unit}: {chunk.page} | ChunkID: {chunk.chunk_id}{heading_label}]"
            context_blocks.append(f"{header}\n{chunk.text}")

        formatted_context = "\n\n" + ("=" * 40) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 40)

        prompt = (
            f"{self.DEFAULT_SYSTEM_PROMPT}\n\n"
            f"--- BEGIN RETRIEVED CONTEXT ---\n{formatted_context}\n--- END RETRIEVED CONTEXT ---\n\n"
            f"User Question: {query}\n\n"
            f"Answer with inline citations [Source: filename, page/slide X]:"
        )
        return prompt

    @staticmethod
    def _heading_candidates(text: str) -> List[str]:
        """Returns heading-like lines while excluding bullets, prose, and tables."""
        candidates = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or len(line) > 120 or "|" in line:
                continue
            if line.startswith(("-", "*", "•", "●", "○")):
                continue
            words = re.findall(r"[A-Za-z][A-Za-z0-9'–—-]*", line)
            if not 1 <= len(words) <= 12:
                continue
            if re.match(r"^(?:#{1,6}\s+|[A-Za-z]\)|\d+[.)]\s+)", line):
                candidates.append(line)
                continue
            if line.isupper() and len(words) >= 2:
                candidates.append(line)
                continue
            if not re.search(r"[.!?;:]$", line) and sum(word[0].isupper() for word in words) >= max(2, len(words) // 2):
                candidates.append(line)
        return candidates

    def expand_to_complete_sections(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """Expands hits to the document section between its heading and next heading."""
        if not retrieved_chunks:
            return []

        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        chunks_by_source: Dict[str, List[DocumentChunk]] = {}
        for chunk in self.vector_store.chunks:
            chunks_by_source.setdefault(chunk.source, []).append(chunk)
        for source_chunks in chunks_by_source.values():
            source_chunks.sort(key=lambda item: item.chunk_index)

        expanded: Dict[str, Tuple[DocumentChunk, float]] = {}
        for hit, hit_score in retrieved_chunks:
            source_chunks = chunks_by_source.get(hit.source, [])
            if not source_chunks:
                continue
            hit_index = next((i for i, item in enumerate(source_chunks) if item.chunk_id == hit.chunk_id), None)
            if hit_index is None:
                expanded[hit.chunk_id] = (hit, hit_score)
                continue

            heading_index = hit_index
            best_heading_score = -1
            selected_heading = ""
            for index in range(hit_index, -1, -1):
                headings = self._heading_candidates(source_chunks[index].text)
                heading_score = max(
                    (len(query_terms & set(re.findall(r"[a-z0-9]+", heading.lower()))) for heading in headings),
                    default=0,
                )
                if headings and (heading_score > best_heading_score or index == hit_index):
                    heading_index = index
                    best_heading_score = heading_score
                    selected_heading = max(
                        headings,
                        key=lambda heading: len(query_terms & set(re.findall(r"[a-z0-9]+", heading.lower()))),
                    )
                if heading_score > 0:
                    break

            next_heading_index = len(source_chunks)
            for index in range(heading_index + 1, len(source_chunks)):
                if self._heading_candidates(source_chunks[index].text):
                    next_heading_index = index
                    break

            for section_chunk in source_chunks[heading_index:next_heading_index]:
                if section_chunk.chunk_id not in expanded:
                    section_metadata = dict(section_chunk.metadata)
                    section_metadata["section_expanded"] = True
                    section_metadata["section_heading"] = selected_heading
                    expanded[section_chunk.chunk_id] = (
                        DocumentChunk(text=section_chunk.text, metadata=section_metadata),
                        hit_score,
                    )

        if not expanded:
            return retrieved_chunks
        return list(expanded.values())

    def extract_structured_citations(
        self,
        answer: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> List[Dict[str, Any]]:
        """Extracts and validates cited sources from the generated answer."""
        citations = []
        seen_keys = set()

        pattern = r"\[Source:\s*([^,\]]+),\s*(?:page|slide|section)\s*(\d+)\]"
        matches = re.findall(pattern, answer, re.IGNORECASE)

        for src, page_str in matches:
            src_clean = src.strip()
            page_int = int(page_str.strip())
            key = (src_clean, page_int)
            if key not in seen_keys:
                seen_keys.add(key)
                matching_chunk = next(
                    (c for c, _ in retrieved_chunks if c.source == src_clean and c.page == page_int),
                    None,
                )
                score = next((s for c, s in retrieved_chunks if c.source == src_clean and c.page == page_int), 0.0)
                unit_label = (
                    matching_chunk.unit_label
                    if matching_chunk
                    else ("Slide" if src_clean.lower().endswith(".pptx") else "Page")
                )
                citations.append({
                    "source": src_clean,
                    "page": page_int,
                    "unit": unit_label,
                    "chunk_id": matching_chunk.chunk_id if matching_chunk else f"{src_clean}:p{page_int}",
                    "similarity_score": round(score, 4),
                    "snippet": matching_chunk.text[:200] + "..." if matching_chunk else "",
                })

        # Fallback if no citations formatted but chunks retrieved
        if not citations and retrieved_chunks:
            for c, s in retrieved_chunks[:2]:
                key = (c.source, c.page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    citations.append({
                        "source": c.source,
                        "page": c.page,
                        "unit": c.unit_label,
                        "chunk_id": c.chunk_id,
                        "similarity_score": round(s, 4),
                        "snippet": c.text[:200] + "...",
                    })

        return citations

    def collect_relevant_images(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        page_window: int = 2,
    ) -> List[Dict[str, Any]]:
        """Collects figures attached to nearby chunks when text and figure split across pages."""
        relevant_images: List[Dict[str, Any]] = []
        seen_img_urls = set()
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        retrieved_ranges = [
            (
                chunk.source,
                chunk.page,
                chunk.metadata.get("end_page", chunk.page),
                score,
            )
            for chunk, score in retrieved_chunks
        ]

        candidates = []
        for chunk in self.vector_store.chunks:
            chunk_images = getattr(chunk, "images", [])
            if not chunk_images:
                continue
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
            term_overlap = len(query_terms & chunk_terms)

            for source, start_page, end_page, retrieved_score in retrieved_ranges:
                if chunk.source != source:
                    continue
                chunk_end_page = chunk.metadata.get("end_page", chunk.page)
                distance = 0 if chunk.page <= end_page and chunk_end_page >= start_page else min(
                    abs(chunk.page - end_page), abs(start_page - chunk_end_page)
                )
                if distance <= page_window:
                    candidates.append((term_overlap, -distance, retrieved_score, chunk))
                    break

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        for term_overlap, _, retrieved_score, chunk in candidates:
            for image in chunk.images:
                image_url = image.get("url")
                if not image_url or image_url in seen_img_urls:
                    continue
                seen_img_urls.add(image_url)
                relevant_images.append({
                    **image,
                    "relevance_score": round(retrieved_score + min(term_overlap, 10) * 0.001, 4),
                })

        return relevant_images

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Main entry point: executes memory query rewriting, hybrid retrieval, cross-encoder reranking,
        threshold fallback, and citation-enforced answer generation.
        """
        threshold = self.score_threshold if score_threshold is None else score_threshold
        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=threshold,
            conversation_history=conversation_history,
        )
        effective_query = getattr(self, "last_effective_query", query)

        # Insufficient context check
        if not retrieved:
            raw_top = self.vector_store.search(effective_query, top_k=1)
            confidence = round(raw_top[0][1], 4) if raw_top else 0.0
            return {
                "query": query,
                "effective_query": effective_query,
                "answer": "I don't have enough information on that based on the provided documents.",
                "citations": [],
                "images": [],
                "has_images": False,
                "has_tables": False,
                "has_calculations": False,
                "retrieval_confidence": confidence,
                "status": "insufficient_context",
                "llm_provider": self.llm_provider,
                "is_hybrid": True,
                "reranker_used": True,
            }

        top_confidence = round(retrieved[0][1], 4)

        section_context = self.expand_to_complete_sections(effective_query, retrieved)

        # Figures may be stored on a nearby chunk when extraction splits a section across pages.
        relevant_images = self.collect_relevant_images(effective_query, section_context)

        # Generate answer using configured LLM
        if self.llm_provider in ["openai", "anthropic"]:
            prompt = self.format_context_prompt(effective_query, section_context)
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=self.DEFAULT_SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{prompt}\n\nQuestion: {query}"),
                ]
                response = self.llm_client.invoke(messages)
                answer_text = response.content.strip()
            except Exception as e:
                print(f"[RAG ERROR] LLM invocation failed: {e}. Falling back to extractive generator.")
                fallback_gen = ExtractiveFallbackLLM()
                answer_text = fallback_gen.generate(effective_query, section_context)
        else:
            answer_text = self.llm_client.generate(effective_query, section_context)

        # Extract and format citations
        structured_citations = self.extract_structured_citations(answer_text, section_context)

        # Detect tables and calculations
        has_tables = ("|" in answer_text and "---" in answer_text) or any(getattr(c, "has_tables", False) for c, _ in section_context)
        has_calculations = bool(
            re.search(
                r"(=|\+|\-|\*|/|%|₹|\$|\bformula\b|\bcalculate\b|\bcalc\b|\bratio\b|\bcost\b|\bdepreciation\b)",
                answer_text,
                re.IGNORECASE,
            )
        ) or any(getattr(c, "has_calculations", False) for c, _ in section_context)

        return {
            "query": query,
            "effective_query": effective_query,
            "answer": answer_text,
            "citations": structured_citations,
            "images": relevant_images,
            "has_images": len(relevant_images) > 0,
            "has_tables": has_tables,
            "has_calculations": has_calculations,
            "retrieval_confidence": top_confidence,
            "status": "success",
            "llm_provider": self.llm_provider,
            "is_hybrid": True,
            "reranker_used": True,
        }


# Global singleton instance
_default_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def answer_question(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.25,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper to answer a query using the default RAG pipeline."""
    pipeline = get_pipeline()
    return pipeline.answer_question(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        conversation_history=conversation_history,
    )


if __name__ == "__main__":
    print("\n========================================================")
    print("      QASUBJECTBOT HYBRID RAG & RERANKER PIPELINE       ")
    print("========================================================\n")

    pipeline = get_pipeline()

    sample_questions = [
        "What is Biopsychology and what are the fields that contribute to behavioral neuroscience?",
        "What are the major structural imaging techniques in biopsychology like MRI and CT?",
        "What are the ingredients in a chocolate cake?",
    ]

    for q in sample_questions:
        print(f"\n[QUERY]: {q}")
        res = pipeline.answer_question(q)
        print(f"[STATUS]: {res['status']} | [CONFIDENCE]: {res['retrieval_confidence']:.4f}")
        print(f"[ANSWER]:\n{res['answer']}")
        print(f"[IMAGES]: {len(res.get('images', []))}")
        print(f"[CITATIONS]: {len(res.get('citations', []))}")
        print("-" * 60)
