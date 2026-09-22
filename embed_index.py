"""Embedding, BM25 Keyword, and FAISS Vector Indexing Module for QASubjectBot.

Handles:
1. Dense vector embedding generation using Sentence-Transformers (default: all-MiniLM-L6-v2) or OpenAI embeddings.
2. Sparse keyword indexing via BM25 (rank-bm25) for high-accuracy exact keyword matching.
3. Hybrid Search combining Dense Vector Search + Sparse BM25 via Reciprocal Rank Fusion (RRF).
4. Building, incrementally updating, and persisting FAISS + BM25 indexes to disk.
5. Metadata mapping and persistence.
"""

import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from sentence_transformers import SentenceTransformer

from ingestion import DocumentChunk, ingest_documents, load_processed_chunks

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "processed_chunks.json")
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "faiss_index")


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words for BM25 keyword matching."""
    if not text:
        return []
    # Match words, numbers, and hyphenated terms
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.\$₹%]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class EmbeddingEngine:
    """Manages dense vector embedding generation with automatic L2 normalization for Cosine Similarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if self.use_openai and self.openai_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            self.dimension = 1536 if "3-small" in model_name or "ada" in model_name else 3072
            print(f"[EMBED] Initialized OpenAI Embeddings ({model_name})")
        else:
            # Disable symlink warnings on Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self.model = SentenceTransformer(model_name)
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"[EMBED] Initialized SentenceTransformer ({model_name}, dim={self.dimension})")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of texts and returns unit L2-normalized float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.use_openai and self.openai_api_key:
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name if "text-embedding" in self.model_name else "text-embedding-3-small",
                )
                for item in response.data:
                    embeddings_list.append(item.embedding)
            vectors = np.array(embeddings_list, dtype=np.float32)
        else:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)

        # L2-normalize vectors so Inner Product (IP) corresponds to Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms
        return normalized_vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning shape (1, dimension) L2-normalized array."""
        return self.embed_texts([query])


class VectorStoreIndex:
    """Manages the FAISS index and BM25 keyword index lifecycle: creation, serialization, deserialization, and hybrid search."""

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[Any] = None
        self.tokenized_corpus: List[List[str]] = []

    def _rebuild_bm25(self):
        """Builds or rebuilds BM25 index over currently loaded chunks."""
        if BM25Okapi is None or not self.chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return

        self.tokenized_corpus = [tokenize_for_bm25(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] Built BM25 index over {len(self.tokenized_corpus)} chunk documents.")

    def build_from_chunks(self, chunks: List[DocumentChunk]) -> "VectorStoreIndex":
        """Builds an exact Cosine Similarity FAISS index (IndexFlatIP) and BM25 index from DocumentChunk instances."""
        if not chunks:
            raise ValueError("Cannot build vector index from empty chunk list.")

        print(f"[INDEX] Embedding {len(chunks)} chunks using dimension {self.embedding_engine.dimension}...")
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)

        # Initialize FAISS IndexFlatIP (Inner Product over unit vectors = Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.embedding_engine.dimension)
        self.index.add(embeddings)
        self.chunks = list(chunks)

        # Build BM25 Index
        self._rebuild_bm25()

        print(f"[INDEX] Successfully built FAISS Index ({self.index.ntotal} vectors) + BM25 Index.")
        return self

    def save(self, index_dir: str = DEFAULT_INDEX_DIR):
        """Saves FAISS index binary, associated chunk metadata, and BM25 state to disk."""
        if self.index is None or not self.chunks:
            raise ValueError("No index or chunks to save.")

        index_dir = os.path.abspath(index_dir)
        os.makedirs(index_dir, exist_ok=True)

        index_file = os.path.join(index_dir, "index.faiss")
        metadata_file = os.path.join(index_dir, "metadata.json")

        # Write FAISS binary index
        faiss.write_index(self.index, index_file)

        # Write chunk metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2, ensure_ascii=False)

        print(f"[INDEX] Saved FAISS index and metadata to: {index_dir}")

    @classmethod
    def load(
        cls,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> "VectorStoreIndex":
        """Loads FAISS index and metadata from disk and initializes BM25 index."""
        index_dir = os.path.abspath(index_dir)
        index_file = os.path.join(index_dir, "index.faiss")
        metadata_file = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"FAISS index files not found in '{index_dir}'. Run embed_index.py first.")

        if embedding_engine is None:
            embedding_engine = EmbeddingEngine()

        instance = cls(embedding_engine)
        instance.index = faiss.read_index(index_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
        instance.chunks = [DocumentChunk.from_dict(item) for item in raw_metadata]

        # Reconstruct BM25 index
        instance._rebuild_bm25()

        print(f"[INDEX] Loaded FAISS index ({instance.index.ntotal} vectors) + BM25 index from: {index_dir}")
        return instance

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        """Incrementally adds new chunks to the in-memory FAISS and BM25 index without full rebuild."""
        if not new_chunks:
            return

        texts = [c.text for c in new_chunks]
        embeddings = self.embedding_engine.embed_texts(texts)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_engine.dimension)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

        # Rebuild BM25 with all chunks
        self._rebuild_bm25()
        print(f"[INDEX] Added {len(new_chunks)} new chunks. Total index vectors: {self.index.ntotal}")

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Dense semantic vector similarity search via FAISS."""
        if self.index is None or not self.chunks:
            raise ValueError("Index is not loaded.")

        top_k = min(top_k, len(self.chunks))
        query_vec = self.embedding_engine.embed_query(query)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                sim_score = float(score)
                results.append((self.chunks[idx], sim_score))

        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Sparse lexical keyword search via BM25."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        max_score = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0

        for idx in top_indices:
            raw_score = float(bm25_scores[idx])
            if raw_score > 0.0:
                normalized_score = min(1.0, raw_score / max_score)
                results.append((self.chunks[idx], normalized_score))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combines Dense Vector Search + BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
        
        RRF Score: alpha * (1 / (rrf_k + dense_rank)) + (1 - alpha) * (1 / (rrf_k + bm25_rank))
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index is not loaded.")

        candidate_pool_size = min(len(self.chunks), max(top_k * 2, 20))

        # 1. Fetch dense candidates
        dense_results = self.search_dense(query, top_k=candidate_pool_size)

        # 2. Fetch BM25 candidates
        bm25_results = self.search_bm25(query, top_k=candidate_pool_size)

        if not bm25_results:
            # Fallback directly to dense if BM25 has no keyword matches
            return dense_results[:top_k]

        # 3. Reciprocal Rank Fusion Scoring
        chunk_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}
        dense_scores_map: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results, 1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (alpha * (1.0 / (rrf_k + rank)))

        for rank, (chunk, score) in enumerate(bm25_results, 1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + ((1.0 - alpha) * (1.0 / (rrf_k + rank)))

        # Sort candidate pool by RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Return list of (chunk, effective_score)
        # Use dense cosine similarity as the primary calibrated score metric where present
        final_results = []
        for cid, rrf_val in sorted_candidates:
            chunk = chunk_map[cid]
            # Primary score is dense similarity if available, else estimated score
            calibrated_score = dense_scores_map.get(cid, max(0.30, min(0.95, rrf_val * 60.0)))
            final_results.append((chunk, calibrated_score))

        return final_results

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Standard search interface: executes hybrid search if BM25 is available, else dense search."""
        if self.bm25 is not None:
            return self.hybrid_search(query, top_k=top_k)
        return self.search_dense(query, top_k=top_k)


def build_and_save_index(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = "all-MiniLM-L6-v2",
) -> VectorStoreIndex:
    """End-to-end builder that loads chunks, generates embeddings, and saves the FAISS + BM25 index."""
    chunks_path = os.path.abspath(chunks_path)
    if not os.path.exists(chunks_path):
        print(f"[INDEX] Processed chunks not found at '{chunks_path}'. Running ingestion...")
        chunks = ingest_documents(output_path=chunks_path)
    else:
        chunks = load_processed_chunks(chunks_path)

    engine = EmbeddingEngine(model_name=model_name)
    vector_store = VectorStoreIndex(engine)
    vector_store.build_from_chunks(chunks)
    vector_store.save(index_dir)
    return vector_store


def reset_and_rebuild_index(
    new_chunks: List[DocumentChunk],
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = "all-MiniLM-L6-v2",
) -> VectorStoreIndex:
    """Completely resets/clears the index directory and builds a fresh FAISS and BM25 index
    from the provided chunks.
    """
    index_dir = os.path.abspath(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    # Clean out any old index files
    for fname in ["index.faiss", "metadata.json", "bm25.pkl"]:
        fpath = os.path.join(index_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"[INDEX WARNING] Could not remove '{fpath}': {e}")

    engine = EmbeddingEngine(model_name=model_name)
    store = VectorStoreIndex(engine)
    store.build_from_chunks(new_chunks)
    store.save(index_dir)
    return store


def add_chunks_to_index(
    new_chunks: List[DocumentChunk],
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = "all-MiniLM-L6-v2",
) -> VectorStoreIndex:
    """Loads existing index, adds new chunks incrementally, and saves updated index back to disk."""
    index_dir = os.path.abspath(index_dir)
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        # If no index exists, build fresh
        engine = EmbeddingEngine(model_name=model_name)
        store = VectorStoreIndex(engine)
        store.build_from_chunks(new_chunks)
        store.save(index_dir)
        return store

    engine = EmbeddingEngine(model_name=model_name)
    store = VectorStoreIndex.load(index_dir=index_dir, embedding_engine=engine)
    store.add_chunks(new_chunks)
    store.save(index_dir)
    return store


if __name__ == "__main__":
    print("\n========================================================")
    print("     BUILDING HYBRID FAISS + BM25 FOR QASUBJECTBOT      ")
    print("========================================================\n")
    store = build_and_save_index()

    test_query = "What is the scaling factor in Scaled Dot-Product Attention?"
    print(f"\n[TEST HYBRID SEARCH] Query: '{test_query}'")
    results = store.hybrid_search(test_query, top_k=3)
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {rank} (Score: {score:.4f}):")
        print(f"  Source : {chunk.source} (Page {chunk.page})")
        print(f"  ChunkID: {chunk.chunk_id}")
        print(f"  Snippet: {chunk.text[:180]}...")
