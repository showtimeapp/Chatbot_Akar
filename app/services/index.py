"""
index.py
────────
Chunks sections, embeds them with OpenAI, stores in a local FAISS index,
and persists both the index and metadata to disk.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

from app.services.pdf_parser import Section

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "600"))    # smaller = more chunks = better recall
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))
DOC_ID           = "akar_website_pdf_v1"
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE    = "metadata.pkl"

# ── Module-level singletons ───────────────────────────────────────────────────
_openai_client:  OpenAI | None      = None
_faiss_index:    faiss.Index | None = None
_metadata_cache: list[dict] | None  = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
        logger.info("OpenAI client initialised (cached for process lifetime)")
    return _openai_client


def get_cached_index(storage_dir: str) -> tuple[faiss.Index, list[dict]]:
    global _faiss_index, _metadata_cache
    if _faiss_index is None or _metadata_cache is None:
        _faiss_index, _metadata_cache = _load_index_from_disk(storage_dir)
        logger.info("FAISS index loaded into memory (%d vectors)", _faiss_index.ntotal)
    return _faiss_index, _metadata_cache


def invalidate_index_cache() -> None:
    global _faiss_index, _metadata_cache
    _faiss_index    = None
    _metadata_cache = None
    logger.info("FAISS index cache invalidated — will reload on next query")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start    = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Try to break at newline first (better semantic boundaries)
            snap = text.rfind("\n", start + overlap, end)
            if snap <= start:
                # Fall back to whitespace
                snap = text.rfind(" ", start + overlap, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors  = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ── Build / persist index ─────────────────────────────────────────────────────

def build_index(sections: list[Section], storage_dir: str) -> int:
    client = get_openai_client()

    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    all_chunks:   list[str]            = []
    all_metadata: list[dict[str, Any]] = []

    for section in sections:
        chunks = _chunk_text(section.full_text)
        logger.info("Section '%s' → %d chunks", section.section_title, len(chunks))
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "url":           section.url,
                "section_title": section.section_title,
                "chunk_index":   idx,
                "doc_id":        DOC_ID,
                "text":          chunk,
            })

    total = len(all_chunks)
    logger.info("Total chunks to embed: %d", total)

    if total == 0:
        raise ValueError("No chunks generated — check PDF content.")

    all_vectors: list[np.ndarray] = []
    BATCH_SIZE = 100
    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        logger.info("Embedding batch %d–%d of %d …", batch_start + 1, min(batch_start + BATCH_SIZE, total), total)
        all_vectors.append(_embed_texts(batch, client))

    vectors = np.vstack(all_vectors).astype(np.float32)
    faiss.normalize_L2(vectors)

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(storage_path / FAISS_INDEX_FILE))
    with open(storage_path / METADATA_FILE, "wb") as f:
        pickle.dump(all_metadata, f)

    invalidate_index_cache()

    logger.info("FAISS index saved — dim=%d, vectors=%d", dim, index.ntotal)
    return total


# ── Load index from disk ──────────────────────────────────────────────────────

def _load_index_from_disk(storage_dir: str) -> tuple[faiss.Index, list[dict]]:
    storage_path = Path(storage_dir)
    index = faiss.read_index(str(storage_path / FAISS_INDEX_FILE))
    with open(storage_path / METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ── Similarity search ─────────────────────────────────────────────────────────

def similarity_search(query: str, storage_dir: str, top_k: int = 6) -> list[tuple[dict, float]]:
    client          = get_openai_client()
    index, metadata = get_cached_index(storage_dir)

    q_vec = _embed_texts([query], client)
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)

    return [
        (metadata[idx], float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]