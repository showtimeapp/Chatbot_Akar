"""
rag.py
──────
Retrieve relevant chunks → build prompt → call OpenAI chat → return structured answer.

Performance notes:
- OpenAI client is a shared singleton (no re-init per request).
- FAISS index is cached in memory (no disk read per request).
- gpt-4o-mini is used for lowest latency among capable models.
"""

import logging
import os
from typing import Any

from openai import OpenAI

from app.services.index import get_openai_client, similarity_search

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CHAT_MODEL       = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K            = int(os.getenv("TOP_K", "4"))          # reduced from 6 → faster embed + smaller prompt
MAX_SOURCES      = 3
SNIPPET_LENGTH   = 200

HIGH_THRESHOLD   = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD",   "0.75"))
MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.55"))

NOT_FOUND_PHRASE = "I couldn't find this information in the website knowledge base."

SYSTEM_PROMPT_TEMPLATE = """You are the official AI assistant for AKAR Strategic Consultants.
Answer ONLY using the context chunks below. Be concise and professional.
If the answer is not in the context, respond with exactly: "{not_found}"
Never fabricate URLs — only use URLs from the context.

CONTEXT:
{context}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(retrieved: list[tuple[dict, float]]) -> str:
    parts = []
    for i, (meta, score) in enumerate(retrieved, start=1):
        parts.append(f"[{i}] {meta['section_title']} | {meta['url']}\n{meta['text']}")
    return "\n\n---\n\n".join(parts)


def _confidence_level(retrieved: list[tuple[dict, float]], answer: str) -> str:
    if NOT_FOUND_PHRASE in answer or not retrieved:
        return "low"
    top_score = retrieved[0][1]
    if top_score >= HIGH_THRESHOLD and len(retrieved) >= 2:
        return "high"
    if top_score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _build_sources(retrieved: list[tuple[dict, float]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources = []
    for meta, _ in retrieved:
        url = meta["url"]
        if url in seen:
            continue
        seen.add(url)
        snippet = meta["text"][:SNIPPET_LENGTH].strip()
        if len(meta["text"]) > SNIPPET_LENGTH:
            snippet += " …"
        sources.append({"url": url, "section_title": meta["section_title"], "snippet": snippet})
        if len(sources) >= MAX_SOURCES:
            break
    return sources


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warmup(storage_dir: str) -> None:
    """
    Called once at server startup to:
    1. Initialise the OpenAI client (avoids cold-start on first request).
    2. Load FAISS index into memory (avoids disk read on first request).
    """
    try:
        from app.services.index import get_cached_index
        get_openai_client()
        get_cached_index(storage_dir)
        logger.info("Warm-up complete — OpenAI client and FAISS index ready")
    except Exception as exc:
        logger.warning("Warm-up skipped (index not yet built): %s", exc)


# ── Main entry ────────────────────────────────────────────────────────────────

def answer_question(question: str, storage_dir: str) -> dict[str, Any]:
    client = get_openai_client()

    # 1. Retrieve (client + index already cached)
    retrieved = similarity_search(question, storage_dir, top_k=TOP_K)
    logger.info(
        "Retrieved %d chunks. Top score=%.4f",
        len(retrieved),
        retrieved[0][1] if retrieved else 0.0,
    )

    # 2. Build prompt (compact — fewer tokens = faster completion)
    context = _build_context(retrieved)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        not_found=NOT_FOUND_PHRASE,
    )

    # 3. Call LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": question},
        ],
        temperature=0.1,
        max_tokens=400,        # tighter cap = faster response
    )
    answer = response.choices[0].message.content.strip()
    logger.info("LLM answer (first 120 chars): %r", answer[:120])

    return {
        "answer":     answer,
        "sources":    _build_sources(retrieved),
        "confidence": _confidence_level(retrieved, answer),
    }