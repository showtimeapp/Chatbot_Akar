"""
rag.py
──────
Retrieve relevant chunks → build prompt → call OpenAI chat → return structured answer.
"""

import logging
import os
from typing import Any

from openai import OpenAI

from app.services.index import get_openai_client, similarity_search

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CHAT_MODEL       = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K            = int(os.getenv("TOP_K", "6"))
MAX_SOURCES      = 3
SNIPPET_LENGTH   = 200

HIGH_THRESHOLD   = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD",   "0.70"))
MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.45"))

NOT_FOUND_PHRASE = "I couldn't find this information in the website knowledge base."

SYSTEM_PROMPT_TEMPLATE = """You are the official AI assistant for AKAR Strategic Consultants website.

Your job is to answer visitor questions using ONLY the context provided below.

STRICT RULES:
1. Read ALL context chunks carefully before answering.
2. Even if the question is short or phrased differently, look for related information in the context.
   - "clients" → look for "Client Engagement", company names, project work
   - "founders" or "team" → look for "Directors", "Managing Director"
   - "services" → look for solution names, capabilities
   - "work" or "projects" → look for client names, engagement descriptions
3. Synthesise information from multiple chunks if needed.
4. Be concise, friendly, and professional.
5. ONLY say "{not_found}" if after reading all chunks there is truly no relevant information.
6. Never fabricate URLs — only use URLs that appear in the context.
7. Do not mention "context", "chunks", or "documents" in your answer.

CONTEXT:
{context}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(retrieved: list[tuple[dict, float]]) -> str:
    parts = []
    for i, (meta, score) in enumerate(retrieved, start=1):
        parts.append(
            f"[{i}] Section: {meta['section_title']} | URL: {meta['url']}\n{meta['text']}"
        )
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
        sources.append({
            "url": url,
            "section_title": meta["section_title"],
            "snippet": snippet,
        })
        if len(sources) >= MAX_SOURCES:
            break
    return sources


# ── Query expansion ───────────────────────────────────────────────────────────

def _expand_query(question: str) -> str:
    """
    Expand short/vague queries with synonyms so embedding matches better.
    This runs locally — no extra API call.
    """
    q = question.lower().strip()
    expansions = {
        "client":    "clients projects work engagement AMNEX NMDPL",
        "clients":   "clients projects work engagement AMNEX NMDPL",
        "customer":  "clients projects work engagement",
        "work":      "client engagement projects AMNEX NMDPL our work",
        "project":   "client engagement projects our work",
        "founder":   "founders directors managing director team",
        "founders":  "founders directors managing director team",
        "team":      "founders directors managing director team",
        "about":     "about AKAR vision values founders directors",
        "service":   "services solutions field research AI urban transformation",
        "services":  "services solutions field research AI urban transformation",
        "solution":  "services solutions capabilities",
        "solutions": "services solutions capabilities",
        "contact":   "contact us get in touch",
        "price":     "pricing cost consultation contact",
        "cost":      "pricing cost consultation contact",
    }
    for keyword, expansion in expansions.items():
        if keyword in q:
            return f"{question} {expansion}"
    return question


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warmup(storage_dir: str) -> None:
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

    # Expand short queries before embedding
    expanded_query = _expand_query(question)
    if expanded_query != question:
        logger.info("Query expanded: %r → %r", question, expanded_query)

    # Retrieve
    retrieved = similarity_search(expanded_query, storage_dir, top_k=TOP_K)
    logger.info(
        "Retrieved %d chunks. Top score=%.4f",
        len(retrieved),
        retrieved[0][1] if retrieved else 0.0,
    )

    # Build prompt
    context = _build_context(retrieved)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        not_found=NOT_FOUND_PHRASE,
    )

    # Call LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": question},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content.strip()
    logger.info("LLM answer (first 120 chars): %r", answer[:120])

    return {
        "answer":     answer,
        "sources":    _build_sources(retrieved),
        "confidence": _confidence_level(retrieved, answer),
    }