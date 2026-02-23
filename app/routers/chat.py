import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.rag import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

STORAGE_DIR = Path("storage")

MAX_QUESTION_LENGTH = 512


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question", mode="before")
    @classmethod
    def sanitize(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        sanitized = v.strip()
        if not sanitized:
            raise ValueError("question must not be blank")
        return sanitized[:MAX_QUESTION_LENGTH]


class SourceLink(BaseModel):
    url: str
    section_title: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceLink]
    confidence: str          # "low" | "medium" | "high"


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Ask the AKAR chatbot")
async def chat(request: ChatRequest):
    """
    Retrieve relevant chunks from the AKAR website knowledge base and
    generate an answer using an LLM. Always includes source URLs.
    """
    index_file = STORAGE_DIR / "faiss.index"
    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialised. Call POST /api/ingest first.",
        )

    logger.info("Chat question: %r", request.question)

    try:
        result = answer_question(request.question, str(STORAGE_DIR))
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}") from exc

    logger.info(
        "Answer confidence=%s  sources=%d",
        result["confidence"],
        len(result["sources"]),
    )
    return ChatResponse(**result)
