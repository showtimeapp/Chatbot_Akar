import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.pdf_parser import parse_pdf_sections
from app.services.index import build_index

logger = logging.getLogger(__name__)

router = APIRouter()

PDF_PATH    = Path("data/Akar website Consolidated.pdf")
STORAGE_DIR = Path("storage")


class IngestResponse(BaseModel):
    status: str
    sections: int
    chunks: int


@router.post("/ingest", response_model=IngestResponse, summary="Ingest PDF into vector store")
async def ingest():
    """
    Parse the AKAR website PDF, chunk it, embed every chunk and persist
    the FAISS index + metadata to ./storage.
    """
    if not PDF_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found at '{PDF_PATH}'. "
                   "Place the file at: data/Akar website Consolidated.pdf",
        )

    logger.info("Ingest started — reading %s", PDF_PATH)

    try:
        sections = parse_pdf_sections(str(PDF_PATH))
    except Exception as exc:
        logger.exception("PDF parsing failed")
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {exc}") from exc

    if not sections:
        raise HTTPException(status_code=422, detail="No sections detected in PDF.")

    logger.info("Parsed %d sections", len(sections))

    try:
        chunk_count = build_index(sections, str(STORAGE_DIR))
    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(status_code=500, detail=f"Indexing error: {exc}") from exc

    logger.info("Ingest complete — %d chunks indexed", chunk_count)
    return IngestResponse(status="success", sections=len(sections), chunks=chunk_count)
