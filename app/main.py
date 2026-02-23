import logging
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Must run before any OpenAI client is initialised

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import chat, ingest

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STORAGE_DIR = Path("storage")

# ── Rate limiting (in-memory, per IP) ─────────────────────────────────────────
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW   = 60
_rate_store: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(ip: str) -> None:
    now          = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = [t for t in _rate_store[ip] if t > window_start]
    if len(calls) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s.",
        )
    calls.append(now)
    _rate_store[ip] = calls


# ── Lifespan: warm-up on startup ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AKAR RAG backend starting — warming up …")
    # Pre-load OpenAI client + FAISS index into memory so first request is fast
    from app.services.rag import warmup
    warmup(str(STORAGE_DIR))
    logger.info("Warm-up done. Ready to serve requests.")
    yield
    logger.info("AKAR RAG backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AKAR RAG Chatbot API",
    description="Production-grade RAG backend for the AKAR Strategic Consultants website chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    try:
        _check_rate_limit(client_ip)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)

app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(chat.router,   prefix="/api", tags=["Chat"])

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "akar-rag-backend"}