# AKAR RAG Chatbot Backend

Production-grade Retrieval-Augmented Generation (RAG) API for the AKAR Strategic Consultants website chatbot.

---

## Architecture

```
POST /api/ingest              POST /api/chat
       │                             │
       ▼                             ▼
  pdf_parser.py            similarity_search()
  (Section extraction)       (FAISS + OpenAI embed)
       │                             │
       ▼                             ▼
  index.py                    rag.py
  (Chunk → Embed → FAISS)  (Prompt + LLM → Answer)
       │                             │
       ▼                             ▼
  ./storage/               JSON response
  faiss.index              { answer, sources, confidence }
  metadata.pkl
```

---

## Quickstart (Local)

### 1. Prerequisites

- Python 3.11+
- An OpenAI API key

### 2. Clone & install

```bash
git clone <repo-url>
cd akar-rag-backend
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
pip install -r requirements.txt
```

### 3. Place the PDF

```
data/Akar website Consolidated.pdf
```

### 4. Start the server

```bash
make dev          # development (auto-reload)
# or
make run          # production
```

### 5. Ingest the PDF

```bash
# Option A: via API (server must be running)
make ingest

# Option B: direct Python call (no server needed)
make ingest-direct
```

Expected output:
```json
{ "status": "success", "sections": 5, "chunks": 42 }
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does AKAR offer?"}'
```

---

## Docker

```bash
# Build and run
make docker-build
make docker-up

# Ingest inside container
make docker-ingest

# Stop
make docker-down
```

---

## API Reference

### `POST /api/ingest`

Reads `data/Akar website Consolidated.pdf`, parses sections, chunks, embeds, and persists the FAISS index.

**Response:**
```json
{
  "status": "success",
  "sections": 5,
  "chunks": 47
}
```

---

### `POST /api/chat`

**Request:**
```json
{
  "question": "What is AKAR's approach to Social Impact Assessment?"
}
```

**Response:**
```json
{
  "answer": "AKAR offers end-to-end Social Impact Assessment services ...",
  "sources": [
    {
      "url": "https://akar-strategic-consultants.netlify.app",
      "section_title": "SOLUTIONS",
      "snippet": "Social impact is no longer an afterthought ..."
    }
  ],
  "confidence": "high"
}
```

**Confidence levels:**
| Level | Meaning |
|-------|---------|
| `high` | Top similarity score ≥ 0.75 and multiple matching chunks |
| `medium` | Top score ≥ 0.55 or only one good chunk |
| `low` | Weak matches or answer not found |

**Error cases:**
- `503` — Index not built yet (call `/api/ingest` first)
- `429` — Rate limit exceeded (30 req / 60s per IP)
- `422` — Validation error (question too long or empty)

---

### `GET /health`

```json
{ "status": "ok", "service": "akar-rag-backend" }
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | Your OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat/completion model |
| `TOP_K` | `6` | Chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Target chunk size in chars |
| `CHUNK_OVERLAP` | `175` | Overlap between chunks |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.75` | Cosine score for "high" |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.55` | Cosine score for "medium" |

---

## Project Structure

```
akar-rag-backend/
├── app/
│   ├── main.py                  # FastAPI app, CORS, rate limiter
│   ├── routers/
│   │   ├── ingest.py            # POST /api/ingest
│   │   └── chat.py              # POST /api/chat
│   └── services/
│       ├── pdf_parser.py        # Section extraction from PDF
│       ├── index.py             # Chunking, embedding, FAISS
│       └── rag.py               # Retrieval + LLM answer
├── data/
│   └── Akar website Consolidated.pdf
├── storage/                     # Auto-created on ingest
│   ├── faiss.index
│   └── metadata.pkl
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Section Parsing Logic

The PDF parser detects section boundaries using this regex pattern:

```
<SECTION TITLE> ( https://some-url )
```

For example:
```
SOLUTIONS ( https://akar-strategic-consultants.netlify.app )
```

The URL is stored as metadata on every chunk originating from that section and is returned as the `source` link in chat responses.

---

## Notes for Frontend Integration

- Enable CORS is set to `allow_origins=["*"]` by default. Update this in `app/main.py` to your specific frontend domain(s) before going to production.
- Always check `confidence` to optionally surface a warning to the user on `"low"` confidence answers.
- Sources always include at least 1 URL (the closest matching section even when answer is "not found").
- Rate limit: 30 requests per IP per 60 seconds (in-memory; resets on server restart).
