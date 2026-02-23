.PHONY: install ingest run dev docker-build docker-up docker-ingest clean

# ── Local development ─────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

ingest:
	@echo "▶ Running ingest via API call (server must be running)..."
	curl -s -X POST http://localhost:8000/api/ingest | python3 -m json.tool

ingest-direct:
	@echo "▶ Running ingest directly (no server needed)..."
	python3 -c "\
from dotenv import load_dotenv; load_dotenv(); \
from app.services.pdf_parser import parse_pdf_sections; \
from app.services.index import build_index; \
sections = parse_pdf_sections('data/Akar website Consolidated.pdf'); \
chunks = build_index(sections, 'storage'); \
print(f'✅  Ingested {len(sections)} sections → {chunks} chunks')"

dev:
	@echo "▶ Starting development server with auto-reload..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run:
	@echo "▶ Starting production server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-ingest:
	@echo "▶ Triggering ingest inside running container..."
	docker compose exec akar-rag curl -s -X POST http://localhost:8000/api/ingest | python3 -m json.tool

docker-down:
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf storage/faiss.index storage/metadata.pkl
	@echo "✅ Storage cleared. Run 'make ingest-direct' or 'make ingest' to rebuild."
