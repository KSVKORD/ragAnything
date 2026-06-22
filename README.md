# RAG-Anything

Multimodal RAG over PDFs. Parsing by **dots.ocr** (local GPU, served by vLLM); LLM/vision/embeddings
by **Qwen** (DashScope API); storage in **Postgres + Qdrant + Neo4j**. FastAPI service + CLI, all via
Docker Compose.

## Quick start (Docker)

```bash
cp .env.example .env          # set DASHSCOPE_API_KEY (+ region URL); hosts are set by compose

# one-time: fetch the dots.ocr parser weights from ModelScope (China-friendly, no HuggingFace)
pip install modelscope
modelscope download --model rednote-hilab/dots.ocr-1.5 --local_dir ./data/dots-weights

docker compose up -d --build  # app(:8001) + dots-ocr(GPU) + Postgres + Qdrant + Neo4j

# ingest (put PDFs in ./documents first)
docker compose run --rm app python3 main.py ingest documents/manual.pdf
docker compose run --rm app python3 main.py ingest documents/manual.pdf --start 1 --end 20

# query the running service
curl -s localhost:8001/health
curl -s -XPOST localhost:8001/query -H 'content-type: application/json' \
     -d '{"question":"What is data manipulation?","mode":"naive"}'
curl -s localhost:8001/status
```

API: `GET /health`, `POST /query {question,mode,top_k}`, `POST /ingest {path,start?,end?}`, `GET /status`.

## Document parsing (dots.ocr)

- The `dots-ocr` service runs the parser on the **GPU** (vLLM, OpenAI-compatible). It needs the model
  weights in `./data/dots-weights` — downloaded above from **ModelScope** (no HuggingFace needed).
- PDFs are rasterized per page (PyMuPDF) and sent to dots.ocr: tables→HTML, formulas→LaTeX, figures→
  cropped images (captioned by Qwen). **PDF/image inputs only.**
- Tune render resolution with `DOTS_DPI` (default 200); the parser endpoint is `DOTS_OCR_URL`.

## Run locally (no Docker)

1. **Python 3.10–3.13**:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the backing services: `docker compose up -d postgres qdrant neo4j dots-ocr`
3. `cp .env.example .env` → set `DASHSCOPE_API_KEY`; match `QWEN_BASE_URL` to your key's region.

```bash
python main.py ingest                       # index ./documents (or a file; --start/--end for PDFs)
python main.py query "..." --mode hybrid
python main.py status
python main.py preview documents/x.pdf      # block types + what the filter drops
python main.py reset --yes
```

## Notes

- Re-ingesting the same file/range is skipped automatically (content-based dedup).
- Use `ingest` per document; avoid overlapping page ranges to prevent double-indexing.
