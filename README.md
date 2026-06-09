# RAG-Anything

Multimodal RAG over your documents (text, tables, charts, formulas), built on
[RAG-Anything](https://github.com/hkuds/rag-anything) / LightRAG.

- **LLM + vision + embedding** → Qwen via the DashScope API (OpenAI-compatible)
- **Document parsing** → MinerU (uses an NVIDIA GPU if available)
- **Storage** → PostgreSQL (KV + doc-status), Qdrant (vectors), Neo4j (graph)
- **Interface** → FastAPI HTTP service (`api.py`) + CLI (`main.py`)

## Deploy with Docker (server)

Requires Docker, the NVIDIA Container Toolkit, and a working `nvidia-smi` on the host.

```bash
cp .env.example .env          # set DASHSCOPE_API_KEY (+ region URL); DB hosts are set by compose
docker compose up -d --build  # app (:8000) + Postgres + Qdrant + Neo4j

# ingest (one-off; put PDFs in ./documents first)
docker compose run --rm app python3 main.py ingest documents/manual.pdf
docker compose run --rm app python3 main.py ingest documents/manual.pdf --start 1 --end 20

# query the running service
curl -s localhost:8000/health
curl -s -XPOST localhost:8000/query -H 'content-type: application/json' \
     -d '{"question":"What is data manipulation?","mode":"naive"}'
curl -s localhost:8000/status
```

API: `GET /health`, `POST /query {question,mode,top_k}`, `POST /ingest {path,start?,end?}`, `GET /status`.

Notes:
- First ingest downloads MinerU models (~GB) into the `modelcache` volume — slow once, cached after.
- The CUDA base in `Dockerfile` (`12.4.1`) must be ≤ the host driver's CUDA (`nvidia-smi`); lower the tag if needed.
- The API is unauthenticated — keep it behind a firewall / reverse proxy, or add an API key.

### China networks (no Docker Hub access)
All three China blockers are handled by config, no host daemon changes needed:
- **Images** (DB + CUDA): pulled via the `REGISTRY` prefix in `.env` — defaults to **`docker.m.daocloud.io`** (DaoCloud). Change it to another mirror or `docker.io` as needed; it applies to every image and the build base.
- **pip** (build): Tsinghua mirror (`Dockerfile` `PIP_INDEX_URL` arg).
- **MinerU models** (ingest): ModelScope (`MINERU_MODEL_SOURCE` in `docker-compose.yml`).

So in China just keep the defaults and run `docker compose up -d --build`. Outside China, set `REGISTRY=docker.io` in `.env` (and optionally `docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple`).

## Run locally (no Docker)

1. **Python 3.10–3.13** (MinerU does not support 3.14+):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the databases: `docker compose up -d postgres qdrant neo4j`
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
- See `DEPLOYMENT_NOTES.md` for deployment gotchas.
