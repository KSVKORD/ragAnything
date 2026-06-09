# RAG-Anything
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
