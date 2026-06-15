"""FastAPI service for the RAG pipeline.

    uvicorn api:app --host 0.0.0.0 --port 8000

The RAGAnything instance is built once at startup and reused across requests.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pipeline as P


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = await P.build_rag()
    yield


app = FastAPI(title="RAG-Anything", lifespan=lifespan)


class QueryIn(BaseModel):
    question: str
    mode: str = "naive"
    top_k: int = 5


class IngestIn(BaseModel):
    path: str                 # file or folder, resolved under DOCUMENTS_DIR
    start: int | None = None  # 1-indexed inclusive, PDF only
    end: int | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(body: QueryIn):
    answer = await P.query(app.state.rag, body.question, mode=body.mode, top_k=body.top_k)
    return {"answer": answer}


@app.post("/ingest")
async def ingest(body: IngestIn):
    if (body.start is None) != (body.end is None):
        raise HTTPException(400, "start and end must be given together")
    rag = app.state.rag
    target = Path(body.path)
    if not target.is_absolute():
        target = Path(P.DOCUMENTS_DIR) / body.path
    docs = P.collect_documents(target, rag)
    if not docs:
        raise HTTPException(404, f"No supported documents found at {target}")
    results = [await P.ingest_one(rag, doc, body.start, body.end) for doc in docs]
    return {"ingested": results}


@app.get("/status")
async def status():
    return await P.status(app.state.rag)
