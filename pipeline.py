"""Core RAG pipeline: Qwen (DashScope API) + MinerU + Postgres/Qdrant/Neo4j.

Shared by main.py (CLI) and api.py (HTTP service).
"""
import os
import sys
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _env(key, default=None):
    return os.getenv(key, default)


def _flag(key, default):
    return str(os.getenv(key, str(default))).strip().lower() in ("1", "true", "yes", "on")


API_KEY = _env("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = _env("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = _env("QWEN_MODEL", "qwen3.5-flash")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(_env("EMBEDDING_DIM", "1024"))
MAX_TOKENS = int(_env("MAX_TOKENS", "8192"))

MINERU_BACKEND = _env("MINERU_BACKEND", "pipeline")
PARSE_METHOD = _env("PARSE_METHOD", "auto")
MINERU_FORMULA = _flag("MINERU_FORMULA", True)
MINERU_TABLE = _flag("MINERU_TABLE", False)

DOCUMENTS_DIR = _env("DOCUMENTS_DIR", "./documents")
WORKING_DIR = _env("WORKING_DIR", "./rag_storage")
OUTPUT_DIR = _env("OUTPUT_DIR", "./output")
RAG_WORKSPACE = _env("RAG_WORKSPACE", "default")

DROP_TYPES = {"header", "footer", "page_number"}
DROP_MAX_CHARS = int(_env("DROP_MAX_CHARS", "50"))

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ── Qwen (DashScope, OpenAI-compatible) ──────────────────────────────────────
async def _qwen_chat(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        if image_data:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]})
        else:
            messages.append({"role": "user", "content": prompt})
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.0),
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{QWEN_BASE_URL}/chat/completions", json=payload, headers=HEADERS)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await _qwen_chat(prompt, system_prompt, history_messages, **kwargs)


async def vision_model_func(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
    return await _qwen_chat(prompt, system_prompt, history_messages, image_data=image_data, messages=messages, **kwargs)


async def qwen_embed(texts):
    texts = list(texts)
    out = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(texts), 10):  # DashScope embeddings: <= 10 inputs/request
            r = await client.post(f"{QWEN_BASE_URL}/embeddings",
                                  json={"model": EMBEDDING_MODEL, "input": texts[i:i + 10]}, headers=HEADERS)
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
    return np.array(out, dtype=np.float32)


def _mineru_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


async def build_rag():
    """Construct and initialise a RAGAnything instance backed by the databases."""
    if not API_KEY or API_KEY.startswith("your_"):
        sys.exit("ERROR: set DASHSCOPE_API_KEY in .env (copy from .env.example).")
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc

    os.environ["MINERU_DEVICE_MODE"] = _mineru_device()
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser_output_dir=OUTPUT_DIR,
        parse_method=PARSE_METHOD,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=MINERU_FORMULA,
        max_concurrent_files=1,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=qwen_embed),
        lightrag_kwargs={
            "kv_storage": "PGKVStorage",
            "doc_status_storage": "PGDocStatusStorage",
            "vector_storage": "QdrantVectorDBStorage",
            "graph_storage": "Neo4JStorage",
            "workspace": RAG_WORKSPACE,
        },
    )
    await rag._ensure_lightrag_initialized()
    return rag


def is_noise(block):
    text = (block.get("text") or "").strip()
    return block.get("type") in DROP_TYPES and len(text) <= DROP_MAX_CHARS


def collect_documents(path, rag):
    p = Path(path)
    if p.is_file():
        return [p]
    exts = set(rag.config.supported_file_extensions)
    return sorted(f for f in p.rglob("*")
                  if f.is_file() and f.suffix.lower() in exts and not f.name.startswith("."))


async def parse_blocks(rag, path, start=None, end=None):
    """Parse a document into MinerU content blocks (start/end are 1-indexed inclusive, PDFs only)."""
    kwargs = {"backend": MINERU_BACKEND, "formula": MINERU_FORMULA, "table": MINERU_TABLE}
    if start is not None and Path(path).suffix.lower() == ".pdf":
        kwargs.update(start_page=start - 1, end_page=end - 1)
    return await rag.parse_document(file_path=str(path), output_dir=OUTPUT_DIR,
                                    parse_method=PARSE_METHOD, **kwargs)


async def ingest_one(rag, path, start=None, end=None):
    content_list, doc_id = await parse_blocks(rag, path, start, end)
    kept = [b for b in content_list if not is_noise(b)]
    dropped = len(content_list) - len(kept)
    print(f"  parsed {len(content_list)} blocks; dropped {dropped} metadata; indexing {len(kept)}")
    await rag.insert_content_list(content_list=kept, file_path=str(path), doc_id=doc_id)
    return {"file": str(path), "parsed": len(content_list), "dropped": dropped, "indexed": len(kept)}


async def query(rag, question, mode="naive", top_k=5):
    return await rag.aquery(question, mode=mode, top_k=top_k, enable_rerank=False)


async def status(rag):
    from lightrag.base import DocStatus
    counts = await rag.lightrag.doc_status.get_status_counts()
    processed = await rag.lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    docs = [{"file_path": getattr(st, "file_path", "?"),
             "chunks": getattr(st, "chunks_count", None)}
            for st in sorted(processed.values(), key=lambda s: getattr(s, "file_path", "") or "")]
    return {"counts": counts, "documents": docs}


async def reset(rag):
    lr = rag.lightrag
    stores = [lr.text_chunks, lr.full_docs, lr.full_entities, lr.full_relations,
              lr.entity_chunks, lr.relation_chunks, lr.llm_response_cache, lr.doc_status,
              lr.chunks_vdb, lr.entities_vdb, lr.relationships_vdb, lr.chunk_entity_relation_graph]
    for s in stores:
        try:
            await s.drop()
        except Exception as e:
            print(f"  drop failed {type(s).__name__}: {e}")
