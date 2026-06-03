import os
import asyncio
import argparse
from pathlib import Path
from functools import partial

from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
# Formula recognition (MFR model) is GPU-heavy — disable on machines with <4GB GPU memory
MINERU_FORMULA = os.getenv("MINERU_FORMULA", "false").lower() == "true"


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        **kwargs,
    )


def vision_model_func(
    prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
):
    if messages:
        return openai_complete_if_cache(
            VISION_MODEL, "", messages=messages, api_key=API_KEY, **kwargs
        )
    if image_data:
        return openai_complete_if_cache(
            VISION_MODEL,
            "",
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                },
            ],
            api_key=API_KEY,
            **kwargs,
        )
    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=partial(openai_embed.func, model=EMBEDDING_MODEL, api_key=API_KEY),
)


def collect_documents(path: Path, config: RAGAnythingConfig) -> list[Path]:
    """Return all supported files under path (file or directory)."""
    if path.is_file():
        return [path]
    extensions = set(config.supported_file_extensions)
    return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in extensions)


async def main(path: str, query: str):
    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    config = RAGAnythingConfig()
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    documents = collect_documents(Path(path), config)
    if not documents:
        raise ValueError(f"No supported documents found in: {path}")

    for i, doc in enumerate(documents, 1):
        print(f"[{i}/{len(documents)}] Processing: {doc}")
        await rag.process_document_complete(
        file_path=str(doc), backend=MINERU_BACKEND, formula=MINERU_FORMULA
    )

    print(f"\nQuery: {query}")
    result = await rag.aquery(query, mode="hybrid")
    print(f"Answer: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("path", help="Path to a document or a folder of documents")
    parser.add_argument("query", help="Question to ask about the document(s)")
    args = parser.parse_args()
    asyncio.run(main(args.path, args.query))
