#!/usr/bin/env python3
"""CLI for the RAG pipeline.

    python main.py ingest [PATH] [--start N --end M]
    python main.py query "your question" [--mode naive|local|global|hybrid]
    python main.py status
    python main.py preview PATH [--start N --end M]
    python main.py reset [--yes]
"""
import sys
import asyncio
import argparse
from collections import Counter
from pathlib import Path

import pipeline as P
import metrics


async def cmd_ingest(args):
    rag = await P.build_rag()
    docs = P.collect_documents(args.path or P.DOCUMENTS_DIR, rag)
    if not docs:
        sys.exit(f"No supported documents found in {args.path or P.DOCUMENTS_DIR}")
    print(f"Found {len(docs)} document(s).")
    for i, doc in enumerate(docs, 1):
        rng = f" (pages {args.start}-{args.end})" if args.start else ""
        print(f"[{i}/{len(docs)}] {doc.name}{rng}")
        await P.ingest_one(rag, doc, args.start, args.end)
    print("Done.")
    report = metrics.write_report("ingest")
    if report:
        print(f"Performance report: {report}")


async def cmd_query(args):
    rag = await P.build_rag()
    print(await P.query(rag, args.question, mode=args.mode, top_k=args.top_k))
    report = metrics.write_report("query")
    if report:
        print(f"Performance report: {report}")


async def cmd_status(args):
    rag = await P.build_rag()
    info = await P.status(rag)
    print("By status:", info["counts"] or "(none)")
    for d in info["documents"]:
        print(f"  {str(d['chunks']):>5} chunks  {d['file_path']}")


async def cmd_preview(args):
    rag = await P.build_rag()
    content_list, _ = await P.parse_blocks(rag, args.path, args.start, args.end)
    print("block types:", dict(Counter(b.get("type") for b in content_list)))
    drop = [b for b in content_list if P.is_noise(b)]
    print(f"would drop {len(drop)}:")
    for b in drop:
        print(f"  [{b.get('type')}] {((b.get('text') or '').strip()[:90])!r}")


async def cmd_reset(args):
    rag = await P.build_rag()
    if not args.yes and input("Type YES to delete all indexed data: ").strip().upper() != "YES":
        print("Cancelled.")
        return
    await P.reset(rag)
    print(f"Cleared workspace={P.RAG_WORKSPACE}.")


def main():
    p = argparse.ArgumentParser(description="Multimodal RAG (Qwen API + MinerU + Postgres/Qdrant/Neo4j)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="parse + index a file or folder")
    pi.add_argument("path", nargs="?", default=None)
    pi.add_argument("--start", type=int, help="first page, 1-indexed (PDF only)")
    pi.add_argument("--end", type=int, help="last page, 1-indexed inclusive")

    pq = sub.add_parser("query", help="ask a question")
    pq.add_argument("question")
    pq.add_argument("--mode", default="naive", choices=["naive", "local", "global", "hybrid", "mix"])
    pq.add_argument("--top-k", type=int, default=5)

    sub.add_parser("status", help="show indexed documents")

    pp = sub.add_parser("preview", help="show parsed block types + what the filter would drop")
    pp.add_argument("path")
    pp.add_argument("--start", type=int)
    pp.add_argument("--end", type=int)

    pr = sub.add_parser("reset", help="wipe the workspace")
    pr.add_argument("--yes", action="store_true")

    args = p.parse_args()
    if args.cmd in ("ingest", "preview") and (args.start is None) != (args.end is None):
        p.error("--start and --end must be given together")
    fns = {"ingest": cmd_ingest, "query": cmd_query, "status": cmd_status,
           "preview": cmd_preview, "reset": cmd_reset}
    asyncio.run(fns[args.cmd](args))


if __name__ == "__main__":
    main()
