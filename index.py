#!/usr/bin/env python3
"""RAG engine entrypoint and CLI."""

import argparse
import asyncio
import time

from api.app import app, stream_ollama
from config.logging import setup_logging
from config.settings import ensure_dirs, get_settings
from core.deps import App, uvicorn
from evaluation.eval import evaluate
from ingestion.ingest import ingest_documents
from retrieval.retriever import UltimateRetriever
from ui.tui import RAGDashboard

log = setup_logging()


def main():
    settings = get_settings()
    ensure_dirs(settings)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ingest")
    sub.add_parser("api")
    sub.add_parser("tui")
    sub.add_parser("eval")
    q_parser = sub.add_parser("query")
    q_parser.add_argument("text")
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_documents()
    elif args.cmd == "api":
        if not uvicorn or app is None:
            log.error("FastAPI and uvicorn are required")
            return
        uvicorn.run(app, host=settings.api_host, port=settings.api_port)
    elif args.cmd == "tui":
        if App is None or RAGDashboard is None:
            log.error("Textual not installed")
            return
        RAGDashboard().run()
    elif args.cmd == "eval":
        evaluate()
    elif args.cmd == "query":
        engine = UltimateRetriever()
        start = time.time()
        results = engine.search(args.text)
        print("Retrieved %s chunks in %.3fs" % (len(results), time.time() - start))
        for i, r in enumerate(results):
            print("%s. [%s:%s] %s..." % (i + 1, r.chunk.source_document, r.chunk.page_number, r.chunk.text[:80]))
        print("\nGenerating answer...")
        prompt = "Answer using only the context.\n%s\nQuestion: %s\nAnswer:" % (
            ". ".join(r.chunk.text for r in results),
            args.text,
        )
        for token in asyncio.run(stream_ollama(prompt)):
            print(token, end="", flush=True)
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
