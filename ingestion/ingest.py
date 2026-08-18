from pathlib import Path
from typing import List

from config.logging import setup_logging
from config.settings import ensure_dirs, get_settings
from core.models import ModelHub
from core.types import DocumentChunk
from indexing.build import build_indexes
from ingestion.parsers import compute_file_fingerprint, parse_document

log = setup_logging()


def embed_chunk_batch(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    model = ModelHub.get_embed_model()
    return model.encode(
        texts,
        batch_size=settings.gpu_embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def ingest_documents(docs_dir=None, db_dir=None, use_gpu=False):
    settings = get_settings()
    ensure_dirs(settings)
    docs_dir = docs_dir or settings.docs_dir
    db_dir = db_dir or settings.db_dir
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        log.warning("Created %s. Add documents and re-run.", docs_dir)
        return

    files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.md"))
    if not files:
        log.error("No documents found.")
        return

    fingerprints = {}
    new_files = []
    for file_path in files:
        fp = compute_file_fingerprint(file_path)
        if fp not in fingerprints:
            fingerprints[fp] = file_path.name
            new_files.append(file_path)
        else:
            log.info("Skipping duplicate: %s (same as %s)", file_path.name, fingerprints[fp])

    all_chunks: List[DocumentChunk] = []
    for file_path in new_files:
        chunks = parse_document(file_path)
        all_chunks.extend(chunks)

    seen_chunks = set()
    unique_chunks: List[DocumentChunk] = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen_chunks:
            seen_chunks.add(chunk.chunk_id)
            unique_chunks.append(chunk)

    log.info("%s unique chunks from %s files", len(unique_chunks), len(files))
    texts = [c.text for c in unique_chunks]
    log.info("Generating embeddings...")
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), settings.gpu_embed_batch_size):
        batch = texts[i : i + settings.gpu_embed_batch_size]
        emb = embed_chunk_batch(batch)
        embeddings.extend(emb)

    for i, chunk in enumerate(unique_chunks):
        chunk.dense_embedding = embeddings[i]

    build_indexes(unique_chunks, db_dir=db_dir)
    log.info("All indexes built successfully.")
