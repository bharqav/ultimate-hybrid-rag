import os
from dataclasses import dataclass
from pathlib import Path


def _get_env(name, default):
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _get_env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _get_env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    embed_model_name: str = "all-MiniLM-L6-v2"
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_model_default: str = "llama3"
    ollama_url: str = "http://localhost:11434/api/generate"
    db_dir: str = "./db"
    docs_dir: str = "./docs"
    target_chunk_tokens: int = 500
    chunk_overlap: int = 50
    max_context_tokens: int = 4000
    rrf_k: int = 60
    vector_top_n: int = 60
    bm25_top_n: int = 60
    splade_top_n: int = 60
    colbert_top_n: int = 60
    rerank_top_n: int = 20
    final_top_k: int = 5
    gpu_embed_batch_size: int = 256
    ingestion_workers: int = 4
    splade_model_name: str = "naver/splade-cocondenser-ensembledistil"
    colbert_model_name: str = "colbert-ir/colbertv2.0"
    max_gpu_memory: float = 0.8
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @classmethod
    def from_env(cls):
        return cls(
            embed_model_name=_get_env("RAG_EMBED_MODEL", cls.embed_model_name),
            cross_encoder_name=_get_env("RAG_CROSS_ENCODER", cls.cross_encoder_name),
            ollama_model_default=_get_env("RAG_OLLAMA_MODEL", cls.ollama_model_default),
            ollama_url=_get_env("RAG_OLLAMA_URL", cls.ollama_url),
            db_dir=_get_env("RAG_DB_DIR", cls.db_dir),
            docs_dir=_get_env("RAG_DOCS_DIR", cls.docs_dir),
            target_chunk_tokens=_get_env_int("RAG_TARGET_CHUNK_TOKENS", cls.target_chunk_tokens),
            chunk_overlap=_get_env_int("RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            max_context_tokens=_get_env_int("RAG_MAX_CONTEXT_TOKENS", cls.max_context_tokens),
            rrf_k=_get_env_int("RAG_RRF_K", cls.rrf_k),
            vector_top_n=_get_env_int("RAG_VECTOR_TOP_N", cls.vector_top_n),
            bm25_top_n=_get_env_int("RAG_BM25_TOP_N", cls.bm25_top_n),
            splade_top_n=_get_env_int("RAG_SPLADE_TOP_N", cls.splade_top_n),
            colbert_top_n=_get_env_int("RAG_COLBERT_TOP_N", cls.colbert_top_n),
            rerank_top_n=_get_env_int("RAG_RERANK_TOP_N", cls.rerank_top_n),
            final_top_k=_get_env_int("RAG_FINAL_TOP_K", cls.final_top_k),
            gpu_embed_batch_size=_get_env_int("RAG_GPU_EMBED_BATCH", cls.gpu_embed_batch_size),
            ingestion_workers=_get_env_int("RAG_INGESTION_WORKERS", cls.ingestion_workers),
            splade_model_name=_get_env("RAG_SPLADE_MODEL", cls.splade_model_name),
            colbert_model_name=_get_env("RAG_COLBERT_MODEL", cls.colbert_model_name),
            max_gpu_memory=_get_env_float("RAG_MAX_GPU_MEMORY", cls.max_gpu_memory),
            api_host=_get_env("RAG_API_HOST", cls.api_host),
            api_port=_get_env_int("RAG_API_PORT", cls.api_port),
        )


_SETTINGS = Settings.from_env()


def get_settings():
    return _SETTINGS


def ensure_dirs(settings: Settings):
    Path(settings.db_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
