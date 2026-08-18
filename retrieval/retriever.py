import concurrent.futures
from typing import Dict, List

import numpy as np

from config.settings import get_settings
from core.models import ModelHub
from core.types import DocumentChunk, RetrievalResult
from ranking.fusion import FusionModel
from retrieval.bm25_store import BM25Store
from retrieval.cache import SemanticCache
from retrieval.colbert_store import ColBERTStore
from retrieval.gpu_scheduler import GPUScheduler
from retrieval.planner import QueryPlanner
from retrieval.splade_index import SPLADEIndex
from retrieval.vector_store import VectorStore


class UltimateRetriever:
    def __init__(self, use_gpu=False):
        settings = get_settings()
        self.use_gpu = use_gpu
        self.vec_store = VectorStore()
        self.bm25_store = BM25Store()
        self.splade_store = SPLADEIndex()
        self.colbert_store = ColBERTStore()
        self.fusion_model = FusionModel()
        self.reranker = ModelHub.get_cross_encoder()
        self.cache = SemanticCache()
        self.planner = QueryPlanner()
        self.scheduler = GPUScheduler()
        self.vec_store.load(use_gpu)
        self.bm25_store.load()
        self.splade_store.load()
        self.colbert_store.load()
        self.settings = settings
        self._build_metadata_map()

    def _build_metadata_map(self):
        self.metadata_lookup = {m["chunk_id"]: m for m in self.bm25_store.metadata}

    def _scores_to_results(
        self, chunk_ids: List[str], scores_dict: Dict[str, Dict[str, float]]
    ) -> List[RetrievalResult]:
        results = []
        for cid in chunk_ids:
            meta = self.metadata_lookup.get(cid)
            if not meta:
                continue
            chunk = DocumentChunk(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                source_document=meta["source"],
                page_number=meta["page"],
                section_title=meta.get("section", ""),
                token_count=meta["token_count"],
            )
            sc = scores_dict.get(cid, {})
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    vector_score=sc.get("vec", 0.0),
                    bm25_score=sc.get("bm25", 0.0),
                    splade_score=sc.get("splade", 0.0),
                    colbert_score=sc.get("colbert", 0.0),
                )
            )
        return results

    def search(self, query: str) -> List[RetrievalResult]:
        settings = self.settings
        q_emb = ModelHub.get_embed_model().encode([query], normalize_embeddings=True)[0]
        cached = self.cache.lookup(q_emb)
        if cached:
            return cached[: settings.final_top_k]

        weights = self.planner.classify(query)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            fut_vec = executor.submit(self.vec_store.search, q_emb.tolist(), settings.vector_top_n)
            fut_bm25 = executor.submit(self.bm25_store.search, query, settings.bm25_top_n)
            fut_splade = executor.submit(self.splade_store.search, query, settings.splade_top_n)
            fut_colbert = executor.submit(self.colbert_store.search, query, settings.colbert_top_n)

            vec_ids, vec_scores = fut_vec.result()
            bm25_ids_scores = fut_bm25.result()
            splade_ids_scores = fut_splade.result()
            colbert_ids_scores = fut_colbert.result()

        scores_map: Dict[str, Dict[str, float]] = {}
        for idx, cid in enumerate(vec_ids):
            scores_map.setdefault(cid, {})["vec"] = vec_scores[idx]
        for cid, score in bm25_ids_scores:
            scores_map.setdefault(cid, {})["bm25"] = score
        for cid, score in splade_ids_scores:
            scores_map.setdefault(cid, {})["splade"] = score
        for cid, score in colbert_ids_scores:
            scores_map.setdefault(cid, {})["colbert"] = score

        if self.fusion_model.model is not None:
            all_ids = list(scores_map.keys())
            X = []
            for cid in all_ids:
                sc = scores_map[cid]
                X.append([sc.get("vec", 0), sc.get("bm25", 0), sc.get("splade", 0), sc.get("colbert", 0)])
            X_np = np.array(X)
            fused_scores = self.fusion_model.predict(X_np)
            for cid, fs in zip(all_ids, fused_scores):
                scores_map[cid]["fused"] = fs
        else:
            for cid, sc in scores_map.items():
                fused = (
                    weights["vector"] * sc.get("vec", 0)
                    + weights["bm25"] * sc.get("bm25", 0)
                    + weights["splade"] * sc.get("splade", 0)
                    + weights["colbert"] * sc.get("colbert", 0)
                )
                scores_map[cid]["fused"] = fused

        sorted_ids = sorted(scores_map.items(), key=lambda x: x[1].get("fused", 0), reverse=True)[
            : settings.rerank_top_n
        ]
        candidates = self._scores_to_results([cid for cid, _ in sorted_ids], scores_map)

        if candidates and self.reranker is not None:
            pairs = [(query, r.chunk.text) for r in candidates]
            ce_scores = self.reranker.predict(pairs)
            for r, sc in zip(candidates, ce_scores):
                r.fused_score = float(sc)
            candidates.sort(key=lambda x: x.fused_score, reverse=True)
        else:
            for r in candidates:
                r.fused_score = scores_map.get(r.chunk.chunk_id, {}).get("fused", 0.0)

        final = candidates[: settings.final_top_k]
        self.cache.store(q_emb, final)
        return final
