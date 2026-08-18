import os
import pickle
from typing import List, Tuple

import numpy as np

from config.settings import get_settings
from core.deps import faiss


class VectorStore:
    def __init__(self, db_dir=None, index_path=None, id_map_path=None):
        settings = get_settings()
        db_dir = db_dir or settings.db_dir
        self.index_path = index_path or os.path.join(db_dir, "faiss.index")
        self.id_map_path = id_map_path or os.path.join(db_dir, "faiss_ids.pkl")
        self.dim = 384
        self.index = None
        self.id_map = []

    def build(self, embeddings: List[List[float]], chunk_ids: List[str]):
        if faiss is None:
            raise RuntimeError("faiss not installed")
        emb_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_np)
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, 1024, 16, 8)
        self.index.train(emb_np)
        self.index.add(emb_np)
        self.id_map = chunk_ids
        faiss.write_index(self.index, self.index_path)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, use_gpu=False):
        if faiss is None:
            raise RuntimeError("faiss not installed")
        self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        with open(self.id_map_path, "rb") as f:
            self.id_map = pickle.load(f)

    def search(self, query_emb: List[float], k: int) -> Tuple[List[str], List[float]]:
        if faiss is None:
            raise RuntimeError("faiss not installed")
        q = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        ids = [self.id_map[i] for i in indices[0] if i >= 0 and i < len(self.id_map)]
        return ids, scores[0][: len(ids)].tolist()
