import os
import pickle
from typing import List

import numpy as np

from config.settings import get_settings
from core.types import RetrievalResult


class SemanticCache:
    def __init__(self, db_dir=None, path=None, threshold=0.95):
        settings = get_settings()
        db_dir = db_dir or settings.db_dir
        self.path = path or os.path.join(db_dir, "cache.pkl")
        self.threshold = threshold
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.cache = pickle.load(f)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)

    def lookup(self, query_emb: np.ndarray):
        for emb, res in self.cache.values():
            if np.dot(query_emb, emb) > self.threshold:
                self.hits += 1
                return res
        self.misses += 1
        return None

    def store(self, query_emb: np.ndarray, results: List[RetrievalResult]):
        key = tuple(query_emb.tolist()[:10])
        self.cache[key] = (query_emb, results)
        if len(self.cache) % 100 == 0:
            self.save()
