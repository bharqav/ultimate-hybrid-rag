import os
import pickle
from typing import List, Tuple

from config.settings import get_settings
from core.models import ModelHub
from core.types import DocumentChunk


class ColBERTStore:
    def __init__(self, db_dir=None, index_name="colbert_index", nbits=2):
        settings = get_settings()
        self.index_name = index_name
        self.nbits = nbits
        self.colbert_dir = os.path.join(db_dir or settings.db_dir, "colbert")
        self.searcher = None
        self.pid_to_chunk = {}

    def build(self, chunks: List[DocumentChunk]):
        from colbert import Indexer
        from colbert.infra import RunConfig

        os.makedirs(self.colbert_dir, exist_ok=True)
        collection_path = os.path.join(self.colbert_dir, "collection.tsv")
        with open(collection_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                f.write(f"{idx}\t{chunk.text}\n")
                chunk.colbert_doc_id = idx
                self.pid_to_chunk[idx] = chunk.chunk_id
        indexer = Indexer(checkpoint=get_settings().colbert_model_name, config=RunConfig(nbits=self.nbits))
        indexer.index(name=self.index_name, collection=collection_path, overwrite=True)
        with open(os.path.join(self.colbert_dir, "pid_map.pkl"), "wb") as f:
            pickle.dump(self.pid_to_chunk, f)

    def load(self):
        self.searcher = ModelHub.get_colbert_searcher()
        with open(os.path.join(self.colbert_dir, "pid_map.pkl"), "rb") as f:
            self.pid_to_chunk = pickle.load(f)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.searcher is None:
            self.load()
        results = self.searcher.search(query, k=k)
        return [(self.pid_to_chunk[pid], score) for pid, rank, score in results]
