import os
import pickle
from typing import List, Tuple

import numpy as np

from config.settings import get_settings
from core.deps import torch
from core.models import ModelHub
from core.types import DocumentChunk


class SPLADEIndex:
    def __init__(self, db_dir=None, path=None):
        settings = get_settings()
        db_dir = db_dir or settings.db_dir
        self.path = path or os.path.join(db_dir, "splade.pkl")
        self.doc_weights = []
        self.metadata = []

    def build(self, chunks: List[DocumentChunk]):
        tokenizer, model = ModelHub.get_splade()
        self.metadata = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]
        for chunk in chunks:
            inputs = tokenizer(chunk.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            if ModelHub.get_device() == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                term_weights, _ = torch.max(logits, dim=1)
                term_weights = term_weights.squeeze(0)
                weights_dict = {}
                for idx in term_weights.nonzero(as_tuple=False).flatten().tolist():
                    token = tokenizer.convert_ids_to_tokens(idx)
                    weights_dict[token] = term_weights[idx].item()
                chunk.splade_weights = weights_dict
                self.doc_weights.append(weights_dict)
        with open(self.path, "wb") as f:
            pickle.dump({"weights": self.doc_weights, "metadata": self.metadata}, f)

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.doc_weights = data["weights"]
        self.metadata = data["metadata"]

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        tokenizer, model = ModelHub.get_splade()
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=32)
        if ModelHub.get_device() == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            q_weights, _ = torch.max(logits, dim=1)
            q_weights = q_weights.squeeze(0)
        q_dict = {}
        for idx in q_weights.nonzero(as_tuple=False).flatten().tolist():
            token = tokenizer.convert_ids_to_tokens(idx)
            q_dict[token] = q_weights[idx].item()
        scores = []
        for doc_w in self.doc_weights:
            score = sum(q_dict.get(t, 0.0) * w for t, w in doc_w.items())
            scores.append(score)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.metadata[i]["chunk_id"], scores[i]) for i in top_indices]
