import os
import pickle

import numpy as np

from config.settings import get_settings
from core.deps import LGBMRanker


class FusionModel:
    def __init__(self, db_dir=None, model_path=None):
        settings = get_settings()
        db_dir = db_dir or settings.db_dir
        self.model_path = model_path or os.path.join(db_dir, "fusion_model.lgb")
        self.model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

    def fit(self, X, y, groups):
        if LGBMRanker is None:
            raise RuntimeError("lightgbm not installed")
        self.model = LGBMRanker(objective="lambdarank", metric="ndcg", boosting_type="gbdt")
        self.model.fit(X, y, group=groups)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        return np.zeros(len(X))
