import numpy as np
import pytest
from ranking.fusion import FusionModel


def test_fusion_model_initialization():
    fusion = FusionModel()
    assert fusion.model is None  # Initially untrained

def test_fusion_model_train_and_predict():
    fusion = FusionModel(model_path="test_fusion.txt")
    # Features: [vec, bm25, splade, colbert]
    X = np.array([
        [0.9, 10.0, 5.0, 15.0],
        [0.1, 1.0, 0.5, 2.0],
        [0.8, 8.0, 4.0, 12.0]
    ])
    # Labels (relevance 0-1)
    y = np.array([1.0, 0.0, 1.0])
    groups = np.array([3]) # All 3 samples in one query group
    fusion.fit(X, y, groups)
    assert fusion.model is not None
    
    # Predict
    preds = fusion.predict(X)
    assert len(preds) == 3
