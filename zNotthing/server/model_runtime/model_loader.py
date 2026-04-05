"""
Model loading and prediction for early_fusion LightGBM Baseline1 + hour_sin_cos.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np


class InferenceModel:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.model_path = self.project_root / "model" / "early_fusion" / "lightgbm" / "B1_hour_sin_cos_model.txt"
        self.meta_path = self.project_root / "model" / "early_fusion" / "lightgbm" / "B1_hour_sin_cos_meta.joblib"

        self.booster = None
        self.threshold = 0.5
        self.feature_names: List[str] = []

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        import joblib

        meta = joblib.load(self.meta_path)
        self.threshold = float(meta.get("threshold", 0.5))
        self.feature_names = list(meta.get("feature_names", []))

        self.booster = lgb.Booster(model_file=str(self.model_path))

    def _compose_input_vector(
        self,
        features: Dict[str, float],
        embedding_bundle: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        expected_total = int(self.booster.num_feature())
        handcraft = np.array([float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)

        expected_embed_dim = max(0, expected_total - len(handcraft))
        if expected_embed_dim == 0:
            return handcraft.reshape(1, -1)

        bert_part = np.zeros(88, dtype=np.float32)
        tfidf_part = np.zeros(120, dtype=np.float32)
        if embedding_bundle:
            b = embedding_bundle.get("phobert_pretrain_embedding")
            t = embedding_bundle.get("tfidf_embedding")
            if isinstance(b, np.ndarray) and b.size > 0:
                bert_part[: min(88, b.size)] = b[: min(88, b.size)]
            if isinstance(t, np.ndarray) and t.size > 0:
                tfidf_part[: min(120, t.size)] = t[: min(120, t.size)]

        embedding = np.concatenate([bert_part, tfidf_part], axis=0)
        if embedding.size < expected_embed_dim:
            pad = np.zeros(expected_embed_dim - embedding.size, dtype=np.float32)
            embedding = np.concatenate([embedding, pad], axis=0)
        elif embedding.size > expected_embed_dim:
            embedding = embedding[:expected_embed_dim]

        vector = np.concatenate([embedding, handcraft], axis=0)
        return vector.reshape(1, -1)

    def predict(
        self,
        features: Dict[str, float],
        embedding_bundle: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[int, float, Dict[str, float]]:
        if self.booster is None:
            raise RuntimeError("Model not loaded")
        if not self.feature_names:
            raise RuntimeError("Feature names missing in metadata")

        vector = self._compose_input_vector(features, embedding_bundle)
        prob_fake = float(self.booster.predict(vector)[0])
        label = 1 if prob_fake >= self.threshold else 0
        confidence = prob_fake if label == 1 else 1.0 - prob_fake

        return label, confidence, {
            "prob_fake": prob_fake,
            "threshold": self.threshold,
            "model_num_features": int(self.booster.num_feature()),
            "input_vector_dim": int(vector.shape[1]),
        }
