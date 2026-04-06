"""
Embedding loaders for online inference.
Target layout for B1_hour_sin_cos: [PhoBERT_pretrain(88)] + [TFIDF(120)]
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class EmbeddingService:
    def __init__(self, project_root: Path, enable_bert: bool = True):
        self.project_root = Path(project_root)
        self.enable_bert = enable_bert

        self._bert_tokenizer = None
        self._bert_model = None

        self._tfidf_vectorizer = None
        self._tfidf_svd = None

        self._tfidf_dim = 120
        self._bert_dim = 88

        self._init_tfidf_pipeline()

    @property
    def bert_dim(self) -> int:
        return self._bert_dim

    @property
    def tfidf_dim(self) -> int:
        return self._tfidf_dim

    def _init_tfidf_pipeline(self) -> None:
        """Fit TF-IDF + SVD(120) from raw dataset so online text can be transformed."""
        data_path = self.project_root / "data" / "raw" / "data.csv"
        if not data_path.exists():
            return

        try:
            import pandas as pd
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            df = pd.read_csv(data_path, usecols=["post_message"])
            texts = [str(x) for x in df["post_message"].fillna("").tolist()]

            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 2),
                min_df=2,
            )
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

            self._tfidf_svd = TruncatedSVD(n_components=self._tfidf_dim, random_state=42)
            self._tfidf_svd.fit(tfidf_matrix)
        except Exception:
            self._tfidf_vectorizer = None
            self._tfidf_svd = None

    def _load_bert(self) -> None:
        if not self.enable_bert:
            return
        if self._bert_model is not None and self._bert_tokenizer is not None:
            return

        model_dir = self.project_root / "model" / "bert_embedding" / "phobert-v2-finetuned"
        if not model_dir.exists():
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            self._bert_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
            self._bert_model = AutoModel.from_pretrained(str(model_dir))
            self._bert_model.eval()
        except Exception:
            self._bert_tokenizer = None
            self._bert_model = None

    def encode(self, text_bert: str, text_tfidf: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Returns dict:
        - phobert_pretrain_embedding: np.ndarray shape (88,) or None
        - tfidf_embedding: np.ndarray shape (120,) or None
        """
        output: Dict[str, Optional[np.ndarray]] = {
            "phobert_pretrain_embedding": None,
            "tfidf_embedding": None,
        }

        if self._tfidf_vectorizer is not None and self._tfidf_svd is not None:
            try:
                tfidf_vec = self._tfidf_vectorizer.transform([text_tfidf])
                reduced = self._tfidf_svd.transform(tfidf_vec)[0].astype(np.float32)
                output["tfidf_embedding"] = reduced
            except Exception:
                output["tfidf_embedding"] = None

        self._load_bert()
        if self._bert_model is not None and self._bert_tokenizer is not None:
            try:
                import torch

                with torch.no_grad():
                    tokenized = self._bert_tokenizer(
                        text_bert,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                    )
                    out = self._bert_model(**tokenized)
                    cls_vec = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
                    output["phobert_pretrain_embedding"] = cls_vec[: self._bert_dim]
            except Exception:
                output["phobert_pretrain_embedding"] = None

        return output
