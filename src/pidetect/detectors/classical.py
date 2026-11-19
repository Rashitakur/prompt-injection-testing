"""TF-IDF + LR and sentence-embedding + LR baselines."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..utils import count_params, dtype_kwargs, length_sorted_batches
from .base import Detector

LOG = logging.getLogger(__name__)


def _fit_lr_grid(Xtr, ytr, Xva, yva, c_grid, seed) -> tuple[LogisticRegression, float, dict]:
    best, best_auc, curve = None, -1.0, {}
    for c in c_grid:
        lr = LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed)
        lr.fit(Xtr, ytr)
        auc = roc_auc_score(yva, lr.predict_proba(Xva)[:, 1]) if len(set(yva)) > 1 else 0.0
        curve[str(c)] = float(auc)
        if auc > best_auc:
            best, best_auc = lr, auc
    LOG.info("LR C-grid val AUROC: %s", curve)
    # refit on train only (val stays clean for threshold selection)
    return best, best_auc, curve


# =============================================================================
class TfidfLR(Detector):
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        mf = self.spec.get("max_features", 200000)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, max_features=mf, lowercase=True)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True, max_features=mf // 2, lowercase=True)
        Xtr = self._fit_transform(train.text.tolist())
        Xva = self._transform(val.text.tolist())
        self.lr, auc, self.c_curve = _fit_lr_grid(Xtr, train.label.values, Xva, val.label.values,
                                                  self.spec.get("c_grid", [1.0]), self.seed)

    def _fit_transform(self, texts):
        return sparse.hstack([self.char.fit_transform(texts), self.word.fit_transform(texts)]).tocsr()

    def _transform(self, texts):
        return sparse.hstack([self.char.transform(texts), self.word.transform(texts)]).tocsr()

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {self.v(): self.lr.predict_proba(self._transform(list(texts)))[:, 1]}

    def save(self, out_dir: Path) -> None:
        joblib.dump({"char": self.char, "word": self.word, "lr": self.lr}, out_dir / "tfidf_lr.joblib")

    def load(self, out_dir: Path) -> None:
        d = joblib.load(out_dir / "tfidf_lr.joblib")
        self.char, self.word, self.lr = d["char"], d["word"], d["lr"]

    def n_params(self) -> int:
        return int(self.lr.coef_.size + self.lr.intercept_.size)


# =============================================================================
class Encoder:
    """Frozen transformer encoder with mean/CLS pooling (no sentence-transformers dependency)."""

    def __init__(self, model_name: str, pooling: str, normalize: bool, max_len: int, device: str, dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **dtype_kwargs(dtype or torch.float32)).to(device).eval()
        self.pooling, self.normalize, self.max_len, self.device = pooling, normalize, max_len, device

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        import torch

        texts = list(texts)
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for idx in length_sorted_batches(texts, batch_size):
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
                if self.pooling == "cls":
                    e = h[:, 0]
                else:
                    m = mask.unsqueeze(-1).to(h.dtype)
                    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                if self.normalize:
                    e = torch.nn.functional.normalize(e.float(), dim=-1)
                out[idx] = e.float().cpu().numpy()
        return out


class EmbedLR(Detector):
    def _enc(self) -> Encoder:
        if not hasattr(self, "encoder"):
            s = self.spec
            self.encoder = Encoder(s["model"], s.get("pooling", "mean"), s.get("normalize", True),
                                   s.get("max_len", 256), self.device)
        return self.encoder

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        bs = self.spec.get("batch_size", 128)
        Xtr = self._enc().encode(train.text.tolist(), bs)
        Xva = self._enc().encode(val.text.tolist(), bs)
        self.lr, auc, self.c_curve = _fit_lr_grid(Xtr, train.label.values, Xva, val.label.values,
                                                  self.spec.get("c_grid", [1.0]), self.seed)

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        X = self._enc().encode(texts, self.spec.get("batch_size", 128))
        return {self.v(): self.lr.predict_proba(X)[:, 1]}

    def save(self, out_dir: Path) -> None:
        joblib.dump(self.lr, out_dir / "lr.joblib")

    def load(self, out_dir: Path) -> None:
        self.lr = joblib.load(out_dir / "lr.joblib")

    def n_params(self) -> int:
        return count_params(self._enc().model) + int(self.lr.coef_.size + 1)
