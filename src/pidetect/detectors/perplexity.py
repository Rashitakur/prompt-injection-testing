"""Perplexity-based detectors (Jain et al., 2023; Alon & Kamfonas, 2023).

One forward pass of a small causal LM yields per-token negative log-likelihoods,
from which three variants are derived:
    <name>-ppl      mean NLL (= log perplexity); zero-shot, threshold on val
    <name>-winppl   max mean-NLL over sliding windows of `window` tokens; zero-shot
    <name>-lr       logistic regression on [mean, max-window, std, log #tokens]
                    (supervised, as in Alon & Kamfonas)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import count_params, dtype_kwargs, length_sorted_batches, torch_dtype
from .base import Detector

LOG = logging.getLogger(__name__)


class Perplexity(Detector):
    def _ensure(self):
        if hasattr(self, "model"):
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self.spec
        self.tok = AutoTokenizer.from_pretrained(s["model"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        dtype = torch_dtype(s.get("dtype", "fp32"), self.device)
        self.model = AutoModelForCausalLM.from_pretrained(s["model"], **dtype_kwargs(dtype)).to(self.device).eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def features(self, texts: Sequence[str]) -> np.ndarray:
        """[mean_nll, max_window_nll, std_nll, log_n_tokens] per text."""
        import torch

        self._ensure()
        s, texts = self.spec, list(texts)
        w, max_len = int(s.get("window", 16)), int(s.get("max_len", 1024))
        feats = np.zeros((len(texts), 4), dtype=np.float32)
        with torch.inference_mode():
            for idx in length_sorted_batches(texts, s.get("batch_size", 32)):
                enc = [self.tok([texts[i]], add_special_tokens=False, truncation=True,
                                max_length=max_len - 1)["input_ids"][0] for i in idx]
                seqs = [[self.bos] + e for e in enc]   # BOS so the first real token is scored
                L = max(len(x) for x in seqs)
                ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), L), dtype=torch.long)
                for j, x in enumerate(seqs):
                    ids[j, : len(x)] = torch.tensor(x)
                    mask[j, : len(x)] = 1
                ids, mask = ids.to(self.device), mask.to(self.device)
                logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                nll = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), ids[:, 1:], reduction="none")   # (B, L-1)
                tmask = mask[:, 1:].float()
                nll = (nll * tmask).cpu().numpy()
                lens = tmask.sum(1).long().cpu().numpy()
                for j, i in enumerate(idx):
                    n = int(lens[j])
                    if n == 0:
                        continue
                    v = nll[j, :n]
                    if n <= w:
                        win = v.mean()
                    else:
                        c = np.concatenate([[0.0], np.cumsum(v)])
                        win = ((c[w:] - c[:-w]) / w).max()
                    feats[i] = [v.mean(), win, v.std(), np.log(n)]
        return feats

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        Xtr = self.features(train.text.tolist())
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        self.lr.fit(Xtr, train.label.values)

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        F = self.features(texts)
        return {self.v("ppl"): F[:, 0], self.v("winppl"): F[:, 1], self.v("lr"): self.lr.predict_proba(F)[:, 1]}

    def save(self, out_dir: Path) -> None:
        joblib.dump(self.lr, out_dir / "ppl_lr.joblib")

    def load(self, out_dir: Path) -> None:
        self.lr = joblib.load(out_dir / "ppl_lr.joblib")
        self._ensure()

    def n_params(self) -> int:
        self._ensure()
        return count_params(self.model)
