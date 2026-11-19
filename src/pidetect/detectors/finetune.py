"""Fine-tuned sequence classifier (DistilBERT by default) + off-the-shelf HF classifier reference."""
from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..utils import count_params, length_sorted_batches, torch_dtype
from .base import Detector

LOG = logging.getLogger(__name__)


def _predict_proba(model, tok, texts, max_len, batch_size, device, pos_index=1, amp_dtype=None) -> np.ndarray:
    import torch

    texts = list(texts)
    out = np.zeros(len(texts), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for idx in length_sorted_batches(texts, batch_size):
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=ids, attention_mask=mask).logits
            out[idx] = torch.softmax(logits.float(), -1)[:, pos_index].cpu().numpy()
    return out


class FineTuned(Detector):
    """Full fine-tuning with AdamW + linear warmup; best epoch selected on validation."""

    def _amp(self):
        if self.device != "cuda":
            return None
        import torch

        d = torch_dtype(self.spec.get("dtype", "auto"), self.device)
        return None if d == torch.float32 else d

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        s = self.spec
        self.tok = AutoTokenizer.from_pretrained(s["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(s["model"], num_labels=2).to(self.device)
        amp = self._amp()
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))

        texts, labels = train.text.tolist(), train.label.values.astype(int)
        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(list(range(len(texts))), batch_size=s["batch_size"], shuffle=True, generator=g)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(s["lr"]), weight_decay=s.get("weight_decay", 0.01))
        total = len(loader) * s["epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(s.get("warmup_ratio", 0.1) * total), total)

        best_metric, best_state, self.history = -math.inf, None, []
        for ep in range(s["epochs"]):
            self.model.train()
            run_loss = 0.0
            for step, idx in enumerate(loader):
                idx = idx.tolist()
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=s["max_len"], return_tensors="pt")
                y = torch.tensor(labels[idx], device=self.device)
                with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
                    logits = self.model(input_ids=enc["input_ids"].to(self.device),
                                        attention_mask=enc["attention_mask"].to(self.device)).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += loss.item()
            p = _predict_proba(self.model, self.tok, val.text.tolist(), s["max_len"],
                               s.get("eval_batch_size", 128), self.device, amp_dtype=amp)
            yv = val.label.values
            m = {"epoch": ep + 1, "train_loss": run_loss / max(1, len(loader)),
                 "val_auroc": float(roc_auc_score(yv, p)), "val_auprc": float(average_precision_score(yv, p))}
            self.history.append(m)
            LOG.info("epoch %d  %s", ep + 1, m)
            key = m["val_" + s.get("select_metric", "auroc")]
            if key > best_metric:
                best_metric, best_state = key, copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        s = self.spec
        return {self.v(): _predict_proba(self.model, self.tok, texts, s["max_len"],
                                         s.get("eval_batch_size", 128), self.device, amp_dtype=self._amp())}

    def save(self, out_dir: Path) -> None:
        self.model.save_pretrained(out_dir / "model")
        self.tok.save_pretrained(out_dir / "model")

    def load(self, out_dir: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(out_dir / "model")
        self.model = AutoModelForSequenceClassification.from_pretrained(out_dir / "model").to(self.device).eval()

    def n_params(self) -> int:
        return count_params(self.model)


class HFClassifier(Detector):
    """Off-the-shelf injection classifier used zero-shot (no training on our data)."""

    def _ensure(self):
        if hasattr(self, "model"):
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.spec["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.spec["model"]).to(self.device).eval()
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        want = str(self.spec.get("positive_label", "")).lower()
        self.pos = next((i for i, l in id2label.items() if l == want), 1)
        LOG.info("%s labels=%s positive index=%d", self.spec["model"], id2label, self.pos)
        self.default_thresholds = {self.v(): 0.5}

    def fit(self, train, val) -> None:
        self._ensure()

    def load(self, out_dir: Path) -> None:
        self._ensure()

    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        self._ensure()
        amp = None
        if self.device == "cuda":
            amp = torch_dtype("auto", self.device)   # bf16 where supported (DeBERTa is fp16-sensitive)
        return {self.v(): _predict_proba(self.model, self.tok, texts, self.spec.get("max_len", 512),
                                         self.spec.get("batch_size", 64), self.device, self.pos, amp)}

    def n_params(self) -> int:
        self._ensure()
        return count_params(self.model)
