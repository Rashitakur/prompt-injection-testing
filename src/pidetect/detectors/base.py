"""Detector interface.

Every detector maps a list of strings to one or more score vectors ("variants").
Higher score = more likely to be a prompt injection. Thresholds are chosen
outside the detector, on the validation split, so all methods are calibrated
with the same rule.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class Detector(abc.ABC):
    #: thresholds to also report without validation calibration, {variant: thr}
    default_thresholds: dict[str, float] = {}

    def __init__(self, name: str, spec: dict, cfg: dict, seed: int, device: str):
        self.name, self.spec, self.cfg, self.seed, self.device = name, spec, cfg, seed, device

    # ---------------------------------------------------------------- required
    @abc.abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Train / calibrate. Zero-shot detectors may do nothing here."""

    @abc.abstractmethod
    def score(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Return {variant_name: scores (float array, len == len(texts))}."""

    # ---------------------------------------------------------------- optional
    def save(self, out_dir: Path) -> None:  # noqa: B027
        """Persist fitted state (needed by the latency benchmark)."""

    def load(self, out_dir: Path) -> None:  # noqa: B027
        """Restore fitted state written by save()."""

    def n_params(self) -> int | None:
        return None

    def set_batch_size(self, bs: int) -> None:
        self.spec = {**self.spec, "batch_size": bs, "eval_batch_size": bs}

    def v(self, suffix: str | None = None) -> str:
        return self.name if not suffix else f"{self.name}-{suffix}"
