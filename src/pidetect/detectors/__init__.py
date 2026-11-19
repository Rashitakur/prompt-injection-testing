"""Detector registry."""
from __future__ import annotations

from .base import Detector

_TYPES = {
    "tfidf_lr": ("classical", "TfidfLR"),
    "embed_lr": ("classical", "EmbedLR"),
    "finetune": ("finetune", "FineTuned"),
    "hf_classifier": ("finetune", "HFClassifier"),
    "perplexity": ("perplexity", "Perplexity"),
    "llm_judge": ("llm_judge", "LLMJudge"),
}


def build_detector(name: str, cfg: dict, seed: int, device: str) -> Detector:
    import importlib

    spec = cfg["detectors"][name]
    mod, cls = _TYPES[spec["type"]]
    klass = getattr(importlib.import_module(f".{mod}", __name__), cls)
    return klass(name, spec, cfg, seed, device)


__all__ = ["Detector", "build_detector"]
