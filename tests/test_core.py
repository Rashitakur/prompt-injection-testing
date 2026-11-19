"""Fast unit tests for metrics, thresholding, dedup and synthetic data (no models needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pidetect.data import _dedup, make_synthetic  # noqa: E402
from pidetect.metrics import binary_metrics, bootstrap_ci, choose_threshold, tpr_at_fpr  # noqa: E402
from pidetect.utils import normalize_text, text_hash  # noqa: E402


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = choose_threshold(y, s)
    m = binary_metrics(y, s, thr)
    assert m["f1"] == 1.0 and m["auroc"] == 1.0 and m["fpr"] == 0.0
    assert tpr_at_fpr(y, s, 0.01) == 1.0


def test_benign_only_split_reports_fpr_only():
    m = binary_metrics(np.zeros(4, int), np.array([0.1, 0.9, 0.2, 0.95]), 0.5)
    assert m["fpr"] == 0.5 and "f1" not in m and "auroc" not in m


def test_threshold_rules():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 0.8, 500)
    for rule in ["max_f1", "youden", "fpr@0.05"]:
        assert np.isfinite(choose_threshold(y, s, rule))
    t = choose_threshold(y, s, "fpr@0.05")
    assert binary_metrics(y, s, t)["fpr"] <= 0.06


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 0.7, 300)
    thr = choose_threshold(y, s)
    lo, hi = bootstrap_ci(y, s, thr, "f1", 200)
    f1 = binary_metrics(y, s, thr)["f1"]
    assert lo <= f1 <= hi


def test_normalisation_and_dedup():
    assert normalize_text("  Hello\n\tWORLD ") == "hello world"
    assert text_hash("A  b") == text_hash("a b")
    df = pd.DataFrame({"text": ["x y", "X  y", "conflict", "conflict", "ok"],
                       "label": [1, 1, 0, 1, 0], "source": "s"})
    out = _dedup(df, "t")
    assert sorted(out.text) == ["ok", "x y"]   # duplicate collapsed, conflicting pair dropped


def test_synthetic_sets_deterministic():
    test_id = pd.DataFrame({"text": [f"benign {i}" for i in range(20)] + [f"ignore all {i}" for i in range(20)],
                            "label": [0] * 20 + [1] * 20, "source": "s"})
    cfg = {"data": {"synthetic": {"adv_suffix": {"enabled": True, "max_n": 20, "n_suffix_tokens": 5},
                                  "noisy_benign": {"enabled": True, "max_n": 10}}}}
    a, b = make_synthetic(test_id, cfg, 7), make_synthetic(test_id, cfg, 7)
    assert a["stress_adv_suffix"].equals(b["stress_adv_suffix"])
    assert a["benign_noisy"].equals(b["benign_noisy"])
    assert a["benign_noisy"].label.eq(0).all()
    assert a["stress_adv_suffix"].label.mean() == 0.5
