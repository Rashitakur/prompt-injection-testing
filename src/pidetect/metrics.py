"""Thresholding, metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def choose_threshold(y: np.ndarray, s: np.ndarray, rule: str = "max_f1") -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float(np.median(s))
    if rule == "youden":
        fpr, tpr, thr = roc_curve(y, s)
        return float(thr[np.argmax(tpr - fpr)])
    if rule.startswith("fpr@"):
        target = float(rule.split("@")[1])
        neg = np.sort(s[y == 0])
        return float(np.quantile(neg, 1 - target)) if len(neg) else 0.5
    p, r, thr = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def tpr_at_fpr(y, s, target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_metrics(y: np.ndarray, s: np.ndarray, thr: float, low_fpr: float = 0.01) -> dict:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    pred = (s >= thr).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    m = {"n": len(y), "n_pos": n_pos, "n_neg": n_neg, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
         "accuracy": (tp + tn) / max(1, len(y)), "threshold": thr}
    if n_pos:
        m["recall"] = tp / n_pos
    if n_neg:
        m["fpr"] = fp / n_neg
    if n_pos and n_neg:
        prec = tp / max(1, tp + fp)
        m.update({
            "precision": prec,
            "f1": 2 * prec * m["recall"] / max(1e-12, prec + m["recall"]),
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            f"tpr@{low_fpr:g}fpr": tpr_at_fpr(y, s, low_fpr),
        })
    return m


def bootstrap_ci(y, s, thr, metric: str, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI (stratified by label so every resample has both classes)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)) if len(pos) else pos,
                              rng.choice(neg, len(neg)) if len(neg) else neg])
        m = binary_metrics(y[idx], s[idx], thr)
        if metric in m:
            vals.append(m[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def paired_bootstrap_pvalue(y, s_a, s_b, thr_a, thr_b, metric="f1", n_boot=1000, seed=0) -> float:
    """One-sided p-value that detector A is NOT better than B on `metric` (paired resampling)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    worse = 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a = binary_metrics(y[idx], np.asarray(s_a)[idx], thr_a).get(metric, np.nan)
        b = binary_metrics(y[idx], np.asarray(s_b)[idx], thr_b).get(metric, np.nan)
        worse += a <= b
    return (worse + 1) / (n_boot + 1)
