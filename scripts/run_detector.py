#!/usr/bin/env python
"""Fit one detector, calibrate its threshold on validation, score every eval split.

Writes:
  results/runs/<variant>/seed<seed>/scores_<split>.parquet   (label, score, source)
  results/runs/<variant>/seed<seed>/meta.json                (threshold, timings, params, env)
  artifacts/<detector>/seed<seed>/                           (fitted state for latency benchmark)

    python scripts/run_detector.py --detector distilbert --seed 42
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from pidetect.data import eval_split_names, load_split
from pidetect.detectors import build_detector
from pidetect.metrics import binary_metrics, choose_threshold
from pidetect.utils import (ensure_dir, env_info, get_device, load_config, save_json, set_seed,
                            setup_logging, sync, timer)

LOG = logging.getLogger("run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-eval", type=int, default=None, help="debug: cap rows per eval split")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    spec = cfg["detectors"][args.detector]
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    set_seed(seed)
    device = get_device(args.device or cfg.get("device", "auto"))
    LOG.info("detector=%s seed=%d device=%s", args.detector, seed, device)

    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    det = build_detector(args.detector, cfg, seed, device)

    with timer() as t_fit:
        det.fit(train, val)
        sync(device)
    LOG.info("fit took %.1fs", t_fit["seconds"])

    rule = cfg["eval"].get("threshold_rule", "max_f1")
    val_scores = det.score(val.text.tolist())
    thresholds = {v: choose_threshold(val.label.values, s, rule) for v, s in val_scores.items()}
    LOG.info("thresholds (%s on val): %s", rule, thresholds)

    timings, split_rows = {}, {}
    all_scores = {v: {"val": s} for v, s in val_scores.items()}
    for split in eval_split_names(cfg):
        df = load_split(cfg, split)
        if args.max_eval:
            df = df.head(args.max_eval)
        if not len(df):
            continue
        with timer() as t:
            sc = det.score(df.text.tolist())
            sync(device)
        timings[split] = {"seconds": t["seconds"], "n": len(df), "ms_per_sample": 1000 * t["seconds"] / len(df)}
        split_rows[split] = df
        for v, s in sc.items():
            all_scores[v][split] = s
        quick = {v: round(binary_metrics(df.label.values, s, thresholds[v]).get("f1", float("nan")), 4)
                 for v, s in sc.items()}
        LOG.info("%-20s n=%-5d %.2f ms/sample  F1=%s", split, len(df), timings[split]["ms_per_sample"], quick)
    split_rows["val"] = val

    n_params = det.n_params()
    for v, per_split in all_scores.items():
        out = ensure_dir(Path(cfg["paths"]["results_dir"]) / "runs" / v / f"seed{seed}")
        for split, s in per_split.items():
            df = split_rows[split]
            pd.DataFrame({"label": df.label.values, "score": s, "source": df.source.values}).to_parquet(
                out / f"scores_{split}.parquet", index=False)
        save_json({
            "variant": v, "detector": args.detector, "type": spec["type"], "seed": seed,
            "model": spec.get("model"), "threshold": thresholds[v], "threshold_rule": rule,
            "default_threshold": det.default_thresholds.get(v), "fit_seconds": t_fit["seconds"],
            "score_timings": timings, "n_params": n_params, "device": device, "env": env_info(),
            "train_n": len(train), "val_n": len(val),
            "history": getattr(det, "history", None), "c_curve": getattr(det, "c_curve", None),
        }, out / "meta.json")

    art = ensure_dir(Path(cfg["paths"]["artifacts_dir"]) / args.detector / f"seed{seed}")
    det.save(art)
    LOG.info("done -> %s", Path(cfg["paths"]["results_dir"]) / "runs")


if __name__ == "__main__":
    main()
