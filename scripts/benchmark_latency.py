#!/usr/bin/env python
"""Controlled latency benchmark for fitted detectors.

Per detector and device we measure, on a fixed random sample of test_id texts:
  * batch size 1  -> per-request latency (mean / p50 / p95 / p99, ms)
  * batch size 32 -> throughput (samples / s) and amortised ms/sample
Timings include tokenisation and pre/post-processing (end-to-end `score` call),
are preceded by warm-up calls, and are synchronised on CUDA.

    python scripts/benchmark_latency.py --detector distilbert --device cuda
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from pidetect.data import load_split
from pidetect.detectors import build_detector
from pidetect.utils import ensure_dir, env_info, get_device, load_config, save_json, setup_logging, sync

LOG = logging.getLogger("latency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    lcfg = cfg["latency"]
    import torch

    device = get_device(args.device)
    if device != args.device:
        LOG.error("requested %s but got %s; aborting so numbers are not mislabelled", args.device, device)
        return
    if device == "cpu":
        torch.set_num_threads(int(lcfg.get("cpu_threads", 8)))
        if args.detector in lcfg.get("cpu_skip", []):
            LOG.info("skipping %s on CPU (cpu_skip)", args.detector)
            return

    seed = cfg["seeds"][0]
    det = build_detector(args.detector, cfg, seed, device)
    art = Path(cfg["paths"]["artifacts_dir"]) / args.detector / f"seed{seed}"
    det.load(art)

    n = args.n or lcfg["n_samples"]
    slow = device == "cpu" and cfg["detectors"][args.detector]["type"] == "llm_judge"
    if slow:
        n = min(n, lcfg.get("cpu_n_samples_slow", 30))
    pool = load_split(cfg, "test_id").text.tolist()
    rng = np.random.default_rng(seed)
    texts = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    LOG.info("%s on %s: %d texts (mean %.0f chars)", args.detector, device, len(texts), np.mean([len(t) for t in texts]))

    # warm-up (CUDA kernels, allocator, tokenizer caches)
    det.set_batch_size(1)
    for t in texts[: lcfg.get("warmup", 10)]:
        det.score([t])
    sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    res = {"detector": args.detector, "device": device, "n": len(texts), "env": env_info(),
           "n_params": det.n_params(), "batch": {}}
    for bs in lcfg["batch_sizes"]:
        det.set_batch_size(bs)
        per_call = []
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            sync(device)
            t0 = time.perf_counter()
            det.score(chunk)
            sync(device)
            per_call.append(((time.perf_counter() - t0) * 1000.0, len(chunk)))
        ms = np.array([m for m, _ in per_call])
        sizes = np.array([k for _, k in per_call])
        total_s = ms.sum() / 1000.0
        res["batch"][str(bs)] = {
            "calls": len(ms),
            "ms_per_call_mean": float(ms.mean()), "ms_per_call_p50": float(np.percentile(ms, 50)),
            "ms_per_call_p95": float(np.percentile(ms, 95)), "ms_per_call_p99": float(np.percentile(ms, 99)),
            "ms_per_sample": float(ms.sum() / sizes.sum()), "throughput_per_s": float(sizes.sum() / total_s),
        }
        LOG.info("bs=%-3d %s", bs, {k: round(v, 2) for k, v in res["batch"][str(bs)].items()})
    if device == "cuda":
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    save_json(res, ensure_dir(Path(cfg["paths"]["results_dir"]) / "latency") / f"{args.detector}__{device}.json")


if __name__ == "__main__":
    main()
