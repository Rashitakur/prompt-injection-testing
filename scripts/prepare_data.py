#!/usr/bin/env python
"""Build deduplicated, leakage-free train/val/test splits + OOD and stress sets."""
import argparse
import json

import _bootstrap  # noqa: F401
from pidetect.data import build_dataset
from pidetect.utils import load_config, set_seed, setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = cfg["seeds"][0]
    set_seed(seed)
    man = build_dataset(cfg, seed)
    print(json.dumps(man["splits"], indent=1))


if __name__ == "__main__":
    main()
