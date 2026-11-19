#!/usr/bin/env python
"""Pre-fetch every dataset and model into $HF_HOME.

Run this on a node WITH internet (usually the login node). Compute jobs then run
with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1.

    python scripts/download_assets.py --config configs/default.yaml
"""
import argparse
import logging

import _bootstrap  # noqa: F401
from pidetect.utils import load_config, setup_logging

LOG = logging.getLogger("download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    failures = []

    if not args.skip_data:
        from datasets import load_dataset

        for name, spec in cfg["data"]["sources"].items():
            if "hf_id" not in spec or not spec.get("enabled", True):
                continue
            for split in spec["splits"]:
                try:
                    ds = load_dataset(spec["hf_id"], spec.get("hf_config"), split=split)
                    LOG.info("dataset %-45s %-16s rows=%-6d cols=%s", spec["hf_id"], split, len(ds), ds.column_names)
                except Exception as e:  # noqa: BLE001
                    LOG.error("dataset %s[%s] FAILED: %s", spec["hf_id"], split, e)
                    failures.append(f"{spec['hf_id']}[{split}]")

    if not args.skip_models:
        from huggingface_hub import snapshot_download

        for name, spec in cfg["detectors"].items():
            if "model" not in spec or not spec.get("enabled", True):
                continue
            try:
                p = snapshot_download(
                    spec["model"],
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.tiktoken", "tokenizer*"],
                    ignore_patterns=["onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.h5", "*.msgpack", "*.ot",
                                     "training_args.bin"])
                LOG.info("model   %-45s -> %s", spec["model"], p)
            except Exception as e:  # noqa: BLE001
                LOG.error("model %s FAILED: %s", spec["model"], e)
                failures.append(spec["model"])

    if failures:
        LOG.warning("Failed assets (optional sources are skipped automatically later): %s", failures)
    else:
        LOG.info("All assets cached.")


if __name__ == "__main__":
    main()
