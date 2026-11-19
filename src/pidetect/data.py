"""Dataset assembly: HF sources -> deduplicated, leakage-free parquet splits.

Output (in <data_dir>/processed):
    train.parquet, val.parquet, test_id.parquet, <ood sets>.parquet, manifest.json
Each parquet has columns: text (str), label (int 0/1), source (str).
"""
from __future__ import annotations

import base64
import logging
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_json, load_json, text_hash

LOG = logging.getLogger(__name__)
ID_SPLITS = ("train", "val", "test_id")


# --------------------------------------------------------------------------- loading
def _pick_col(columns, candidates, what: str, source: str) -> str:
    if isinstance(candidates, str):
        candidates = [candidates]
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"[{source}] none of {what} columns {candidates} found in {list(columns)}")


def _load_split_df(spec: dict, hf_split: str) -> pd.DataFrame:
    if "local_files" in spec:  # used by smoke tests / offline mirrors
        path = spec["local_files"][hf_split]
        return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    from datasets import load_dataset

    ds = load_dataset(spec["hf_id"], spec.get("hf_config"), split=hf_split)
    return ds.to_pandas()


def load_source(name: str, spec: dict, seed: int) -> dict[str, pd.DataFrame]:
    """Return {target_split: DataFrame[text,label,source]} for one source."""
    out: dict[str, list[pd.DataFrame]] = {}
    for hf_split, target in spec["splits"].items():
        raw = _load_split_df(spec, hf_split)
        tcol = _pick_col(raw.columns, spec["text_cols"], "text", name)
        df = pd.DataFrame({"text": raw[tcol].astype(str)})
        if "fixed_label" in spec:
            df["label"] = int(spec["fixed_label"])
        else:
            lcol = _pick_col(raw.columns, spec["label_col"], "label", name)
            lab = raw[lcol]
            if spec.get("label_map"):
                lab = lab.astype(str).str.strip().str.lower().map(
                    {str(k).lower(): v for k, v in spec["label_map"].items()})
            df["label"] = pd.to_numeric(lab, errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int).clip(0, 1)
        df["source"] = name
        if target == "ood":
            target = spec.get("test_set", f"ood_{name}")
        out.setdefault(target, []).append(df)
    res = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    if spec.get("max_n"):
        for k, df in res.items():
            if len(df) > spec["max_n"]:
                res[k] = _stratified_sample(df, spec["max_n"], seed)
    return res


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- synthetic stress sets
_NOISE_ALPHABET = string.ascii_letters + string.digits + string.punctuation


def _random_suffix(rng: random.Random, n_tokens: int) -> str:
    """GCG-like gibberish: short random strings mixed with punctuation and rare words."""
    toks = []
    for _ in range(n_tokens):
        k = rng.randint(1, 6)
        toks.append("".join(rng.choice(_NOISE_ALPHABET) for _ in range(k)))
    return " ".join(toks)


def _benign_noise(rng: random.Random) -> str:
    kind = rng.choice(["b64", "hex", "url", "code", "uuid"])
    if kind == "b64":
        return "Attachment: " + base64.b64encode(rng.randbytes(rng.randint(24, 64))).decode()
    if kind == "hex":
        return "Checksum: " + rng.randbytes(rng.randint(16, 32)).hex()
    if kind == "url":
        q = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(20, 50)))
        return f"See https://example.com/r/{q}?utm_src={rng.randint(1000, 9999)}&sig={rng.randbytes(8).hex()}"
    if kind == "code":
        v = rng.choice(["x", "tmp", "buf", "acc"])
        return f"```py\n{v}=[i**2 for i in range({rng.randint(3, 99)}) if i%{rng.randint(2, 7)}]\nprint({v}[-1])\n```"
    return "Ref: " + "-".join(rng.randbytes(n).hex() for n in (4, 2, 2, 2, 6))


def make_synthetic(test_id: pd.DataFrame, cfg: dict, seed: int) -> dict[str, pd.DataFrame]:
    syn = cfg["data"].get("synthetic", {})
    rng = random.Random(seed)
    out = {}
    pos = test_id[test_id.label == 1]
    neg = test_id[test_id.label == 0]
    if syn.get("adv_suffix", {}).get("enabled") and len(pos) and len(neg):
        s = syn["adv_suffix"]
        n = min(s.get("max_n", 600) // 2, len(pos), len(neg))
        p = pos.sample(n=n, random_state=seed).copy()
        p["text"] = [t + " " + _random_suffix(rng, s.get("n_suffix_tokens", 20)) for t in p.text]
        q = neg.sample(n=n, random_state=seed).copy()
        df = pd.concat([p, q]).sample(frac=1.0, random_state=seed)
        df["source"] = "synthetic_adv_suffix"
        out["stress_adv_suffix"] = df.reset_index(drop=True)
    if syn.get("noisy_benign", {}).get("enabled") and len(neg):
        n = min(syn["noisy_benign"].get("max_n", 600), len(neg))
        q = neg.sample(n=n, random_state=seed + 1).copy()
        q["text"] = [t + "\n" + _benign_noise(rng) for t in q.text]
        q["source"] = "synthetic_noisy_benign"
        out["benign_noisy"] = q.reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- build
def build_dataset(cfg: dict, seed: int) -> dict:
    dcfg = cfg["data"]
    pools: dict[str, list[pd.DataFrame]] = {}
    status = {}
    for name, spec in dcfg["sources"].items():
        if not spec.get("enabled", True):
            continue
        try:
            parts = load_source(name, spec, seed)
            for k, df in parts.items():
                pools.setdefault(k, []).append(df)
            status[name] = {k: int(len(v)) for k, v in parts.items()}
            LOG.info("loaded %-10s %s", name, status[name])
        except Exception as e:  # noqa: BLE001
            if spec.get("required"):
                raise
            LOG.warning("SKIPPING source %s: %s: %s", name, type(e).__name__, e)
            status[name] = {"error": f"{type(e).__name__}: {e}"}

    splits = {k: pd.concat(v, ignore_index=True) for k, v in pools.items()}
    for k, df in splits.items():
        df["text"] = df["text"].str.slice(0, dcfg["max_chars"])
        splits[k] = df[df["text"].str.strip().str.len() >= dcfg["min_chars"]].reset_index(drop=True)

    # in-distribution: carve validation from train (stratified by source x label)
    from sklearn.model_selection import train_test_split

    train = splits.pop("train")
    if dcfg.get("dedup", True):
        train = _dedup(train, "train")
    strat = train["source"] + "_" + train["label"].astype(str)
    train, val = train_test_split(train, test_size=dcfg["val_frac"], random_state=seed, stratify=strat)
    if dcfg.get("max_train") and len(train) > dcfg["max_train"]:
        train = _stratified_sample(train, dcfg["max_train"], seed)
    splits["train"], splits["val"] = train.reset_index(drop=True), val.reset_index(drop=True)

    leak = {}
    if dcfg.get("dedup", True):
        seen = set(train["text"].map(text_hash))
        for k in list(splits):
            if k == "train":
                continue
            df = _dedup(splits[k], k)
            h = df["text"].map(text_hash)
            leak[k] = int(h.isin(seen).sum())
            splits[k] = df[~h.isin(seen)].reset_index(drop=True)
            if k == "val":
                seen |= set(h)  # also keep val disjoint from test sets

    splits.update(make_synthetic(splits["test_id"], cfg, seed))

    out_dir = ensure_dir(Path(cfg["paths"]["data_dir"]) / "processed")
    manifest = {"sources": status, "leakage_removed": leak, "splits": {}}
    for k, df in splits.items():
        df = df[["text", "label", "source"]]
        df.to_parquet(out_dir / f"{k}.parquet", index=False)
        manifest["splits"][k] = {
            "n": int(len(df)), "n_pos": int(df.label.sum()), "n_neg": int((df.label == 0).sum()),
            "sources": df.source.value_counts().to_dict(),
            "mean_chars": float(df.text.str.len().mean()) if len(df) else 0.0,
        }
    save_json(manifest, out_dir / "manifest.json")
    LOG.info("wrote %d splits to %s", len(splits), out_dir)
    return manifest


def _dedup(df: pd.DataFrame, name: str) -> pd.DataFrame:
    h = df["text"].map(text_hash)
    # drop exact duplicates; if duplicates disagree on label, drop all copies (label noise)
    g = pd.DataFrame({"h": h, "label": df["label"]}).groupby("h")["label"].nunique()
    conflicted = set(g[g > 1].index)
    keep = ~h.duplicated() & ~h.isin(conflicted)
    n_drop = int((~keep).sum())
    if n_drop:
        LOG.info("dedup %-22s dropped %d rows (%d conflicting-label groups)", name, n_drop, len(conflicted))
    return df[keep.values].reset_index(drop=True)


# --------------------------------------------------------------------------- access
def processed_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "processed"


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg) / f"{name}.parquet")


def eval_split_names(cfg: dict) -> list[str]:
    man = load_json(processed_dir(cfg) / "manifest.json")
    names = [k for k in man["splits"] if k not in ("train", "val")]
    return sorted(names, key=lambda s: (s != "test_id", s))
