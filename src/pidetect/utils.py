"""Shared utilities: config, seeding, devices, batching, IO."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import yaml

LOG = logging.getLogger("pidetect")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- config
def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _expand(cfg)
    root = Path(cfg.get("paths", {}).get("root", ".")).expanduser()
    if not root.is_absolute():  # relative paths are resolved against the repo root (configs/..)
        root = (path.resolve().parent.parent / root).resolve()
    cfg.setdefault("paths", {})["root"] = str(root)
    for key in ("data_dir", "results_dir", "artifacts_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else root / p)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def enabled_detectors(cfg: dict) -> list[str]:
    return [k for k, v in cfg["detectors"].items() if v.get("enabled", True)]


# --------------------------------------------------------------------------- seeding / device
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto") -> str:
    import torch

    if prefer == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if prefer == "cuda":
        LOG.warning("CUDA requested but unavailable; falling back to CPU")
    return "cpu"


def torch_dtype(name: str | None, device: str):
    import torch

    if device == "cpu":
        return torch.float32
    name = (name or "auto").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def dtype_kwargs(dtype) -> dict:
    """transformers>=4.56 uses `dtype=`, older versions `torch_dtype=`."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def env_info() -> dict:
    info = {"python": platform.python_version(), "host": platform.node(), "machine": platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        info["cpu_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    info["cpu_model"] = _cpu_model()
    return info


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- text
_WS = re.compile(r"\s+")


def normalize_text(t: str) -> str:
    return _WS.sub(" ", str(t)).strip().lower()


def text_hash(t: str) -> str:
    return hashlib.sha1(normalize_text(t).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- batching
def length_sorted_batches(texts: Sequence[str], batch_size: int) -> Iterator[list[int]]:
    """Yield index batches sorted by length (minimises padding); caller restores order."""
    order = np.argsort([len(t) for t in texts], kind="stable")[::-1]
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size].tolist()


@contextmanager
def timer() -> Iterator[dict]:
    rec: dict = {}
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = time.perf_counter() - t0


# --------------------------------------------------------------------------- io
def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def chunked(it: Iterable, n: int) -> Iterator[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
