"""Configuration (spec §9): YAML over built-in defaults, validated against the frozen regime."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from . import labels
from .melio import REGIME

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "afe.yaml"

DEFAULTS: dict = {
    "labels": {"num_classes": 38},
    "input": {"n_mels": 64, "frames": 48, "sample_rate": 48000, "normalization": "per_clip_zscore"},
    "model": {"name": "smallcnn", "dropout": 0.2},
    "infer": {"top_n": 5, "calibrate": True, "temperature": 1.0, "batch_size": 64},
    "export": {"runtime": "tflite_int8", "calib_clips": 300, "max_top1_drop_pp": 2.0, "max_size_bytes": 1_000_000},
    "train": {"split": "session", "seed": 1234, "epochs": 30, "batch_size": 64, "lr": 0.002,
              "weight_decay": 1e-4, "label_smoothing": 0.0, "class_weighted": False,
              "val_frac": 0.1, "test_frac": 0.2},
    "data": {"min_per_jamo": 25, "min_per_special": 25, "sessions": None, "on_missing_mel": "error",
             "require_param_hash": False},
    "eval": {"n_bins": 15},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate(cfg: dict) -> dict:
    if cfg["labels"]["num_classes"] != labels.num_classes():
        raise ValueError(f"labels.num_classes {cfg['labels']['num_classes']} != {labels.num_classes()}")
    i = cfg["input"]
    for key in ("n_mels", "frames", "sample_rate", "normalization"):
        if i[key] != getattr(REGIME, key):
            raise ValueError(f"input.{key}={i[key]!r} conflicts with the frozen regime value {getattr(REGIME, key)!r}")
    if not 1 <= cfg["infer"]["top_n"] <= labels.num_classes():
        raise ValueError("infer.top_n must be in [1, 38]")
    if cfg["infer"]["temperature"] <= 0:
        raise ValueError("infer.temperature must be > 0")
    d = cfg["data"]
    if d["on_missing_mel"] not in ("error", "drop"):
        raise ValueError("data.on_missing_mel must be error | drop")
    if d["sessions"] is not None and not (isinstance(d["sessions"], list) and all(isinstance(x, str) for x in d["sessions"])):
        raise ValueError("data.sessions must be null or a list of session ids")
    if cfg["train"]["split"] not in ("session", "participant", "random"):
        raise ValueError("train.split must be session | participant | random")
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    """Load `path` (default configs/afe.yaml if present) merged over DEFAULTS."""
    p = Path(path) if path else DEFAULT_PATH
    over = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            over = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(p)
    return validate(_merge(DEFAULTS, over))
