"""Temperature calibration + ECE (FR-8, T7, AC-5). Pure numpy.

Usage (after training; needs a checkpoint and its split.json — run on your workstation):
    python -m src.calibrate --ckpt runs/x/best.pt --data data/ [--split val]
writes runs/x/calibration.json, which `infer.afe_from_checkpoint` picks up (meta.calibrated = true).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_GOLD = (np.sqrt(5.0) - 1.0) / 2.0


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.exp(_log_softmax(logits, temperature))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    lp = _log_softmax(logits, temperature)
    return float(-lp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0, iters: int = 100) -> float:
    """Single temperature minimising validation NLL. NLL is convex in beta = 1/T, so golden-section is exact."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    if len(logits) == 0 or len(logits) != len(labels):
        raise ValueError("need a non-empty validation set with matching logits/labels")
    a, b = 1.0 / hi, 1.0 / lo
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    f = lambda beta: nll(logits, labels, 1.0 / beta)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = f(d)
    return float(1.0 / ((a + b) / 2.0))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Top-1 confidence vs accuracy per equal-width bin."""
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(ids, minlength=n_bins).astype(np.float64)
    safe = np.maximum(count, 1.0)
    return {"edges": edges.tolist(), "count": count.tolist(),
            "confidence": (np.bincount(ids, weights=conf, minlength=n_bins) / safe).tolist(),
            "accuracy": (np.bincount(ids, weights=correct, minlength=n_bins) / safe).tolist()}


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the top-1 prediction."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    if len(labels) == 0:
        raise ValueError("ECE of an empty set is undefined")
    r = reliability_bins(probs, labels, n_bins)
    w = np.array(r["count"]) / len(labels)
    return float((w * np.abs(np.array(r["accuracy"]) - np.array(r["confidence"]))).sum())


def calibrate(val_logits: np.ndarray, val_labels: np.ndarray, n_bins: int = 15) -> dict:
    """Fit T on the validation split and report before/after ECE and NLL (on that same split)."""
    t = fit_temperature(val_logits, val_labels)
    return {"temperature": t,
            "ece_before": ece(softmax(val_logits), val_labels, n_bins), "ece_after": ece(softmax(val_logits, t), val_labels, n_bins),
            "nll_before": nll(val_logits, val_labels), "nll_after": nll(val_logits, val_labels, t),
            "n_val": int(len(val_labels)), "n_bins": n_bins}


def write_calibration(path: str | Path, result: dict) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def read_temperature(path: str | Path) -> float | None:
    p = Path(path)
    return float(json.loads(p.read_text(encoding="utf-8"))["temperature"]) if p.exists() else None


def main(argv: list[str] | None = None) -> None:
    from . import dataset, infer  # heavy imports only for the CLI
    from .config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ckpt = Path(args.ckpt)
    samples = dataset.load_samples(args.data, **dataset.load_options(cfg))
    split = dataset.load_split(ckpt.parent / "split.json", samples)
    ds = dataset.LogMelDataset([samples[i] for i in split[args.split]])
    predictor, _ = infer.load_predictor(ckpt)
    logits, y = infer.predict_dataset(predictor, ds, cfg["infer"]["batch_size"])
    result = calibrate(logits, y, cfg["eval"]["n_bins"])
    write_calibration(ckpt.parent / "calibration.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
