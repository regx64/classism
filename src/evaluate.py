"""Evaluation (FR-7, T6): top-1 / top-N, top-N coverage, per-scenario, confusion, keytype accuracy, ECE.

`compute_metrics` / `acceptance` are pure numpy. Figures are grayscale (matplotlib imported lazily).

    python -m src.evaluate --ckpt runs/smallcnn/best.pt --data data/ [--split test] [--no-calibrate]

Fits the temperature on the val split (writes <ckpt dir>/calibration.json), then measures on --split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import labels
from .calibrate import calibrate, ece, nll, reliability_bins, softmax

TARGETS = {"near_top1": 0.85, "near_top5_coverage": 0.95, "keytype_acc": 0.95, "ece": 0.05}  # NFR-3, AC-3/4/5


def _topk_hit(order: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return (order[:, :k] == y[:, None]).any(axis=1)


def compute_metrics(logits: np.ndarray, y: np.ndarray, scenarios=None, *, top_n: int = 5, temperature: float = 1.0,
                    calibrated: bool = False, n_bins: int = 15) -> dict:
    logits, y = np.asarray(logits), np.asarray(y)
    if len(y) == 0:
        raise ValueError("cannot evaluate an empty split")
    probs = softmax(logits, temperature)
    order = np.argsort(-probs, axis=-1, kind="stable")
    pred = order[:, 0]
    hit = {k: _topk_hit(order, y, k) for k in sorted({1, 3, 5, top_n})}

    def acc_block(mask):
        return {"n": int(mask.sum()), "top1": float(hit[1][mask].mean()), "top5": float(hit[5][mask].mean()),
                "top_n": float(hit[top_n][mask].mean())}

    per_scenario = {}
    if scenarios is not None:
        sc = np.asarray(scenarios)
        per_scenario = {str(s): acc_block(sc == s) for s in sorted(set(sc.tolist()))}

    kt_true, kt_pred = labels.KEYTYPE_OF_CLASS[y], labels.KEYTYPE_OF_CLASS[pred]
    per_keytype = {kt: {"n": int((kt_true == i).sum()), "recall": float((kt_pred[kt_true == i] == i).mean())}
                   for kt, i in labels.KEYTYPE_TO_IDX.items() if (kt_true == i).any()}
    cm = np.zeros((labels.num_classes(),) * 2, dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return {"n": int(len(y)), "top_n_value": top_n, "temperature": float(temperature), "calibrated": bool(calibrated),
            "top1": float(hit[1].mean()), "top3": float(hit[3].mean()), "top5": float(hit[5].mean()),
            "top_n_accuracy": float(hit[top_n].mean()),
            "top_n_coverage": float(hit[top_n].mean()),  # identical by definition; the LM's ceiling
            "keytype_acc": float((kt_true == kt_pred).mean()), "per_keytype": per_keytype,
            "per_scenario": per_scenario, "ece": ece(probs, y, n_bins), "nll": nll(logits, y, temperature),
            "reliability": reliability_bins(probs, y, n_bins), "confusion": cm.tolist(), "labels": list(labels.LABELS)}


def acceptance(m: dict) -> dict:
    """AC-3/AC-4/AC-5 verdicts; `pass: None` = could not be measured (AC-6 is reported by export_esp32)."""
    def row(target, value):
        return {"target": target, "value": value, "pass": None if value is None else bool(value >= target)}

    near = m["per_scenario"].get("near")
    ac5 = {"target": f"<= {TARGETS['ece']}", "value": m["ece"] if m["calibrated"] else None,
           "pass": bool(m["ece"] <= TARGETS["ece"]) if m["calibrated"] else None}
    return {"AC-3_near_top1": row(TARGETS["near_top1"], near and near["top1"]),
            "AC-3_near_top5_coverage": row(TARGETS["near_top5_coverage"], near and near["top5"]),
            "AC-4_keytype_acc": row(TARGETS["keytype_acc"], m["keytype_acc"]),
            "AC-5_ece": ac5}


# ---- figures (grayscale) --------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _hangul_font() -> str | None:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "UnDotum") if n in have), None)


def plot_confusion(cm, path: Path) -> None:
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    font = _hangul_font()
    ticks = list(labels.LABELS) if font else list(range(len(labels.LABELS)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=7, **({"fontname": font} if font else {}))
    ax.set_yticks(range(len(ticks)), ticks, fontsize=7, **({"fontname": font} if font else {}))
    ax.set_xlabel("predicted" + ("" if font else " (class index; see metrics.json labels)"))
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(per_scenario: dict, path: Path) -> None:
    plt = _plt()
    names = list(per_scenario)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    w = 0.27
    for j, (key, hatch, shade) in enumerate((("top1", "", "0.25"), ("top5", "//", "0.55"), ("top_n", "xx", "0.8"))):
        ax.bar(np.arange(len(names)) + (j - 1) * w, [per_scenario[n][key] for n in names], w, label=key,
               color=shade, edgecolor="black", hatch=hatch)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by scenario")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(rel: dict, path: Path, ece_value: float) -> None:
    plt = _plt()
    conf, acc, cnt = (np.array(rel[k]) for k in ("confidence", "accuracy", "count"))
    keep = cnt > 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--")
    ax.plot(conf[keep], acc[keep], color="black", marker="o", markerfacecolor="0.5")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={ece_value:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(curves: list[dict], path: Path) -> None:
    plt = _plt()
    ep = [c["epoch"] for c in curves]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, title in ((a1, "loss", "Loss"), (a2, "acc", "Accuracy")):
        ax.plot(ep, [c[f"train_{key}"] for c in curves], color="black", label="train")
        ax.plot(ep, [c[f"val_{key}"] for c in curves], color="0.5", linestyle="--", label="val")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(m: dict, out: Path, curves: list[dict] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = {**m, "acceptance": acceptance(m)}
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion(m["confusion"], out / "confusion.png")
    if m["per_scenario"]:
        plot_scenarios(m["per_scenario"], out / "per_scenario.png")
    plot_reliability(m["reliability"], out / "reliability.png", m["ece"])
    if curves:
        plot_curves(curves, out / "curves.png")
    return m


def main(argv: list[str] | None = None) -> None:
    from . import dataset, infer
    from .calibrate import write_calibration
    from .config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="default: <ckpt dir>/eval_<split>")
    ap.add_argument("--no-calibrate", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ckpt = Path(a.ckpt)
    out = Path(a.out) if a.out else ckpt.parent / f"eval_{a.split}"

    samples = dataset.load_samples(a.data, **dataset.load_options(cfg))
    split = dataset.load_split(ckpt.parent / "split.json", samples)
    predictor, _ = infer.load_predictor(ckpt)
    bs = cfg["infer"]["batch_size"]

    temperature, calibrated = 1.0, False
    if cfg["infer"]["calibrate"] and not a.no_calibrate:
        if not split["val"]:
            raise SystemExit("calibration needs a non-empty val split (or pass --no-calibrate)")
        v_logits, v_y = infer.predict_dataset(predictor, dataset.LogMelDataset([samples[i] for i in split["val"]]), bs)
        cal = calibrate(v_logits, v_y, cfg["eval"]["n_bins"])
        write_calibration(ckpt.parent / "calibration.json", cal)
        temperature, calibrated = cal["temperature"], True
        print(f"calibration: T={temperature:.3f}, val ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}")

    subset = [samples[i] for i in split[a.split]]
    logits, y = infer.predict_dataset(predictor, dataset.LogMelDataset(subset), bs)
    m = compute_metrics(logits, y, [s.scenario for s in subset], top_n=cfg["infer"]["top_n"],
                        temperature=temperature, calibrated=calibrated, n_bins=cfg["eval"]["n_bins"])
    curves_path = ckpt.parent / "curves.json"
    m = write_report(m, out, json.loads(curves_path.read_text()) if curves_path.exists() else None)
    print(f"[{a.split}] n={m['n']} top1={m['top1']:.3f} top5={m['top5']:.3f} keytype={m['keytype_acc']:.3f} ece={m['ece']:.4f}")
    for k, v in m["acceptance"].items():
        print(f"  {k:26s} {v['pass']}  (value={v['value']}, target={v['target']})")
    print("report ->", out)


if __name__ == "__main__":
    main()
