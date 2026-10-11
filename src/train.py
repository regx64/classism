"""Training on log-mel (FR-6, NFR-4, T4). Run on a workstation (needs PyTorch; not meant for the Codespace).

    python -m src.train --data data/ --out runs/smallcnn [--epochs 30] [--split session|participant|random]

Writes to --out: best.pt, last.pt, split.json (fixed split), curves.json, model_summary.json, config_used.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import dataset, labels
from .config import load_config
from .melio import REGIME
from .model import build_model, complexity


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_epoch(model, loader, loss_fn, device, opt=None) -> tuple[float, float]:
    train = opt is not None
    model.train(train)
    tot_loss = tot_ok = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(y)
            tot_ok += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return (tot_loss / n, tot_ok / n) if n else (float("nan"), float("nan"))


def train(cfg: dict, data_root: str | Path, out_dir: str | Path, *, epochs: int | None = None,
          max_samples: int | None = None, device: str = "cpu") -> dict:
    tc = cfg["train"]
    epochs = epochs or tc["epochs"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(tc["seed"])

    samples = dataset.load_samples(data_root, **dataset.load_options(cfg))
    split = dataset.make_split(samples, tc["split"], tc["seed"], tc["val_frac"], tc["test_frac"])
    dataset.save_split(out / "split.json", samples, split, tc["split"], tc["seed"])  # full split, saved before truncation
    if max_samples:  # smoke runs: truncate each subset, keep the split itself intact
        split = {k: v[:max_samples] for k, v in split.items()}
    subsets = {k: [samples[i] for i in v] for k, v in split.items()}
    print({k: len(v) for k, v in subsets.items()}, "clips; split mode:", tc["split"])
    if not subsets["train"]:
        raise ValueError("empty training split")

    gen = torch.Generator().manual_seed(tc["seed"])
    loaders = {k: DataLoader(dataset.LogMelDataset(v), batch_size=tc["batch_size"], shuffle=(k == "train"), generator=gen if k == "train" else None)
               for k, v in subsets.items() if v}
    model_kwargs = {"dropout": cfg["model"]["dropout"]}
    model = build_model(cfg["model"]["name"], labels.num_classes(), **model_kwargs).to(device)
    summary = {"model": cfg["model"]["name"], **complexity(model.cpu()), "device": device}
    model.to(device)
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    weight = torch.tensor(dataset.class_weights(dataset.class_counts(subsets["train"]))) if tc["class_weighted"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None, label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def checkpoint(epoch: int, val_acc: float) -> dict:
        return {"model_name": cfg["model"]["name"], "model_kwargs": model_kwargs, "num_classes": labels.num_classes(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "labels": list(labels.LABELS), "param_hash": REGIME.param_hash(), "seed": tc["seed"], "epoch": epoch,
                "val_acc": val_acc, "model_id": f"{cfg['model']['name']}-fp32-v1"}

    curves, best = [], (-1.0, float("inf"))
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], loss_fn, device, opt)
        va_loss, va_acc = _run_epoch(model, loaders["val"], loss_fn, device) if "val" in loaders else (tr_loss, tr_acc)
        sched.step()
        curves.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:3d}  train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
        if (va_acc, -va_loss) > (best[0], -best[1]):
            best = (va_acc, va_loss)
            torch.save(checkpoint(epoch, va_acc), out / "best.pt")
        (out / "curves.json").write_text(json.dumps(curves, indent=1), encoding="utf-8")
    torch.save(checkpoint(epochs, curves[-1]["val_acc"]), out / "last.pt")
    return {"out": str(out), "best_val_acc": best[0], "epochs": epochs, **summary}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="cap clips per subset (smoke runs)")
    ap.add_argument("--split", choices=["session", "participant", "random"], default=None)
    ap.add_argument("--model", choices=["smallcnn", "coatnetlite"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.split:
        cfg["train"]["split"] = a.split
    if a.model:
        cfg["model"]["name"] = a.model
    if a.seed is not None:
        cfg["train"]["seed"] = a.seed
    print(json.dumps(train(cfg, a.data, a.out, epochs=a.epochs, max_samples=a.max_samples, device=a.device), indent=2))


if __name__ == "__main__":
    main()
