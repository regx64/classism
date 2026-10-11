"""Synthetic log-mel dataset in the §3.1/§7 layout — for smoke-testing the pipeline ONLY.

Each class gets a fixed smooth spectro-temporal prototype; clips are prototype + noise (scenario-dependent),
per-clip z-scored like the real Mel output. It says nothing about real keyboard acoustics, so accuracy
figures obtained on it must never be reported as AFE results.

    python -m src.synth --out data/synth --participants 3 --sessions 2 --per-session 190
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import labels
from .melio import MEL_SPEC_VERSION, REGIME

SCENARIOS = {"near": 0.6, "far": 1.2, "noise": 1.8}  # noise sigma
_SHIFTED = set("ㄲㄸㅃㅆㅉㅒㅖ")


def _prototype(label: int, n_mels: int, frames: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + label)
    f, t = np.mgrid[0:n_mels, 0:frames]
    proto = np.zeros((n_mels, frames))
    for _ in range(3):
        cf, ct = rng.uniform(0, n_mels), rng.uniform(0, frames)
        sf, st = rng.uniform(3, 9), rng.uniform(2, 6)
        proto += rng.uniform(0.8, 1.6) * np.exp(-(((f - cf) / sf) ** 2 + ((t - ct) / st) ** 2) / 2)
    return proto


def make_clip(rng: np.random.Generator, label: int, sigma: float, tilt: float) -> np.ndarray:
    n, fr = REGIME.n_mels, REGIME.frames
    x = _prototype(label, n, fr) * 2.0 + tilt * np.linspace(-1, 1, n)[:, None] + rng.normal(0, sigma, (n, fr))
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32)


def generate(out: str | Path, participants: int = 3, sessions_per_scenario: int = 1, per_session: int = 190,
             seed: int = 0) -> Path:
    out = Path(out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "mel").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_cls = labels.num_classes()
    for p in range(participants):
        tilt = rng.uniform(-0.5, 0.5)  # participant-specific spectral tilt
        for scen, sigma in SCENARIOS.items():
            for s in range(sessions_per_scenario):
                sid = f"p{p + 1}_{scen}_s{s}"
                # every class appears at least once (in random order), the rest sampled with specials favoured
                cover = rng.permutation(n_cls).tolist()
                extra = rng.choice(n_cls, size=max(per_session - n_cls, 0),
                                   p=np.where(np.arange(n_cls) >= len(labels.JAMO), 2.0, 1.0) / (len(labels.JAMO) + 2.0 * len(labels.SPECIAL)))
                seq = (cover + extra.tolist())[:per_session]
                t = 1.0
                with open(out / "labels" / f"{sid}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["onset_s", "jamo", "keytype", "shift", "scenario", "participant", "clip_id"])
                    for i, lab in enumerate(seq):
                        sym = labels.IDX_TO_SYMBOL[lab]
                        kt = labels.keytype_of(sym)
                        t += float(rng.uniform(0.12, 0.6))
                        cid = f"{sid}_{i:05d}"
                        np.save(out / "mel" / f"{cid}.npy", make_clip(rng, lab, sigma, tilt))
                        w.writerow([f"{t:.3f}", sym if kt == "normal" else "", kt,
                                    int(sym in _SHIFTED or kt == "shift"), scen, f"p{p + 1}", cid])
    (out / "meta.json").write_text(json.dumps({"synthetic": True, "param_hash": REGIME.param_hash(), "seed": seed}), encoding="utf-8")
    # the Mel component's header (§3.1): lets load_samples verify the regime (NFR-2)
    (out / "mel" / "_meta.json").write_text(json.dumps({
        "mel_spec_version": MEL_SPEC_VERSION, "param_hash": REGIME.param_hash(), "sample_rate": REGIME.sample_rate,
        "n_mels": REGIME.n_mels, "frames": REGIME.frames, "synthetic": True}), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--participants", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=1, help="sessions per participant per scenario")
    ap.add_argument("--per-session", type=int, default=190)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print("wrote", generate(a.out, a.participants, a.sessions, a.per_session, a.seed))


if __name__ == "__main__":
    main()
