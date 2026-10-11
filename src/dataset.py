"""Training data: label CSVs + pre-computed log-mel tensors, session/participant splits (FR-6, T4, T8).

Layout (option A of §3.1, "pre-computed tensors keyed by clip_id"):
    <root>/labels/<session_id>.csv   columns: onset_s, jamo, keytype, shift, scenario, participant [, clip_id]
    <root>/mel/<clip_id>.npy         float32 (64,48) or (1,64,48), final z-scored log-mel
    <root>/mel/_meta.json            the Mel component's header: mel_spec_version, param_hash, ...
    <root>/mel_packed/<session_id>.npz   optional session shard, used instead of mel/*.npy for that session:
                                     logmel (N,64,48) float32, clip_id (N,) str, param_hash, mel_spec_version

`session_id` is the CSV file stem. If the CSV has no `clip_id` column, clip_id is `<session_id>_<row:05d>`
(row = 0-based data row), which is the name the Mel component is expected to write. Pure numpy/stdlib.

The log-mel's `param_hash` (shard header, else mel/_meta.json) must equal the frozen regime's (NFR-2): a
different hash raises RegimeError; data that carries no hash only warns unless `require_hash` is set.
"""
from __future__ import annotations

import csv
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import labels
from .melio import REGIME, Regime, RegimeError, check_logmel, load_clip

REQUIRED_COLUMNS = ("onset_s", "jamo", "keytype", "shift", "scenario", "participant")
MISSING_POLICIES = ("error", "drop")  # a CSV row without a log-mel: raise, or leave the row out (counted)


@dataclass(frozen=True)
class Sample:
    clip_id: str
    session_id: str
    participant: str
    scenario: str
    keytype: str
    symbol: str
    label: int
    onset_s: float
    mel_path: Path
    shard: Path | None = None  # set when the clip is read from mel_packed/<session_id>.npz


# ---- session shards ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _shard(path: str, mtime_ns: int) -> tuple[dict[str, int], np.ndarray, str | None]:
    with np.load(path, allow_pickle=False) as z:
        logmel, ids = z["logmel"], z["clip_id"].tolist()
        h = str(z["param_hash"]) if "param_hash" in z.files else None
    if logmel.ndim != 3 or len(logmel) != len(ids):
        raise RegimeError(f"shard {path}: logmel {logmel.shape} / clip_id ({len(ids)},) are inconsistent")
    return {c: i for i, c in enumerate(ids)}, logmel, h


def read_shard(path: str | Path) -> tuple[dict[str, int], np.ndarray, str | None]:
    """(clip_id -> row, logmel (N,n_mels,frames), param_hash or None); cached until the file changes."""
    p = Path(path)
    return _shard(str(p), p.stat().st_mtime_ns)


def shard_path(root: Path, session_id: str) -> Path | None:
    p = root / "mel_packed" / f"{session_id}.npz"
    return p if p.exists() else None


# ---- param_hash (NFR-2) -----------------------------------------------------------------------
def mel_param_hashes(root: str | Path, session_ids: list[str]) -> dict[str, str | None]:
    """The param_hash each session's log-mel was extracted under: its shard header, else mel/_meta.json."""
    root = Path(root)
    meta_path = root / "mel" / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get("param_hash") if meta_path.exists() else None
    out = {}
    for sid in session_ids:
        sp = shard_path(root, sid)
        out[sid] = read_shard(sp)[2] if sp else meta
    return out


def verify_param_hash(root: str | Path, session_ids: list[str], regime: Regime = REGIME, *,
                      require: bool = False) -> str:
    """Raise RegimeError if any session's log-mel was extracted under another regime. Returns the expected hash."""
    want = regime.param_hash()
    got = mel_param_hashes(root, session_ids)
    bad = {sid: h for sid, h in got.items() if h is not None and h != want}
    if bad:
        raise RegimeError(f"log-mel of {len(bad)} session(s) (e.g. {sorted(bad)[:3]}) was extracted under param_hash "
                          f"{sorted(set(bad.values()))}, the frozen regime is {want} (NFR-2); re-extract it")
    unknown = sorted(sid for sid, h in got.items() if h is None)
    if unknown:
        msg = (f"{len(unknown)} session(s) carry no param_hash (no mel/_meta.json, no shard header), "
               f"e.g. {unknown[:3]}: the regime cannot be verified (NFR-2)")
        if require:
            raise RegimeError(msg)
        warnings.warn(msg, stacklevel=3)
    return want


# ---- loading ----------------------------------------------------------------------------------
def load_dataset(root: str | Path, *, check_files: bool = True, sessions: list[str] | None = None,
                 on_missing: str = "error", verify_hash: bool = True,
                 require_hash: bool = False) -> tuple[list[Sample], dict]:
    """Samples of every label CSV (or only `sessions`) + info {param_hash, sessions, shard_sessions, dropped_no_mel}.

    on_missing="error" raises when a row has no log-mel (when check_files); "drop" leaves such rows out and
    counts them (the Mel component reports why it skipped them)."""
    if on_missing not in MISSING_POLICIES:
        raise ValueError(f"on_missing must be one of {MISSING_POLICIES}, got {on_missing!r}")
    root = Path(root)
    csvs = sorted((root / "labels").glob("*.csv"))
    if sessions is not None:
        unknown = sorted(set(sessions) - {p.stem for p in csvs})
        if unknown:
            raise FileNotFoundError(f"no label CSV for session(s) {unknown} under {root / 'labels'}")
        csvs = [p for p in csvs if p.stem in set(sessions)]
    if not csvs:
        raise FileNotFoundError(f"no label CSVs under {root / 'labels'}")
    out: list[Sample] = []
    missing: list[str] = []
    dropped = 0
    shard_sessions = []
    for path in csvs:
        sid = path.stem
        shard = shard_path(root, sid)
        in_shard = read_shard(shard)[0] if shard else None
        if shard:
            shard_sessions.append(sid)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            absent = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if absent:
                raise ValueError(f"{path.name}: missing columns {absent}")
            for i, row in enumerate(reader):
                symbol = labels.symbol_of(row["keytype"].strip(), (row["jamo"] or "").strip())
                cid = (row.get("clip_id") or "").strip() or f"{sid}_{i:05d}"
                mp = root / "mel" / f"{cid}.npy"
                if check_files or on_missing == "drop":
                    present = cid in in_shard if shard else mp.exists()
                    if not present:
                        if on_missing == "drop":
                            dropped += 1
                            continue
                        missing.append(f"{shard}:{cid}" if shard else str(mp))
                out.append(Sample(cid, sid, row["participant"].strip(), row["scenario"].strip(), row["keytype"].strip(),
                                  symbol, labels.label_index(symbol), float(row["onset_s"]), mp, shard))
    if missing:
        raise FileNotFoundError(f"{len(missing)} log-mel files missing, e.g. {missing[:3]}")
    used = [p.stem for p in csvs]
    want = verify_param_hash(root, used, require=require_hash) if verify_hash else REGIME.param_hash()
    return out, {"param_hash": want, "sessions": used, "shard_sessions": shard_sessions, "dropped_no_mel": dropped}


def load_samples(root: str | Path, *, check_files: bool = True, **options) -> list[Sample]:
    """`load_dataset` without the info dict. `options`: sessions, on_missing, verify_hash, require_hash."""
    return load_dataset(root, check_files=check_files, **options)[0]


def load_options(cfg: dict) -> dict:
    """load_samples options from the config's `data` section."""
    d = cfg["data"]
    return {"sessions": d["sessions"], "on_missing": d["on_missing_mel"], "require_hash": d["require_param_hash"]}


# ---- splitting ------------------------------------------------------------------------------
def _group_split(keys: list[str], val_frac: float, test_frac: float, seed: int, what: str) -> dict[str, list[int]]:
    groups = sorted(set(keys))
    n_test = max(1, round(len(groups) * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(len(groups) * val_frac)) if val_frac > 0 else 0
    if len(groups) - n_test - n_val < 1:
        raise ValueError(f"a {what} split into train/val/test needs at least {n_test + n_val + 1} {what}s, found {len(groups)}")
    perm = np.random.default_rng(seed).permutation(len(groups))
    which = {}
    for rank, g in enumerate(perm):
        which[groups[g]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, k in enumerate(keys):
        split[which[k]].append(i)
    return split


def make_split(samples: list[Sample], mode: str = "session", seed: int = 1234, val_frac: float = 0.1,
               test_frac: float = 0.2) -> dict[str, list[int]]:
    """Index lists {train,val,test}. `session`: a recording session never spans two subsets (FR-6);
    `participant`: cross-user generalisation; `random`: per clip (optimistic — sanity checks only)."""
    if mode == "session":
        return _group_split([s.session_id for s in samples], val_frac, test_frac, seed, "session")
    if mode == "participant":
        return _group_split([s.participant for s in samples], val_frac, test_frac, seed, "participant")
    if mode == "random":
        perm = np.random.default_rng(seed).permutation(len(samples))
        n_test, n_val = int(round(len(samples) * test_frac)), int(round(len(samples) * val_frac))
        return {"test": sorted(perm[:n_test].tolist()), "val": sorted(perm[n_test:n_test + n_val].tolist()),
                "train": sorted(perm[n_test + n_val:].tolist())}
    raise ValueError(f"unknown split mode {mode!r}")


def save_split(path: str | Path, samples: list[Sample], split: dict[str, list[int]], mode: str, seed: int) -> None:
    doc = {"mode": mode, "seed": seed, **{k: [samples[i].clip_id for i in v] for k, v in split.items()}}
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_split(path: str | Path, samples: list[Sample]) -> dict[str, list[int]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    pos = {s.clip_id: i for i, s in enumerate(samples)}
    try:
        return {k: [pos[c] for c in doc[k]] for k in ("train", "val", "test")}
    except KeyError as e:
        raise ValueError(f"split file {path} refers to clip {e} that is not in the dataset") from None


# ---- dataset --------------------------------------------------------------------------------
class LogMelDataset:
    """Map-style dataset (works with torch DataLoader without importing torch). Yields ((1,64,48) f32, int)."""

    def __init__(self, samples: list[Sample], *, cache: bool = True):
        self.samples, self._cache = samples, ({} if cache else None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        s = self.samples[i]
        x = self._cache.get(i) if self._cache is not None else None
        if x is None:
            if s.shard is not None:
                index, logmel, _ = read_shard(s.shard)
                x = check_logmel(logmel[index[s.clip_id]], REGIME)[None]
            else:
                x = load_clip(s.mel_path, REGIME)[None]
            if self._cache is not None:
                self._cache[i] = x
        return x, s.label

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            return np.zeros((0, *REGIME.input_shape), np.float32), np.zeros(0, np.int64)
        xs = np.stack([self[i][0] for i in range(len(self))])
        return xs, np.array([s.label for s in self.samples], dtype=np.int64)


def class_counts(samples: list[Sample]) -> np.ndarray:
    return np.bincount([s.label for s in samples], minlength=labels.num_classes())


def class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse-sqrt-frequency weights, mean 1 over present classes; absent classes get 0."""
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32)


def coverage_report(samples: list[Sample], min_per_jamo: int = 25, min_per_special: int = 25) -> dict:
    """T8/§7: >= min_per_jamo keystrokes per jamo *per participant*, and >= min_per_special per special token overall."""
    per_part = defaultdict(Counter)
    total = Counter()
    for s in samples:
        per_part[s.participant][s.symbol] += 1
        total[s.symbol] += 1
    short: list[dict] = []
    for part, cnt in sorted(per_part.items()):
        short += [{"participant": part, "symbol": j, "have": cnt[j], "need": min_per_jamo}
                  for j in labels.JAMO if cnt[j] < min_per_jamo]
    short += [{"participant": "*", "symbol": t, "have": total[t], "need": min_per_special}
              for t in labels.SPECIAL if total[t] < min_per_special]
    return {"ok": not short, "n_samples": len(samples), "counts": {k: total[k] for k in labels.LABELS}, "short": short}


EXIT_REGIME = 3  # the log-mel's param_hash does not match the frozen regime


def main(argv: list[str] | None = None) -> None:
    """Coverage check (T8/§7). Exit 0 = ok, 1 = coverage short, 3 = param_hash mismatch (NFR-2)."""
    import argparse
    import sys

    from .config import load_config

    ap = argparse.ArgumentParser(description="Check label/mel coverage and the param_hash of a dataset root")
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None, help="also write the full report (coverage, hash, sessions) here")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        samples, info = load_dataset(args.data, **load_options(cfg))
    except RegimeError as e:
        print(f"RegimeError: {e}", file=sys.stderr)
        raise SystemExit(EXIT_REGIME) from None
    rep = coverage_report(samples, cfg["data"]["min_per_jamo"], cfg["data"]["min_per_special"])
    if args.json:
        doc = {**info, "n_samples": rep["n_samples"], "participants": sorted({s.participant for s in samples}),
               "split": cfg["train"]["split"], "coverage_ok": rep["ok"], "coverage_short": rep["short"],
               "counts": rep["counts"]}
        Path(args.json).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):  # a piped Windows stdout (cp1252) cannot encode jamo: escape, don't crash
        sys.stdout.reconfigure(errors="backslashreplace")
    print(json.dumps({k: rep[k] for k in ("ok", "n_samples")}, indent=1))
    for row in rep["short"][:40]:
        print(f"  short: {row['participant']:>10} {row['symbol']:>8} have {row['have']} < {row['need']}")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
