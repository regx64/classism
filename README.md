# classism — KDAA Acoustic Front-End (AFE)

CNN that classifies one keystroke's log-mel `(1,64,48)` into 38 symbols (33 Hangul jamo + `<sp> <bs> <shift> <caps> <other>`)
and emits a top-N distribution per keystroke as `kdaa.afe.v1` JSON for the LM Server. Implements [`spec.md`](spec.md),
which is the source of truth (requirement IDs FR/NFR/AC and tickets T1–T10 are referenced in the code).

```
src/labels.py        T1  38-symbol space, keytype_of, index maps
src/hangul.py        D6  jamo tables + composition/beam-search DEMO (never used by infer.py)
src/melio.py         T2  Mel input adapter, frozen regime, param_hash, shape validation (AC-8)
src/model.py         T3  SmallCNN (deployed), CoAtNetLite (offline comparison only)
src/dataset.py       T4/T8  label CSV + mel/<clip_id>.npy loader, session/participant splits, coverage report
src/train.py         T4  training (seeded, session-wise split saved to split.json)
src/infer.py         T5/T10  log-mel -> kdaa.afe.v1; batch + streaming; calibration.json pickup
src/schema.py            JSON-Schema + semantic validation of output documents
src/calibrate.py     T7  temperature fit + ECE (numpy)
src/evaluate.py      T6  top-1/N, coverage, per-scenario, keytype acc, confusion, ECE, AC verdicts, grayscale figures
src/export_esp32.py  T9  PyTorch -> ONNX -> TFLite int8, (scale, zero_point), AC-6 report
src/synth.py             synthetic dataset in the real layout — smoke tests only, never report its accuracy
schemas/kdaa_afe_v1.schema.json   configs/afe.yaml   tests/
```

## Setup
```bash
pip install -r requirements.txt            # training / evaluation / inference
pip install -r requirements-export.txt     # + ONNX / TensorFlow / onnx2tf, only for export
```

## Workflow
```bash
# data layout:  <data>/labels/<session>.csv   (onset_s,jamo,keytype,shift,scenario,participant[,clip_id])
#               <data>/mel/<clip_id>.npy      (float32 (64,48), final z-scored log-mel from the Mel component)
#               <data>/mel/_meta.json         (Mel header; its param_hash is checked against the frozen regime)
#               <data>/mel_packed/<session>.npz  (optional session shard: logmel (N,64,48), clip_id (N,), param_hash)
python -m src.dataset  --data data/                       # T8: coverage check (>=25 per jamo per participant, per special token)
python -m src.train    --data data/ --out runs/a          # T4
python -m src.evaluate --ckpt runs/a/best.pt --data data/ # T6 (+ fits T, writes runs/a/calibration.json)  -> runs/a/eval_test/
python -m src.export_esp32 --ckpt runs/a/best.pt --data data/ --out export/   # T9 / AC-6
```
```python
from src.infer import afe_from_checkpoint
from src.melio import MelSession
afe = afe_from_checkpoint("runs/a/best.pt", top_n=5)      # uses calibration.json if present -> meta.calibrated
doc = afe.run(MelSession.from_npz("session.npz"))         # batch;  afe.stream(sid).push(onset_s, logmel) for streaming
```
Without any ML framework you can smoke-test the post-processing: `python -m pytest --ignore=tests/test_torch_pipeline.py`.
Tests that run a real tiny model are marked `torch` (`-m "not torch"` skips them).

## Decisions where the spec is silent (change if wrong)
- **`param_hash`**: first 8 hex of sha256 over the canonical JSON of the frozen `Regime`. The Mel component must produce the same
  value (O-O3); a session with a different hash is rejected (NFR-2). Pass `verify_hash=False` only while agreeing on this.
  Training data is checked too: `load_samples` reads each session's hash from its shard header, else `mel/_meta.json`,
  and raises `RegimeError` on a mismatch; data without any hash only warns unless `data.require_param_hash: true`.
- **Session shards** (`mel_packed/<session>.npz`): read instead of the per-clip `.npy` files when present, so tens of
  thousands of clips are a handful of files. `data.sessions` limits training to listed sessions, and
  `data.on_missing_mel: drop` leaves out label rows the Mel component could not cut (it reports them), instead of failing.
  `python -m src.dataset --json out.json` writes the check (coverage, hash, sessions); exit 3 = hash mismatch.
- **Label CSV → clip**: `clip_id` column if present, else `<session_id>_<row:05d>`; `session_id` = CSV file stem.
- **`alts` length**: always exactly `topN` (the spec's example shows one alt for `<sp>`; the schema allows 1..38 but the AFE emits N).
- **`model_id`**: `<model>-fp32-v1` for the PyTorch model; the int8 deployment is identified by `deployment.json`.
- **Special-token minimum** (`data.min_per_special: 25`) is a placeholder for open item O-Q6.
- The recorder in `flow.png` says 44.1 kHz; `spec.md` D2 freezes **48 kHz**, which is what the code enforces.

## Verification status
Verified here: label/Mel/schema/inference/streaming/calibration/metrics/dataset/split logic (numpy-only tests) and figure generation.
**Not run here** (needs a real machine, no ML stack was run in the dev Codespace): training, `evaluate` on a checkpoint, the torch
tests, and the ONNX/TFLite export — these are written to the spec but untested. Accuracy targets (AC-3/4/5) and int8 loss / on-device
latency (AC-6, NFR-1: ≤30 ms per keystroke on the ESP32-S3) can only be measured on real data and hardware.
