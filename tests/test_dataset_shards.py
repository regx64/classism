"""Session shards, the param_hash gate (NFR-2) and the data.* load options."""
import json

import numpy as np
import pytest

from src import dataset, synth
from src.config import load_config
from src.melio import REGIME, RegimeError


@pytest.fixture
def root(tmp_path):
    return synth.generate(tmp_path / "d", participants=3, sessions_per_scenario=1, per_session=40, seed=3)


def sessions_of(root):
    return sorted(p.stem for p in (root / "labels").glob("*.csv"))


def pack(root, sid, *, param_hash=None, drop_npy=True, skip=()):
    """Write mel_packed/<sid>.npz the way the pipeline does (same arrays, CSV order)."""
    ids = [s.clip_id for s in dataset.load_samples(root, sessions=[sid]) if s.clip_id not in skip]
    logmel = np.stack([np.load(root / "mel" / f"{c}.npy") for c in ids])
    (root / "mel_packed").mkdir(exist_ok=True)
    np.savez(root / "mel_packed" / f"{sid}.npz", logmel=logmel, clip_id=np.array(ids, dtype=str),
             param_hash=param_hash or REGIME.param_hash(), mel_spec_version="1.0", session_id=sid)
    if drop_npy:
        for c in ids:
            (root / "mel" / f"{c}.npy").unlink()
    return ids


def test_synth_writes_the_mel_header(root):
    meta = json.loads((root / "mel" / "_meta.json").read_text(encoding="utf-8"))
    assert meta["param_hash"] == REGIME.param_hash() and (meta["n_mels"], meta["frames"]) == (64, 48)


def test_a_shard_replaces_the_per_clip_files(root):
    sid = sessions_of(root)[0]
    before = dataset.LogMelDataset(dataset.load_samples(root, sessions=[sid])).arrays()
    pack(root, sid)
    samples, info = dataset.load_dataset(root)
    mine = [s for s in samples if s.session_id == sid]
    assert info["shard_sessions"] == [sid] and all(s.shard is not None for s in mine)
    assert not any(s.shard for s in samples if s.session_id != sid)
    after = dataset.LogMelDataset(mine).arrays()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_a_clip_missing_from_its_shard_is_loud_or_dropped(root):
    sid = sessions_of(root)[0]
    gone = dataset.load_samples(root, sessions=[sid])[5].clip_id
    pack(root, sid, skip={gone})
    with pytest.raises(FileNotFoundError, match=gone):
        dataset.load_samples(root)
    samples, info = dataset.load_dataset(root, on_missing="drop")
    assert info["dropped_no_mel"] == 1 and gone not in {s.clip_id for s in samples}


def test_drop_also_covers_missing_npy(root):
    (root / "mel" / f"{sessions_of(root)[1]}_00003.npy").unlink()
    _, info = dataset.load_dataset(root, on_missing="drop")
    assert info["dropped_no_mel"] == 1


def test_sessions_option_selects_and_rejects_unknown(root):
    keep = sessions_of(root)[:2]
    samples, info = dataset.load_dataset(root, sessions=keep)
    assert {s.session_id for s in samples} == set(keep) == set(info["sessions"])
    with pytest.raises(FileNotFoundError, match="nope"):
        dataset.load_samples(root, sessions=["nope"])


def test_header_from_another_regime_is_refused(root):
    meta = root / "mel" / "_meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "param_hash": "5711c12b"}), encoding="utf-8")
    with pytest.raises(RegimeError, match="5711c12b"):
        dataset.load_samples(root)
    dataset.load_samples(root, verify_hash=False)  # explicit opt-out still loads


def test_shard_from_another_regime_is_refused(root):
    sid = sessions_of(root)[0]
    pack(root, sid, param_hash="deadbeef")
    with pytest.raises(RegimeError, match=sid):
        dataset.load_samples(root)
    dataset.load_samples(root, sessions=sessions_of(root)[1:])  # other sessions are unaffected


def test_missing_header_warns_or_is_refused_when_required(root):
    (root / "mel" / "_meta.json").unlink()
    with pytest.warns(UserWarning, match="no param_hash"):
        dataset.load_samples(root)
    with pytest.raises(RegimeError, match="no param_hash"):
        dataset.load_samples(root, require_hash=True)


def test_load_options_come_from_the_config(tmp_path):
    cfg = load_config()
    assert dataset.load_options(cfg) == {"sessions": None, "on_missing": "error", "require_hash": False}
    bad = tmp_path / "c.yaml"
    bad.write_text("data:\n  on_missing_mel: ignore\n", encoding="utf-8")
    with pytest.raises(ValueError, match="on_missing_mel"):
        load_config(bad)


def test_cli_writes_json_and_exits_3_on_a_foreign_hash(root, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data:\n  sessions: {json.dumps(sessions_of(root)[:3])}\n  min_per_jamo: 0\n  min_per_special: 0\n",
                   encoding="utf-8")
    out = tmp_path / "check.json"
    with pytest.raises(SystemExit) as e:
        dataset.main(["--data", str(root), "--config", str(cfg), "--json", str(out)])
    assert e.value.code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["param_hash"] == REGIME.param_hash() and len(doc["sessions"]) == 3 and doc["coverage_ok"]
    assert doc["n_samples"] == 120 and doc["participants"]
    (root / "mel" / "_meta.json").write_text(json.dumps({"param_hash": "5711c12b"}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        dataset.main(["--data", str(root), "--config", str(cfg)])
    assert e.value.code == dataset.EXIT_REGIME
