"""ESP32-S3 export (FR-9, T9, AC-6): PyTorch -> ONNX -> TFLite int8 (post-training quantization).

Run on a workstation with the export toolchain (`pip install -r requirements-export.txt`); the numpy
helpers below (`estimate_input_quant`, `check_ac6`) need nothing extra.

    python -m src.export_esp32 --ckpt runs/smallcnn/best.pt --data data/ --out export/

Outputs in --out: model.onnx, model_int8.tflite, deployment.json (labels, input regime, input/output
(scale, zero_point) to hand to the Mel component — O-O5), export_report.json (AC-6 numbers).

Host-side checks cannot prove the 30 ms NFR-1 latency: measure that on the device. MACs are reported as a proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import labels
from .melio import REGIME


# ---- pure numpy helpers ---------------------------------------------------------------------
def estimate_input_quant(calib: np.ndarray) -> dict:
    """Asymmetric int8 (scale, zero_point) from calibration min/max. An ESTIMATE for sanity checks; the
    converter's own values (read from the .tflite) are authoritative and are what deployment.json carries."""
    lo, hi = min(float(calib.min()), 0.0), max(float(calib.max()), 0.0)
    scale = (hi - lo) / 255.0 or 1.0
    zp = int(np.clip(round(-128 - lo / scale), -128, 127))
    return {"scale": scale, "zero_point": zp, "dtype": "int8"}


def quantize(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


def check_ac6(float_top1: float, int8_top1: float, size_bytes: int, *, max_drop_pp: float = 2.0,
              max_size_bytes: int = 1_000_000) -> dict:
    drop = (float_top1 - int8_top1) * 100.0
    return {"float_top1": float_top1, "int8_top1": int8_top1, "top1_drop_pp": drop, "size_bytes": size_bytes,
            "pass_accuracy": bool(drop <= max_drop_pp), "pass_size": bool(size_bytes <= max_size_bytes),
            "latency": "not measurable on host; measure the 30 ms budget (NFR-1) on the ESP32-S3"}


def pick_calibration(x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic representative subset (cfg export.calib_clips, default 300)."""
    idx = np.random.default_rng(seed).permutation(len(x))[:n]
    return x[np.sort(idx)]


# ---- steps needing the toolchain ------------------------------------------------------------
def export_onnx(ckpt: Path, onnx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """ONNX with static shape (1,1,64,48). Returns (float reference logits, probe inputs) for `verify_onnx`."""
    import torch

    from .infer import load_predictor

    predictor, _ = load_predictor(ckpt)
    model = predictor.model.eval().cpu()
    dummy = torch.zeros(1, *REGIME.input_shape)
    kw = dict(input_names=["logmel"], output_names=["logits"], opset_version=13)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kw)  # legacy exporter: plain ops, TFLite-friendly
    except TypeError:  # older torch without the `dynamo` flag
        torch.onnx.export(model, dummy, str(onnx_path), **kw)
    probe = torch.randn(4, *REGIME.input_shape, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        return model(probe).numpy(), probe.numpy()


def verify_onnx(onnx_path: Path, probe: np.ndarray, ref: np.ndarray, atol: float = 1e-3) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(None, {"logmel": probe[i:i + 1]})[0] for i in range(len(probe))])
    err = float(np.abs(got - ref).max())
    if err > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {err:.2e} (> {atol})")
    return err


def _nhwc_if_needed(x: np.ndarray, shape) -> np.ndarray:
    """onnx2tf turns NCHW inputs into NHWC; adapt (N,1,64,48) to whatever the converted model expects."""
    shape = tuple(int(s) for s in shape)
    return x.transpose(0, 2, 3, 1) if len(shape) == 4 and shape[-1] == 1 and shape[1] == REGIME.n_mels else x


def convert_tflite_int8(onnx_path: Path, calib: np.ndarray, work: Path, tflite_path: Path) -> None:
    import onnx2tf
    import tensorflow as tf

    saved = work / "saved_model"
    onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(saved), non_verbose=True)
    sig = tf.saved_model.load(str(saved)).signatures["serving_default"]
    in_shape = list(sig.structured_input_signature[1].values())[0].shape

    def rep():
        for i in range(len(calib)):
            yield [_nhwc_if_needed(calib[i:i + 1], in_shape).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_path.write_bytes(conv.convert())


class TFLitePredictor:
    """Runs the int8 model on the host (quantise input with the model's (scale, zp); dequantise logits)."""

    def __init__(self, tflite_path: Path):
        import tensorflow as tf

        self.it = tf.lite.Interpreter(model_path=str(tflite_path))
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def quant_info(self) -> dict:
        return {"input": {"scale": float(self.in_scale), "zero_point": int(self.in_zp), "dtype": "int8",
                          "shape": [int(s) for s in self.inp["shape"]]},
                "output": {"scale": float(self.out_scale), "zero_point": int(self.out_zp), "dtype": "int8"}}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(len(x)):
            q = quantize(_nhwc_if_needed(x[i:i + 1], self.inp["shape"]), self.in_scale, int(self.in_zp))
            self.it.set_tensor(self.inp["index"], q)
            self.it.invoke()
            rows.append(dequantize(self.it.get_tensor(self.out["index"])[0], self.out_scale, int(self.out_zp)))
        return np.stack(rows)


def main(argv: list[str] | None = None) -> None:
    from . import dataset, infer
    from .config import load_config
    from .model import complexity

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ckpt, out = Path(a.ckpt), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = dataset.load_samples(a.data, **dataset.load_options(cfg))
    split = dataset.load_split(ckpt.parent / "split.json", samples)
    calib_x, _ = dataset.LogMelDataset([samples[i] for i in split["train"]]).arrays()
    calib = pick_calibration(calib_x, cfg["export"]["calib_clips"])
    test_x, test_y = dataset.LogMelDataset([samples[i] for i in split["test"]]).arrays()
    if len(test_y) == 0:
        raise SystemExit("AC-6 needs a non-empty test split")

    ref, probe = export_onnx(ckpt, out / "model.onnx")
    onnx_err = verify_onnx(out / "model.onnx", probe, ref)
    convert_tflite_int8(out / "model.onnx", calib, out / "work", out / "model_int8.tflite")

    predictor, _ = infer.load_predictor(ckpt)
    f_logits, _ = infer.predict_dataset(predictor, dataset.LogMelDataset([samples[i] for i in split["test"]]), cfg["infer"]["batch_size"])
    tfl = TFLitePredictor(out / "model_int8.tflite")
    q_logits = np.concatenate([tfl(test_x[i:i + 64]) for i in range(0, len(test_x), 64)])
    size = (out / "model_int8.tflite").stat().st_size
    ac6 = check_ac6(float((f_logits.argmax(1) == test_y).mean()), float((q_logits.argmax(1) == test_y).mean()), size,
                    max_drop_pp=cfg["export"]["max_top1_drop_pp"], max_size_bytes=cfg["export"]["max_size_bytes"])
    ac6.update(onnx_max_abs_err=onnx_err, n_test=int(len(test_y)), macs=complexity(predictor.model)["macs"],
               input_quant_estimate=estimate_input_quant(calib))
    (out / "export_report.json").write_text(json.dumps(ac6, indent=2), encoding="utf-8")

    deploy = {"runtime": "tflite_int8", "model_file": "model_int8.tflite", "labels": list(labels.LABELS),
              "input_regime": {"shape": list(REGIME.input_shape), "sample_rate": REGIME.sample_rate,
                               "normalization": REGIME.normalization, "param_hash": REGIME.param_hash()},
              "quantization": tfl.quant_info(),
              "note": "Mel output (per-clip z-scored log-mel) must be quantised to int8 with input.scale/zero_point "
                      "and the layout given by input.shape before entering the CNN."}
    (out / "deployment.json").write_text(json.dumps(deploy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ac6, indent=2))
    print("hand deployment.json['quantization']['input'] to the Mel component")


if __name__ == "__main__":
    main()
