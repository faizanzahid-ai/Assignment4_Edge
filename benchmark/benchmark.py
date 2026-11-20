"""
Automated benchmark of every trained model (students baseline/KD, NAS model, teacher).

For each model it measures
  * parameters, checkpoint size, ONNX size
  * test top-1 accuracy with PyTorch and with ONNX Runtime (same images)
  * batch-1 CPU latency (median) with PyTorch and ONNX Runtime, 1 intra-op thread
  * ONNX Runtime batch-32 throughput (images / s)
  * peak inference memory (PyTorch CPU allocator)
and writes results/benchmark.csv plus results/q1/kd_vs_baseline.csv.

Usage:
    python -m benchmark.benchmark --limit 2000
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from common.data import get_loaders
from common.train_utils import RESULTS_DIR, count_params, evaluate
from inference.model_zoo import ALL_MODELS, ONNX_DIR, checkpoint_path, load_model
from q2_nas.hardware import measure_latency_ms, measure_peak_memory_mb


def ort_metrics(name, loader, threads=1):
    from inference.onnx_infer import OnnxPredictor
    path = ONNX_DIR / f"{name}.onnx"
    if not path.exists():
        return {}
    p = OnnxPredictor(name, threads)
    correct = total = 0
    for x, y in loader:
        out = p.logits(x.numpy().astype(np.float32))
        correct += int((out.argmax(1) == y.numpy()).sum())
        total += len(y)
    r = p.meta["resolution"]
    xb = np.random.randn(32, 3, r, r).astype(np.float32)
    p.logits(xb)
    t0 = time.perf_counter()
    for _ in range(5):
        p.logits(xb)
    thr = 32 * 5 / (time.perf_counter() - t0)
    return {"onnx_size_mb": path.stat().st_size / 1e6, "onnx_top1": 100.0 * correct / total,
            "onnx_latency_ms": p.latency_ms(runs=50), "onnx_throughput_ips_b32": thr}


def main():
    ap = argparse.ArgumentParser(description="Benchmark all trained models")
    ap.add_argument("--limit", type=int, default=2000, help="number of test images (0 = all 10k)")
    ap.add_argument("--threads", type=int, default=1, help="intra-op threads for latency")
    args = ap.parse_args()

    rows = []
    for name in ["teacher"] + ALL_MODELS:
        if not checkpoint_path(name).exists():
            print(f"skip {name}: no checkpoint")
            continue
        model, meta = load_model(name)
        _, test = get_loaders(meta["dataset"], 250, "cpu", resolution=meta["resolution"],
                              test_subset=args.limit or None)
        torch.set_num_threads(4)
        top1, top5, _ = evaluate(model, test)
        row = {"model": name, "arch": meta["arch"], "method": meta["method"], "dataset": meta["dataset"],
               "input_res": meta["resolution"], "params_M": count_params(model) / 1e6,
               "ckpt_size_mb": Path(checkpoint_path(name)).stat().st_size / 1e6,
               "torch_top1": top1, "torch_top5": top5,
               "torch_latency_ms": measure_latency_ms(model, meta["resolution"], threads=args.threads),
               "peak_mem_mb": measure_peak_memory_mb(model, meta["resolution"])}
        row.update(ort_metrics(name, test, args.threads))
        rows.append(row)
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    keys = list(dict.fromkeys(list(rows[0].keys()) + ["onnx_size_mb", "onnx_top1", "onnx_latency_ms",
                                                       "onnx_throughput_ips_b32"]))
    out = RESULTS_DIR / "benchmark.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out}")

    # KD vs baseline comparison table
    by = {r["model"]: r for r in rows}
    comp = []
    for a in ("mobilenetv2", "squeezenet", "efficientnet_lite0"):
        b, k = by.get(f"{a}_baseline"), by.get(f"{a}_kd")
        if b and k:
            comp.append({"student": a, "params_M": round(b["params_M"], 3),
                         "baseline_top1": round(b["torch_top1"], 2), "kd_top1": round(k["torch_top1"], 2),
                         "gain_pts": round(k["torch_top1"] - b["torch_top1"], 2),
                         "teacher_top1": round(by["teacher"]["torch_top1"], 2) if "teacher" in by else None,
                         "compression_x": round(by["teacher"]["params_M"] / b["params_M"], 1) if "teacher" in by else None})
    if comp:
        (RESULTS_DIR / "q1").mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "q1" / "kd_vs_baseline.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
            w.writeheader()
            w.writerows(comp)
        print(f"Saved {RESULTS_DIR / 'q1' / 'kd_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
