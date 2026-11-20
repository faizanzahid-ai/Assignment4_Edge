"""
CPU inference with ONNX Runtime for all exported student models and the NAS model.
Depends only on numpy, Pillow and onnxruntime (no PyTorch needed at deployment time).

Examples
    python -m inference.onnx_infer --model all                        # bundled sample images
    python -m inference.onnx_infer --model squeezenet_kd --images my.png
    python -m inference.onnx_infer --model nas_best --evaluate --limit 2000
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "onnx_models"
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}


def preprocess(path, dataset, resolution):
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    if resolution != 32:
        img = img.resize((resolution, resolution), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean, std = (np.array(v, dtype=np.float32) for v in STATS[dataset])
    return ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OnnxPredictor:
    def __init__(self, name, threads=1):
        self.path = ONNX_DIR / f"{name}.onnx"
        self.meta = json.load(open(self.path.with_suffix(".json")))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        from common.data import class_names
        self.classes = class_names(self.meta["dataset"])

    def logits(self, x):
        return self.sess.run(None, {"input": x})[0]

    def predict_files(self, paths, topk=3):
        x = np.concatenate([preprocess(p, self.meta["dataset"], self.meta["resolution"]) for p in paths])
        probs = softmax(self.logits(x))
        idx = np.argsort(-probs, axis=1)[:, :topk]
        return [[(self.classes[i], float(probs[r, i])) for i in row] for r, row in enumerate(idx)]

    def latency_ms(self, runs=100, warmup=10):
        r = self.meta["resolution"]
        x = np.random.randn(1, 3, r, r).astype(np.float32)
        for _ in range(warmup):
            self.logits(x)
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.logits(x)
            t.append((time.perf_counter() - t0) * 1000)
        return float(np.median(t))

    def evaluate(self, limit=None, batch=256):
        from common.data import load_cifar, normalize
        import torch
        import torch.nn.functional as F
        _, _, x, y = load_cifar(self.meta["dataset"])
        if limit:   # seeded random subset (the PNG-mirror test set is ordered by class)
            idx = np.random.RandomState(0).permutation(len(y))[:limit]
            x, y = x[idx], y[idx]
        correct = 0
        for i in range(0, len(y), batch):
            xb = torch.from_numpy(x[i:i + batch]).float() / 255.0
            if self.meta["resolution"] != 32:
                xb = F.interpolate(xb, size=self.meta["resolution"], mode="bilinear", align_corners=False)
            xb = normalize(xb, self.meta["dataset"]).numpy().astype(np.float32)
            correct += int((self.logits(xb).argmax(1) == y[i:i + batch]).sum())
        return 100.0 * correct / len(y)


def main():
    from inference.model_zoo import ALL_MODELS
    ap = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    ap.add_argument("--model", default="all")
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on a random subset of N test images")
    args = ap.parse_args()

    print(f"ONNX Runtime {ort.__version__} | providers {ort.get_available_providers()}")
    for name in (ALL_MODELS if args.model == "all" else [args.model]):
        p = OnnxPredictor(name, args.threads)
        files = args.images or sorted(glob.glob(str(ROOT / "samples" / p.meta["dataset"] / "*.png")))
        files = [f for pat in files for f in glob.glob(pat)]
        print(f"\n=== {name} [{p.path.name}, {p.path.stat().st_size / 1e6:.2f} MB, "
              f"input {p.meta['resolution']}px] latency(batch 1, {args.threads} thread) "
              f"{p.latency_ms():.2f} ms")
        for f, r in zip(files, p.predict_files(files) if files else []):
            print(f"  {Path(f).name:<28} -> " + ", ".join(f"{c} {s * 100:.1f}%" for c, s in r))
        if args.evaluate:
            print(f"  ONNX test top-1: {p.evaluate(args.limit):.2f}%")


if __name__ == "__main__":
    main()
