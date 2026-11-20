"""
Unified PyTorch inference pipeline for all trained student models (Q1.6) and the NAS model (Q2).

Examples
    # classify image files with one model
    python -m inference.infer --model mobilenetv2_kd --images samples/cifar100/*.png
    # run every model on the bundled sample images
    python -m inference.infer --model all
    # evaluate a model on the full test set
    python -m inference.infer --model nas_best --evaluate
"""
import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.data import class_names, get_loaders, preprocess_image
from common.train_utils import ROOT, evaluate
from inference.model_zoo import ALL_MODELS, load_model


class Predictor:
    """Loads a checkpoint once and classifies images / tensors with it."""

    def __init__(self, name, device="cpu"):
        self.model, self.meta = load_model(name, device)
        self.device = device
        self.classes = class_names(self.meta["dataset"])

    @torch.no_grad()
    def predict_array(self, x, topk=3):
        logits = self.model(torch.as_tensor(x, device=self.device))
        probs = F.softmax(logits, dim=1)
        p, idx = probs.topk(topk, dim=1)
        return [[(self.classes[i], float(v)) for v, i in zip(pr, ix)] for pr, ix in zip(p.tolist(), idx.tolist())]

    def predict_files(self, paths, topk=3):
        x = np.concatenate([preprocess_image(p, self.meta["dataset"], self.meta["resolution"]) for p in paths])
        return self.predict_array(x, topk)


def default_images(dataset):
    return sorted(glob.glob(str(ROOT / "samples" / dataset / "*.png")))


def main():
    ap = argparse.ArgumentParser(description="Unified PyTorch inference for student / NAS models")
    ap.add_argument("--model", default="all", help=f"one of {ALL_MODELS} or 'all'")
    ap.add_argument("--images", nargs="*", help="image files (default: bundled samples)")
    ap.add_argument("--evaluate", action="store_true", help="also report test-set accuracy")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    names = ALL_MODELS if args.model == "all" else [args.model]
    for name in names:
        pred = Predictor(name)
        files = args.images or default_images(pred.meta["dataset"])
        files = [f for pat in files for f in glob.glob(pat)]
        print(f"\n=== {name}  ({pred.meta['arch']}, {pred.meta['dataset']}, input {pred.meta['resolution']}px)")
        t0 = time.perf_counter()
        results = pred.predict_files(files, args.topk) if files else []
        dt = (time.perf_counter() - t0) * 1000
        for f, r in zip(files, results):
            top = ", ".join(f"{c} {p * 100:.1f}%" for c, p in r)
            print(f"  {f.split(chr(92))[-1].split('/')[-1]:<28} -> {top}")
        if files:
            print(f"  {len(files)} images in {dt:.1f} ms (CPU, PyTorch)")
        if args.evaluate:
            _, test = get_loaders(pred.meta["dataset"], 256, "cpu", resolution=pred.meta["resolution"])
            top1, top5, _ = evaluate(pred.model, test)
            print(f"  test accuracy: top-1 {top1:.2f}%  top-5 {top5:.2f}%")


if __name__ == "__main__":
    main()
