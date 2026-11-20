"""
Single entry point that knows how to rebuild every trained model from its checkpoint.
Used by the unified PyTorch inference pipeline, the ONNX exporter and the benchmark.
"""
from pathlib import Path

import torch

from common.train_utils import CKPT_DIR

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "onnx_models"

STUDENT_MODELS = [f"{a}_{m}" for a in ("mobilenetv2", "squeezenet", "efficientnet_lite0")
                  for m in ("baseline", "kd")]
ALL_MODELS = STUDENT_MODELS + ["nas_best"]


def checkpoint_path(name):
    if name == "teacher":
        return CKPT_DIR / "teacher_resnet50.pt"
    return CKPT_DIR / f"{name}.pt"


def load_model(name, device="cpu"):
    """Return (model in eval mode, metadata dict with dataset / resolution / accuracy)."""
    ckpt = torch.load(checkpoint_path(name), map_location="cpu")
    if ckpt["arch"] == "nas":
        from q2_nas.generator import build_from_config
        model = build_from_config(ckpt["arch_config"], ckpt["num_classes"])
        resolution = ckpt["arch_config"]["resolution"]
    else:
        from q1_kd.models import build_model
        model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False,
                            stem_stride=ckpt.get("stem_stride", 1))
        resolution = 32
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {"name": name, "arch": ckpt["arch"], "dataset": ckpt["dataset"],
            "num_classes": ckpt["num_classes"], "resolution": resolution,
            "method": ckpt.get("method", "nas" if ckpt["arch"] == "nas" else "teacher"),
            "train_top1": ckpt.get("test_top1")}
    return model, meta
