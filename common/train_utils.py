"""Generic training / evaluation helpers shared by all experiments."""
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_threads(n):
    if n and n > 0:
        torch.set_num_threads(n)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    """Serialized size of the model weights in MB (what is stored on disk)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def make_optimizer(model, lr, epochs, steps_per_epoch, wd=5e-4):
    """SGD + Nesterov momentum with a one-cycle learning-rate schedule."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                pct_start=0.25)
    return opt, sched


@torch.no_grad()
def evaluate(model, loader):
    """Top-1 / top-5 accuracy (%) and mean CE loss."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct1 = correct5 = total = 0
    loss = 0.0
    use_amp = next(model.parameters()).is_cuda
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        out = out.float()
        loss += ce(out, y).item()
        top5 = out.topk(min(5, out.size(1)), dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(dim=1).sum().item()
        total += y.numel()
    return 100.0 * correct1 / total, 100.0 * correct5 / total, loss / total


def train_supervised(model, train_loader, test_loader, epochs, lr, wd=5e-4,
                     label_smoothing=0.0, log=print, tag=""):
    """Standard cross-entropy training loop (used by teacher, baselines and NAS)."""
    opt, sched = make_optimizer(model, lr, epochs, len(train_loader), wd)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = run_correct = seen = 0
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
        top1, top5, test_loss = evaluate(model, test_loader)
        rec = {"epoch": ep, "train_loss": run_loss / seen, "train_acc": 100.0 * run_correct / seen,
               "test_loss": test_loss, "test_top1": top1, "test_top5": top5,
               "epoch_time_s": time.time() - t0}
        history.append(rec)
        log(f"[{tag}] epoch {ep:3d}/{epochs} | train loss {rec['train_loss']:.3f} "
            f"acc {rec['train_acc']:.2f}% | test loss {test_loss:.3f} top1 {top1:.2f}% "
            f"top5 {top5:.2f}% | {rec['epoch_time_s']:.1f}s")
    return history


def save_checkpoint(path, model, arch, dataset, num_classes, history, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"arch": arch, "dataset": dataset, "num_classes": num_classes,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history": history,
            "test_top1": history[-1]["test_top1"] if history else None}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    return path


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Logger:
    """Print to console and append to a log file at the same time."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
