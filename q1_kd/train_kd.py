"""
Q1.4 - Train the student networks with knowledge distillation from the ResNet-50 teacher.

Usage:
    python -m q1_kd.train_kd --arch all --epochs 30 --T 4 --alpha 0.7 --beta 1.0
    python -m q1_kd.train_kd --arch squeezenet --no-features      # logits-only KD (ablation)
"""
import argparse
import time

import torch

from common.data import get_loaders
from common.train_utils import (CKPT_DIR, RESULTS_DIR, Logger, count_params, evaluate,
                                get_device, make_optimizer, save_checkpoint, save_json, set_seed,
                                set_threads)
from q1_kd.kd import DistillationLoss
from q1_kd.models import STUDENTS, build_model


def load_teacher(device, path=CKPT_DIR / "teacher_resnet50.pt"):
    ckpt = torch.load(path, map_location="cpu")
    teacher = build_model("resnet50", ckpt["num_classes"], pretrained=False,
                          stem_stride=ckpt.get("stem_stride", 1))
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def distill(student, teacher, criterion, train_loader, test_loader, epochs, lr, log, tag):
    holder = torch.nn.ModuleList([student, criterion])   # connectors are trained jointly with the student
    opt, sched = make_optimizer(holder, lr, epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    history = []
    for ep in range(1, epochs + 1):
        student.train()
        criterion.train()
        t0 = time.time()
        sums = {"total": 0.0, "hard": 0.0, "soft": 0.0, "feat": 0.0}
        correct = seen = 0
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    t_logits, t_feats = teacher(x, return_features=True)
                s_logits, s_feats = student(x, return_features=True)
                loss, parts = criterion(s_logits, s_feats, t_logits, t_feats, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] += v.item() * y.size(0)
            correct += (s_logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        top1, top5, test_loss = evaluate(student, test_loader)
        rec = {"epoch": ep, "train_loss": sums["total"] / seen, "hard_loss": sums["hard"] / seen,
               "soft_loss": sums["soft"] / seen, "feat_loss": sums["feat"] / seen,
               "train_acc": 100.0 * correct / seen, "test_loss": test_loss,
               "test_top1": top1, "test_top5": top5, "epoch_time_s": time.time() - t0}
        history.append(rec)
        log(f"[{tag}] epoch {ep:3d}/{epochs} | loss {rec['train_loss']:.3f} (hard {rec['hard_loss']:.3f} "
            f"soft {rec['soft_loss']:.3f} feat {rec['feat_loss']:.3f}) | train acc {rec['train_acc']:.2f}% "
            f"| test top1 {top1:.2f}% top5 {top5:.2f}% | {rec['epoch_time_s']:.1f}s")
    return history


def main():
    ap = argparse.ArgumentParser(description="Knowledge distillation ResNet-50 -> students (CIFAR-100)")
    ap.add_argument("--arch", default="all", choices=STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of the soft-target term")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the feature (hint) term")
    ap.add_argument("--no-soft", action="store_true", help="disable soft-target KD")
    ap.add_argument("--no-features", action="store_true", help="disable feature-based KD")
    ap.add_argument("--train-subset", type=int, default=0, help="use only N training images (0 = all)")
    ap.add_argument("--test-subset", type=int, default=0, help="evaluate on N test images (0 = all 10k)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_threads(args.threads)
    device = get_device()
    log = Logger(RESULTS_DIR / "logs" / "kd_students.log")
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, device,
                                            train_subset=args.train_subset,
                                            test_subset=args.test_subset)
    teacher = load_teacher(device)
    t_top1, _, _ = evaluate(teacher, test_loader)
    log(f"Loaded teacher ResNet-50 (test top-1 {t_top1:.2f}%)")

    archs = STUDENTS if args.arch == "all" else [args.arch]
    for arch in archs:
        set_seed(args.seed)
        student = build_model(arch, 100).to(device)
        criterion = DistillationLoss(student.feature_channels, teacher.feature_channels,
                                     T=args.T, alpha=args.alpha, beta=args.beta,
                                     use_soft=not args.no_soft,
                                     use_features=not args.no_features).to(device)
        log(f"=== KD {arch} | params {count_params(student) / 1e6:.2f}M | {vars(args)}")
        history = distill(student, teacher, criterion, train_loader, test_loader,
                          args.epochs, args.lr, log, f"kd/{arch}")
        path = save_checkpoint(CKPT_DIR / f"{arch}_kd.pt", student, arch, "cifar100", 100, history,
                               extra={"method": "kd", "train_args": vars(args),
                                      "teacher_top1": t_top1})
        save_json(history, RESULTS_DIR / "q1" / f"history_{arch}_kd.json")
        log(f"Saved {path} (top-1 {history[-1]['test_top1']:.2f}%)")


if __name__ == "__main__":
    main()
