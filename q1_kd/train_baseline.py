"""
Q1.5 - Baseline training of the student networks WITHOUT knowledge distillation
(plain cross-entropy on the hard labels).

Usage:
    python -m q1_kd.train_baseline --arch all --epochs 30
    python -m q1_kd.train_baseline --arch mobilenetv2 --epochs 30
"""
import argparse

from common.data import get_loaders
from common.train_utils import (CKPT_DIR, RESULTS_DIR, Logger, count_params, get_device,
                                save_checkpoint, save_json, set_seed, set_threads,
                                train_supervised)
from q1_kd.models import STUDENTS, build_model


def main():
    ap = argparse.ArgumentParser(description="Baseline (no-KD) student training on CIFAR-100")
    ap.add_argument("--arch", default="all", choices=STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-subset", type=int, default=0, help="use only N training images (0 = all)")
    ap.add_argument("--test-subset", type=int, default=0, help="evaluate on N test images (0 = all 10k)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_threads(args.threads)
    device = get_device()
    log = Logger(RESULTS_DIR / "logs" / "baseline_students.log")
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, device,
                                            train_subset=args.train_subset,
                                            test_subset=args.test_subset)
    archs = STUDENTS if args.arch == "all" else [args.arch]

    for arch in archs:
        set_seed(args.seed)   # identical initialisation / data order as the KD run
        model = build_model(arch, 100).to(device)
        log(f"=== Baseline {arch} | params {count_params(model) / 1e6:.2f}M | {vars(args)}")
        history = train_supervised(model, train_loader, test_loader, args.epochs, args.lr,
                                   log=log, tag=f"baseline/{arch}")
        path = save_checkpoint(CKPT_DIR / f"{arch}_baseline.pt", model, arch, "cifar100", 100,
                               history, extra={"method": "baseline", "train_args": vars(args)})
        save_json(history, RESULTS_DIR / "q1" / f"history_{arch}_baseline.json")
        log(f"Saved {path} (top-1 {history[-1]['test_top1']:.2f}%)")


if __name__ == "__main__":
    main()
