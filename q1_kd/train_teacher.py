"""
Q1.1 - Train the ResNet-50 teacher on CIFAR-100.

Usage:
    python -m q1_kd.train_teacher --epochs 15 --lr 0.05
"""
import argparse

from common.data import get_loaders
from common.train_utils import (CKPT_DIR, RESULTS_DIR, Logger, count_params, get_device,
                                save_checkpoint, save_json, set_seed, set_threads,
                                train_supervised)
from q1_kd.models import build_model


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a ResNet-50 teacher on CIFAR-100")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true", help="train from random init")
    ap.add_argument("--stem-stride", type=int, default=1, choices=[1, 2],
                    help="2 = 4x cheaper teacher for CPU-only training")
    ap.add_argument("--train-subset", type=int, default=0, help="use only N training images (0 = all)")
    ap.add_argument("--test-subset", type=int, default=0, help="evaluate on N test images (0 = all 10k)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    set_threads(args.threads)
    device = get_device()
    log = Logger(RESULTS_DIR / "logs" / "teacher_resnet50.log")
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, device,
                                            train_subset=args.train_subset,
                                            test_subset=args.test_subset)

    model = build_model("resnet50", 100, pretrained=not args.no_pretrained,
                        stem_stride=args.stem_stride).to(device)
    log(f"Teacher ResNet-50 | params {count_params(model) / 1e6:.2f}M | device {device} | {vars(args)}")
    history = train_supervised(model, train_loader, test_loader, args.epochs, args.lr,
                               label_smoothing=0.1, log=log, tag="teacher")

    path = save_checkpoint(CKPT_DIR / "teacher_resnet50.pt", model, "resnet50", "cifar100", 100,
                           history, extra={"train_args": vars(args), "stem_stride": args.stem_stride})
    save_json(history, RESULTS_DIR / "q1" / "history_teacher_resnet50.json")
    log(f"Saved teacher checkpoint -> {path} (top-1 {history[-1]['test_top1']:.2f}%)")


if __name__ == "__main__":
    main()
