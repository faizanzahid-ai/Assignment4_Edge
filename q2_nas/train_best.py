"""
Q2.7 - Load the best discovered architecture and train it FROM SCRATCH on the
full CIFAR-10 training set; evaluate on the CIFAR-10 test set.

Usage:
    python -m q2_nas.train_best --epochs 30
"""
import argparse
import json

import torch

from common.data import get_loaders
from common.train_utils import (CKPT_DIR, RESULTS_DIR, Logger, get_device, save_checkpoint,
                                save_json, set_seed, train_supervised)
from q2_nas.generator import build_from_config
from q2_nas.hardware import profile_model
from q2_nas.search_space import describe


def main():
    ap = argparse.ArgumentParser(description="Train the best NAS architecture from scratch")
    ap.add_argument("--arch-file", default=str(RESULTS_DIR / "nas" / "best_arch.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    arch = json.load(open(args.arch_file))["config"]
    set_seed(args.seed)
    device = get_device()
    log = Logger(RESULTS_DIR / "logs" / "nas_train_best.log")
    train_loader, test_loader = get_loaders("cifar10", args.batch_size, device,
                                            resolution=arch["resolution"], train_subset=args.train_subset)
    model = build_from_config(arch, 10)
    prof = profile_model(model, arch["resolution"])
    log(f"Best architecture: {describe(arch)} | {prof} | {vars(args)}")
    history = train_supervised(model.to(device), train_loader, test_loader, args.epochs, args.lr,
                               log=log, tag="nas_best")
    path = save_checkpoint(CKPT_DIR / "nas_best.pt", model, "nas", "cifar10", 10, history,
                           extra={"arch_config": arch, "hardware_profile": prof, "train_args": vars(args)})
    save_json(history, RESULTS_DIR / "nas" / "history_nas_best.json")
    log(f"Saved {path} (test top-1 {history[-1]['test_top1']:.2f}%)")


if __name__ == "__main__":
    main()
