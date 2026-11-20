"""
Q2.3 / Q2.4 / Q2.6 - Hardware-aware evolutionary NAS (regularized / aging evolution,
Real et al., 2019) on CIFAR-10.

Loop
  1. Initial population: random architectures that satisfy the deployment constraints.
  2. Repeat for `cycles`:
       - tournament: sample `tournament` members, best fitness becomes the parent,
       - child = mutate(parent)  (one gene re-sampled),
       - HARDWARE CHECK: measure latency / size / peak memory on the CPU;
         a child violating a constraint is REJECTED (logged, never trained),
       - otherwise: proxy-train on a CIFAR-10 subset and validate on a held-out
         part of the *training* set (the test set is never used during search),
       - add child, remove the OLDEST member (aging).
  3. Best architecture (highest fitness) -> results/nas/best_arch.json

Fitness (MnasNet-style soft multi-objective reward, applied to feasible models only):
    fitness = val_acc * (latency / max_latency) ^ w * (size / max_size) ^ w,   w = -0.07

Usage:
    python -m q2_nas.evolution --population 8 --cycles 20 --proxy-epochs 2
"""
import argparse
import csv
import json
import random
import time

import torch

from common.data import get_loaders
from common.train_utils import RESULTS_DIR, Logger, evaluate, get_device, set_seed, train_supervised
from q2_nas.generator import build_from_config
from q2_nas.hardware import DEFAULT_CONSTRAINTS, check_constraints, profile_model
from q2_nas.search_space import arch_key, describe, mutate, sample_architecture

NAS_DIR = RESULTS_DIR / "nas"


class Evaluator:
    """Profiles, filters and proxy-trains candidate architectures (with caching)."""

    def __init__(self, args, constraints, log):
        self.args, self.constraints, self.log = args, constraints, log
        self.device = get_device()
        self.loaders = {}
        self.cache = {}
        self.history = []

    def _loaders(self, res):
        if res not in self.loaders:
            self.loaders[res] = get_loaders("cifar10", self.args.batch_size, self.device, resolution=res,
                                            train_subset=self.args.proxy_train, val_split=self.args.proxy_val,
                                            seed=self.args.seed)
        return self.loaders[res]

    def __call__(self, arch, origin):
        key = arch_key(arch)
        if key in self.cache:
            return self.cache[key]
        idx = len(self.history)
        model = build_from_config(arch, 10)
        prof = profile_model(model, arch["resolution"], threads=self.args.latency_threads)
        violations = check_constraints(prof, self.constraints)
        rec = {"id": idx, "origin": origin, "arch": describe(arch), "config": json.dumps(arch),
               "params": prof["params"], "size_mb": round(prof["size_mb"], 4),
               "latency_ms": round(prof["latency_ms"], 3),
               "peak_mem_mb": None if prof["peak_mem_mb"] is None else round(prof["peak_mem_mb"], 3),
               "status": "rejected" if violations else "trained",
               "violations": "; ".join(violations), "val_acc": None, "fitness": None, "train_time_s": None}
        if violations:
            self.log(f"[{idx:3d}] REJECTED {describe(arch)} -> {rec['violations']}")
        else:
            t0 = time.time()
            set_seed(self.args.seed)
            model = model.to(self.device)
            tr, va = self._loaders(arch["resolution"])
            train_supervised(model, tr, va, self.args.proxy_epochs, self.args.lr,
                             log=lambda *_: None, tag="proxy")
            acc, _, _ = evaluate(model, va)
            c = self.constraints
            fitness = acc * (prof["latency_ms"] / c["max_latency_ms"]) ** self.args.w \
                          * (prof["size_mb"] / c["max_size_mb"]) ** self.args.w
            rec.update(val_acc=round(acc, 2), fitness=round(fitness, 3), train_time_s=round(time.time() - t0, 1))
            self.log(f"[{idx:3d}] val {acc:5.2f}% fit {fitness:6.2f} | {prof['latency_ms']:.2f}ms "
                     f"{prof['size_mb']:.3f}MB mem {rec['peak_mem_mb']}MB | {describe(arch)}")
        self.history.append(rec)
        self.cache[key] = rec
        return rec


def save_history(history, best):
    NAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(NAS_DIR / "search_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(NAS_DIR / "search_history.json", "w") as f:
        json.dump(history, f, indent=2)
    if best is not None:
        with open(NAS_DIR / "best_arch.json", "w") as f:
            json.dump({"config": json.loads(best["config"]), "summary": best["arch"],
                       "search_metrics": {k: best[k] for k in ("val_acc", "fitness", "latency_ms",
                                                                 "size_mb", "peak_mem_mb", "params")}},
                      f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Hardware-aware evolutionary NAS on CIFAR-10")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--tournament", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=20, help="number of mutated children evaluated")
    ap.add_argument("--proxy-epochs", type=int, default=2)
    ap.add_argument("--proxy-train", type=int, default=10000, help="training images for proxy training")
    ap.add_argument("--proxy-val", type=int, default=2000, help="held-out training images for validation")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--w", type=float, default=-0.07, help="latency/size exponent of the reward")
    ap.add_argument("--max-latency-ms", type=float, default=3.0)
    ap.add_argument("--max-size-mb", type=float, default=1.0)
    ap.add_argument("--max-peak-mem-mb", type=float, default=4.0)
    ap.add_argument("--latency-threads", type=int, default=1, help="CPU threads used for latency measurement")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for training (0 = default)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    constraints = {"max_latency_ms": args.max_latency_ms, "max_size_mb": args.max_size_mb,
                   "max_peak_mem_mb": args.max_peak_mem_mb}
    rng = random.Random(args.seed)
    log = Logger(RESULTS_DIR / "logs" / "nas_search.log")
    log(f"NAS start | constraints {constraints} | {vars(args)}")
    ev = Evaluator(args, constraints, log)

    # 1) initial population of feasible random architectures
    population, tries = [], 0
    while len(population) < args.population and tries < 50 * args.population:
        tries += 1
        arch = sample_architecture(rng)
        rec = ev(arch, "random")
        if rec["status"] == "trained":
            population.append((arch, rec))

    # 2) aging evolution
    for cycle in range(args.cycles):
        sample = rng.sample(population, min(args.tournament, len(population)))
        parent = max(sample, key=lambda p: p[1]["fitness"])[0]
        for _ in range(20):                       # mutate until a feasible child is found
            child = mutate(parent, rng)
            rec = ev(child, f"mutation(cycle {cycle})")
            if rec["status"] == "trained":
                break
        if rec["status"] == "trained":
            population.append((child, rec))
            population.pop(0)                     # remove the oldest individual
        save_history(ev.history, None)

    trained = [r for r in ev.history if r["status"] == "trained"]
    best = max(trained, key=lambda r: r["fitness"])
    save_history(ev.history, best)
    n_rej = sum(r["status"] == "rejected" for r in ev.history)
    log(f"Search finished: {len(ev.history)} candidates, {len(trained)} trained, {n_rej} rejected by constraints")
    log(f"BEST: {best['arch']} | val {best['val_acc']}% | fitness {best['fitness']} | "
        f"{best['latency_ms']}ms {best['size_mb']}MB mem {best['peak_mem_mb']}MB")
    log(f"Saved {NAS_DIR / 'best_arch.json'} and search_history.csv/json")


if __name__ == "__main__":
    main()
