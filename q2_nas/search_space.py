"""
Q2.1 - Searchable CNN architecture space.

An architecture is a plain JSON-serialisable dict:

{
  "resolution": 28,                       # input resolution (variable input size)
  "stem_channels": 24,
  "stages": [                             # variable depth: 2-4 stages x 1-3 blocks each
     {"depth": 2, "channels": 32, "kernel": 3, "block": "dws", "activation": "relu6"},
     ...
  ],
  "dropout": 0.1
}

Searchable dimensions
  * number of stages and blocks per stage   -> convolutional depth
  * channels per stage                      -> width
  * kernel size per stage (3/5/7)
  * block type: standard conv, depthwise-separable conv, or inverted residual (MBConv)
  * activation function per stage
  * input resolution
The first stage keeps the resolution, every following stage halves it.
"""
import copy
import json
import random

SPACE = {
    "resolution": [20, 24, 28, 32],
    "stem_channels": [8, 16, 24, 32],
    "num_stages": [2, 3, 4],
    "depth": [1, 2, 3],
    "channels": [16, 24, 32, 48, 64, 96, 128],
    "kernel": [3, 5, 7],
    "block": ["conv", "dws", "mbconv"],
    "activation": ["relu", "relu6", "hardswish", "silu", "leaky_relu"],
    "dropout": [0.0, 0.1, 0.2],
}


def sample_stage(rng, prev_channels=None):
    channels = rng.choice(SPACE["channels"])
    if prev_channels is not None:                     # keep width non-decreasing
        bigger = [c for c in SPACE["channels"] if c >= prev_channels]
        channels = rng.choice(bigger or [SPACE["channels"][-1]])
    return {"depth": rng.choice(SPACE["depth"]), "channels": channels,
            "kernel": rng.choice(SPACE["kernel"]), "block": rng.choice(SPACE["block"]),
            "activation": rng.choice(SPACE["activation"])}


def sample_architecture(rng=random):
    """Uniformly sample a random architecture configuration."""
    n = rng.choice(SPACE["num_stages"])
    stages, prev = [], None
    for _ in range(n):
        s = sample_stage(rng, prev)
        stages.append(s)
        prev = s["channels"]
    return {"resolution": rng.choice(SPACE["resolution"]),
            "stem_channels": rng.choice(SPACE["stem_channels"]),
            "stages": stages, "dropout": rng.choice(SPACE["dropout"])}


def mutate(arch, rng=random):
    """Return a copy of `arch` with exactly one randomly chosen gene re-sampled."""
    a = copy.deepcopy(arch)
    gene = rng.choice(["resolution", "stem_channels", "dropout", "add_stage", "remove_stage",
                       "depth", "channels", "kernel", "block", "activation"])
    if gene in ("resolution", "stem_channels", "dropout"):
        a[gene] = rng.choice([v for v in SPACE[gene] if v != a[gene]])
    elif gene == "add_stage" and len(a["stages"]) < max(SPACE["num_stages"]):
        a["stages"].append(sample_stage(rng, a["stages"][-1]["channels"]))
    elif gene == "remove_stage" and len(a["stages"]) > min(SPACE["num_stages"]):
        a["stages"].pop(rng.randrange(len(a["stages"])))
    else:
        if gene in ("add_stage", "remove_stage"):
            gene = "depth"
        stage = rng.choice(a["stages"])
        stage[gene] = rng.choice([v for v in SPACE[gene] if v != stage[gene]])
    return a


def arch_key(arch):
    """Canonical string used to detect duplicate architectures."""
    return json.dumps(arch, sort_keys=True)


def describe(arch):
    """Compact human-readable summary, e.g. 'r28 s16 | dws k3 32x2 relu6 | ...'."""
    parts = [f"r{arch['resolution']} s{arch['stem_channels']}"]
    for s in arch["stages"]:
        parts.append(f"{s['block']} k{s['kernel']} {s['channels']}x{s['depth']} {s['activation']}")
    return " | ".join(parts)
