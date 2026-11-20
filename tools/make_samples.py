"""Write a few CIFAR test images (one per class, distinct classes) as PNG files into
samples/<dataset>/ for the inference demos."""
from PIL import Image

from common.data import class_names, load_cifar
from common.train_utils import ROOT

N = 6
for ds in ("cifar10", "cifar100"):
    _, _, x, y = load_cifar(ds)
    names = class_names(ds)
    out = ROOT / "samples" / ds
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    seen = set()
    for i in range(len(y)):
        if y[i] in seen:
            continue
        seen.add(y[i])
        Image.fromarray(x[i].transpose(1, 2, 0)).resize((96, 96), Image.NEAREST).save(
            out / f"test{i:04d}_{names[y[i]]}.png")
        if len(seen) == N:
            break
    print(f"wrote {N} samples to {out}")
