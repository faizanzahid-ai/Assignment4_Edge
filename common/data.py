"""
CIFAR-10 / CIFAR-100 data handling.

The raw CIFAR "python version" pickles are read once and the whole dataset is
kept as a uint8 tensor on the training device.  Augmentation (random crop with
4px zero padding + horizontal flip), normalisation and optional resizing are
done batch-wise on the device.  This avoids the very slow multi-process
DataLoader start-up on Windows and keeps the (small) GPU fully busy.
"""
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Per-channel statistics of the training sets
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]


def cifar100_classes():
    return CIFAR100_CLASSES


def class_names(dataset):
    return CIFAR10_CLASSES if dataset == "cifar10" else cifar100_classes()


def _png_folders_to_npz(dataset):
    """Convert the PNG-folder copy of CIFAR (fast.ai mirror: <dataset>/{train,test}/<class>/*.png)
    into a compact .npz cache.  The label is the name of the folder containing the image."""
    from PIL import Image
    names = class_names(dataset)
    index = {n: i for i, n in enumerate(names)}
    out = {}
    for split in ("train", "test"):
        files = sorted((DATA_DIR / dataset / split).rglob("*.png"))
        xs = np.empty((len(files), 32, 32, 3), dtype=np.uint8)
        ys = np.empty(len(files), dtype=np.int64)
        for k, f in enumerate(files):
            xs[k] = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
            ys[k] = index[f.parent.name]
        out[f"x_{split}"] = xs.transpose(0, 3, 1, 2).copy()     # N,3,32,32
        out[f"y_{split}"] = ys
    np.savez(DATA_DIR / f"{dataset}.npz", **out)


def load_cifar(dataset):
    """Return numpy arrays (x_train, y_train, x_test, y_test); x is N x 3 x 32 x 32 uint8.

    Supported sources (first one found is used):
      1. data/<dataset>.npz                      - compact cache
      2. data/<dataset>/{train,test}/<class>/*.png (fast.ai mirror, converted to 1. automatically)
      3. the original Toronto "python version" pickles (cifar-10-batches-py / cifar-100-python)
    """
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists() and (DATA_DIR / dataset / "train").exists():
        _png_folders_to_npz(dataset)
    if npz.exists():
        d = np.load(npz)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    if dataset == "cifar10":
        base = DATA_DIR / "cifar-10-batches-py"
        batches = [_unpickle(base / f"data_batch_{i}") for i in range(1, 6)]
        x_tr = np.concatenate([b[b"data"] for b in batches])
        y_tr = np.concatenate([b[b"labels"] for b in batches])
        test = _unpickle(base / "test_batch")
        x_te, y_te = test[b"data"], np.array(test[b"labels"])
    elif dataset == "cifar100":
        base = DATA_DIR / "cifar-100-python"
        train, test = _unpickle(base / "train"), _unpickle(base / "test")
        x_tr, y_tr = train[b"data"], np.array(train[b"fine_labels"])
        x_te, y_te = test[b"data"], np.array(test[b"fine_labels"])
    else:
        raise ValueError(f"unknown dataset {dataset}")
    x_tr = x_tr.reshape(-1, 3, 32, 32)
    x_te = x_te.reshape(-1, 3, 32, 32)
    return x_tr, np.asarray(y_tr, dtype=np.int64), x_te, np.asarray(y_te, dtype=np.int64)


def normalize(x_float01, dataset):
    """Normalise a float tensor in [0,1] with the dataset statistics."""
    mean, std = STATS[dataset]
    mean = torch.tensor(mean, device=x_float01.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x_float01.device).view(1, 3, 1, 1)
    return (x_float01 - mean) / std


class DeviceLoader:
    """Mini-batch iterator over a dataset that lives entirely on `device`."""

    def __init__(self, x_uint8, y, dataset, batch_size=128, train=False,
                 device="cuda", resolution=32, drop_last=False):
        self.x = torch.as_tensor(x_uint8).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.resolution = resolution
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.y)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _augment(self, xb):
        # random crop (32x32 out of a 40x40 zero-padded image)
        b = xb.size(0)
        padded = F.pad(xb, (4, 4, 4, 4))
        i = torch.randint(0, 9, (b,), device=xb.device)
        j = torch.randint(0, 9, (b,), device=xb.device)
        ar = torch.arange(32, device=xb.device)
        rows = (i[:, None] + ar)[:, None, :, None]          # B,1,32,1
        cols = (j[:, None] + ar)[:, None, None, :]          # B,1,1,32
        bidx = torch.arange(b, device=xb.device)[:, None, None, None]
        cidx = torch.arange(3, device=xb.device)[None, :, None, None]
        xb = padded[bidx, cidx, rows, cols]
        # random horizontal flip
        flip = torch.rand(b, device=xb.device) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        return xb

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for k in range(len(self)):
            idx = order[k * self.batch_size:(k + 1) * self.batch_size]
            xb = self.x[idx].float().div_(255.0)
            if self.train:
                xb = self._augment(xb)
            if self.resolution != 32:
                xb = F.interpolate(xb, size=(self.resolution, self.resolution),
                                   mode="bilinear", align_corners=False)
            yield normalize(xb, self.dataset), self.y[idx]


def get_loaders(dataset, batch_size=128, device="cuda", resolution=32,
                train_subset=None, val_split=0, seed=0, test_subset=None):
    """
    Build train / evaluation loaders.

    val_split > 0 holds out that many *training* images as a validation set
    (used by NAS so the test set is never touched during the search); the
    second returned loader is then the validation loader instead of the test loader.
    """
    x_tr, y_tr, x_te, y_te = load_cifar(dataset)
    if val_split:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(y_tr))
        val_idx, tr_idx = perm[:val_split], perm[val_split:]
        x_te, y_te = x_tr[val_idx], y_tr[val_idx]
        x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    if train_subset:
        # random (but seeded) subset, used for fast/CPU runs and for NAS proxy training
        sub = np.random.RandomState(seed + 1).permutation(len(y_tr))[:train_subset]
        x_tr, y_tr = x_tr[sub], y_tr[sub]
    if test_subset:
        # class-balanced, seeded evaluation subset (fast CPU-only runs)
        sub = np.random.RandomState(seed + 2).permutation(len(y_te))[:test_subset]
        x_te, y_te = x_te[sub], y_te[sub]
    train = DeviceLoader(x_tr, y_tr, dataset, batch_size, True, device, resolution, drop_last=True)
    evaluation = DeviceLoader(x_te, y_te, dataset, 256, False, device, resolution)
    return train, evaluation


def preprocess_image(path, dataset, resolution=32):
    """Load an image file and convert it into a normalised 1x3xRxR numpy array."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    if resolution != 32:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return normalize(x, dataset).numpy().astype(np.float32)
