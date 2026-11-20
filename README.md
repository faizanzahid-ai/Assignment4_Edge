# Assignment 4 — Edge Computing & Intelligent Systems

**Q1:** Knowledge distillation from a ResNet-50 teacher into MobileNetV2, SqueezeNet and EfficientNet-Lite0 (CIFAR-100).
**Q2:** Hardware-aware evolutionary Neural Architecture Search (CIFAR-10).

Every trained model is exported to ONNX and run on the CPU with ONNX Runtime. An automated benchmark compares all of them. The technical report is in [docs/Technical_Report.pdf](docs/Technical_Report.pdf).

## Repository structure

| Path | Contents (deliverable #) |
|---|---|
| `common/data.py` | CIFAR-10/100 loading. The dataset is kept on the device and augmentation (crop and flip) runs batch-wise; the input resolution is variable. |
| `common/train_utils.py` | Training loop (SGD+Nesterov, one-cycle LR, AMP on GPU), evaluation, checkpoints, logging |
| `q1_kd/models/teacher.py` | ResNet-50 teacher with a CIFAR stem (#1) |
| `q1_kd/models/{mobilenetv2,squeezenet,efficientnet_lite}.py` | Student architectures, implemented from scratch (#2) |
| `q1_kd/train_teacher.py` | Teacher training script (#1) |
| `q1_kd/train_baseline.py` | Baseline students trained without KD (#3) |
| `q1_kd/kd.py`, `q1_kd/train_kd.py` | KD framework: soft targets, temperature softmax, combined hard and soft labels, feature hints; distillation training (#4) |
| `q2_nas/search_space.py`, `q2_nas/generator.py` | Search space and architecture generator (#5) |
| `q2_nas/evolution.py` | Regularized-evolution NAS (#6) |
| `q2_nas/hardware.py` | CPU latency, storage size and peak memory measurement; constraint checks |
| `q2_nas/train_best.py` | Retrains the best architecture from scratch (#8) |
| `results/nas/` | `best_arch.json`, `search_history.csv`, `search_history.json` (#7) |
| `inference/infer.py` | Unified PyTorch inference for all students and the NAS model (#9) |
| `inference/export_onnx.py`, `inference/onnx_infer.py` | ONNX export and ONNX Runtime CPU inference (#10) |
| `benchmark/benchmark.py` | Automated benchmark, writes `results/benchmark.csv` and `results/q1/kd_vs_baseline.csv` (#11) |
| `checkpoints/` | Teacher, 3 baseline students, 3 KD students, NAS best (`*.pt`) |
| `onnx_models/` | Exported `.onnx` models and their metadata `.json` |
| `results/logs/` | Training, search and inference logs |
| `docs/` | Technical report PDF (#12), figures, terminal screenshots |

## Installation

```bash
# Python 3.10 – 3.13 (onnxruntime has no Python 3.15 wheels yet)
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
# CPU-only PyTorch; for an NVIDIA GPU pick the CUDA build at https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Dataset

Put CIFAR in `data/`. The loader accepts any one of these three layouts:

1. The original "python version" archives, extracted:
   - `data/cifar-10-batches-py/`
   - `data/cifar-100-python/`
2. The fast.ai PNG mirror, extracted:
   - `data/cifar10/{train,test}/<class>/*.png`
   - `data/cifar100/{train,test}/<class>/*.png`
   - Download from `https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz` and `.../cifar100.tgz`.
3. A `data/cifar10.npz` / `data/cifar100.npz` cache. This is created automatically the first time layout 2 is read.

## Step-by-step execution

Run every command from the repository root. Each step writes its logs to `results/logs/`.

```bash
# ---- Q1: knowledge distillation (CIFAR-100)
python -m q1_kd.train_teacher  --epochs 15 --lr 0.05                 # -> checkpoints/teacher_resnet50.pt
python -m q1_kd.train_baseline --arch all --epochs 30                # -> checkpoints/<student>_baseline.pt
python -m q1_kd.train_kd       --arch all --epochs 30 --T 4 --alpha 0.7 --beta 1.0   # -> <student>_kd.pt
#   ablations: --no-features (logit KD only), --no-soft (feature hints + CE only)

# ---- Q2: hardware-aware NAS (CIFAR-10)
python -m q2_nas.evolution --population 10 --cycles 40 --proxy-epochs 3 --proxy-train 20000 \
       --max-latency-ms 3 --max-size-mb 1 --max-peak-mem-mb 4      # -> results/nas/best_arch.json
python -m q2_nas.train_best --epochs 30                             # -> checkpoints/nas_best.pt

# ---- inference, deployment, benchmark
python -m tools.make_samples                        # sample PNGs in samples/
python -m inference.infer --model all               # unified PyTorch inference
python -m inference.infer --model mobilenetv2_kd --images samples/cifar100/*.png --evaluate
python -m inference.export_onnx --model all         # -> onnx_models/*.onnx
python -m inference.onnx_infer --model all          # ONNX Runtime, CPU
python -m inference.onnx_infer --model nas_best --images my_photo.png --evaluate
python -m benchmark.benchmark --limit 0             # -> results/benchmark.csv (0 = full test set)
python -m tools.make_report --name "..." --roll ... --section ...   # -> docs/Technical_Report.pdf
```

Model names for `--model` are:
- `mobilenetv2_baseline`, `mobilenetv2_kd`
- `squeezenet_baseline`, `squeezenet_kd`
- `efficientnet_lite0_baseline`, `efficientnet_lite0_kd`
- `nas_best`
- `all`

### Settings used for the submitted results

The submitted results were produced on a laptop CPU (Ryzen 5 5600H), with no GPU build of PyTorch. To fit that budget, the runs used:
- the reduced flags in `run_all.ps1`, i.e. training subsets and 1–4 epochs
- a stride-2 teacher stem
- 2,000 test images for evaluating the Q1 models

All of these are command-line options. The code automatically uses CUDA and mixed precision when a GPU is available, so the full-budget commands above reproduce converged numbers.

> Note: on Windows machines with Smart App Control enabled, the DLLs of `onnx`/`ml_dtypes`/`pyarrow` can be blocked.
> The `onnx` package is optional: `export_onnx.py` detects when it is missing and exports through PyTorch's built-in serializer anyway.
