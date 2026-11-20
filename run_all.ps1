# End-to-end pipeline (Windows PowerShell).  Run from the repository root.
# The flags below are the reduced CPU budget used for the submitted results;
# remove --train-subset/--test-subset and raise --epochs for full training (see README).
$ErrorActionPreference = "Stop"

# ---------- Question 1: knowledge distillation (CIFAR-100)
python -m q1_kd.train_teacher  --epochs 2 --lr 0.05 --stem-stride 2 --train-subset 6000 --test-subset 2000 --threads 4
python -m q1_kd.train_baseline --arch all --epochs 2 --train-subset 6000 --test-subset 2000 --threads 4
python -m q1_kd.train_kd       --arch all --epochs 2 --train-subset 6000 --test-subset 2000 --threads 4

# ---------- Question 2: hardware-aware NAS (CIFAR-10)
python -m q2_nas.evolution  --population 6 --cycles 14 --proxy-epochs 1 --proxy-train 6000 --proxy-val 1000 --threads 2
python -m q2_nas.train_best --epochs 4 --train-subset 25000 --threads 2

# ---------- Deployment, inference and benchmarking
python -m tools.make_samples
python -m inference.infer --model all
python -m inference.export_onnx --model all
python -m inference.onnx_infer --model all
python -m benchmark.benchmark --limit 2000
python -m tools.make_report
