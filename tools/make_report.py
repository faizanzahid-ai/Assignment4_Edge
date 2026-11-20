"""
Builds the technical report docs/Technical_Report.pdf from the experiment outputs
(results/*.csv|json, logs) - figures, tables and terminal-output captures are regenerated.

Usage:
    python -m tools.make_report --name "Your Name" --roll 22i-0000 --section A
"""
import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
SHOT = ROOT / "docs" / "screenshots"


# ----------------------------------------------------------------------------- helpers
def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path, default=None):
    return json.load(open(path)) if Path(path).exists() else default


def fnum(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-" if v in (None, "") else str(v)


def terminal_capture(text, out, title):
    """Render console output as a terminal-style screenshot image."""
    lines = text.rstrip().splitlines()[-30:]
    lines = [ln if len(ln) <= 150 else ln[:147] + "..." for ln in lines]
    h = 0.55 + 0.122 * max(len(lines), 1)
    fig = plt.figure(figsize=(12, h), dpi=130)
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.18 / h, f"  {title}", color="#9cdcfe", family="monospace", fontsize=8.5,
             va="top", weight="bold")
    fig.text(0.01, 1 - 0.42 / h, "\n".join(lines), color="#d4d4d4", family="monospace",
             fontsize=6.4, va="top", linespacing=1.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def tail(path, n=30, contains=None):
    if not Path(path).exists():
        return ""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if contains:
        lines = [ln for ln in lines if any(c in ln for c in contains)]
    return "\n".join(lines[-n:])


# ----------------------------------------------------------------------------- figures
def fig_q1_curves():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, a in zip(axes, ("mobilenetv2", "squeezenet", "efficientnet_lite0")):
        for m, st in (("baseline", "--o"), ("kd", "-s")):
            h = read_json(RES / "q1" / f"history_{a}_{m}.json")
            if h:
                ax.plot([r["epoch"] for r in h], [r["test_top1"] for r in h], st, label=m.upper())
        ax.set_title(a)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test top-1 (%)")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "q1_curves.png", dpi=140)
    plt.close(fig)
    return FIG / "q1_curves.png"


def fig_q1_bars(bench):
    rows = [r for r in bench if r["dataset"] == "cifar100" and r["model"] != "teacher"]
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    names = [r["model"] for r in rows]
    col = ["#4c72b0" if "baseline" in n else "#dd8452" for n in names]
    ax[0].bar(names, [float(r["torch_top1"]) for r in rows], color=col)
    ax[0].set_ylabel("top-1 (%)")
    ax[0].set_title("CIFAR-100 accuracy (blue = baseline, orange = KD)")
    ax[1].bar(names, [float(r["onnx_latency_ms"] or 0) for r in rows], color=col)
    ax[1].set_ylabel("ms / image")
    ax[1].set_title("ONNX Runtime CPU latency (batch 1, 1 thread)")
    for a in ax:
        a.tick_params(axis="x", rotation=35, labelsize=7)
        a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "q1_bars.png", dpi=140)
    plt.close(fig)
    return FIG / "q1_bars.png"


def fig_nas(hist, best):
    if not hist:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    tr = [r for r in hist if r["status"] == "trained"]
    rj = [r for r in hist if r["status"] == "rejected"]
    ax[0].scatter([float(r["latency_ms"]) for r in rj], [float(r["size_mb"]) for r in rj], c="#c44e52",
                  marker="x", label=f"rejected ({len(rj)})")
    ax[0].scatter([float(r["latency_ms"]) for r in tr], [float(r["size_mb"]) for r in tr], c="#55a868",
                  label=f"feasible ({len(tr)})")
    if best:
        ax[0].scatter([best["search_metrics"]["latency_ms"]], [best["search_metrics"]["size_mb"]], s=160,
                      facecolors="none", edgecolors="k", label="best")
    ax[0].set_xlabel("measured CPU latency (ms)")
    ax[0].set_ylabel("model size (MB)")
    ax[0].set_yscale("log")
    ax[0].set_title("Hardware profile of all candidates")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ids = [int(r["id"]) for r in tr]
    acc = [float(r["val_acc"]) for r in tr]
    run_best, b = [], 0
    for a_ in acc:
        b = max(b, a_)
        run_best.append(b)
    ax[1].scatter(ids, acc, c="#55a868", s=18, label="candidate val acc")
    ax[1].plot(ids, run_best, "k-", label="best so far")
    ax[1].set_xlabel("candidate #")
    ax[1].set_ylabel("proxy val acc (%)")
    ax[1].set_title("Search progress")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "nas_search.png", dpi=140)
    plt.close(fig)
    return FIG / "nas_search.png"


# ----------------------------------------------------------------------------- report
def build(args):
    styles = getSampleStyleSheet()
    H1, H2, B = styles["Heading1"], styles["Heading2"], styles["BodyText"]
    B.fontSize, B.leading = 9.5, 13
    small = ParagraphStyle("small", parent=B, fontSize=8, leading=10)
    center = ParagraphStyle("c", parent=B, alignment=TA_CENTER)
    code = ParagraphStyle("code", parent=styles["Code"], fontSize=7.2, leading=9)

    def table(data, widths=None, fs=7.5):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4f6f")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), .3, colors.grey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f6")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def img(path, width=17 * cm):
        if not path or not Path(path).exists():
            return Spacer(1, 1)
        from PIL import Image as PI
        w, h = PI.open(path).size
        return Image(str(path), width=width, height=width * h / w)

    bench = read_csv(RES / "benchmark.csv")
    comp = read_csv(RES / "q1" / "kd_vs_baseline.csv")
    hist = read_csv(RES / "nas" / "search_history.csv")
    best = read_json(RES / "nas" / "best_arch.json")
    teacher_h = read_json(RES / "q1" / "history_teacher_resnet50.json", [])
    nas_h = read_json(RES / "nas" / "history_nas_best.json", [])
    import torch
    teacher_args = {}
    tpath = ROOT / "checkpoints" / "teacher_resnet50.pt"
    if tpath.exists():
        teacher_args = torch.load(tpath, map_location="cpu").get("train_args", {})
    kd_args = {}
    kpath = ROOT / "checkpoints" / "mobilenetv2_kd.pt"
    if kpath.exists():
        kd_args = torch.load(kpath, map_location="cpu").get("train_args", {})

    f_curves, f_bars, f_nas = fig_q1_curves(), fig_q1_bars(bench), fig_nas(hist, best)

    shots = [
        terminal_capture(tail(RES / "logs" / "teacher_resnet50.log", 8), SHOT / "01_teacher.png",
                         "python -m q1_kd.train_teacher"),
        terminal_capture(tail(RES / "logs" / "baseline_students.log", 12), SHOT / "02_baseline.png",
                         "python -m q1_kd.train_baseline --arch all"),
        terminal_capture(tail(RES / "logs" / "kd_students.log", 12), SHOT / "03_kd.png",
                         "python -m q1_kd.train_kd --arch all"),
        terminal_capture(tail(RES / "logs" / "nas_search.log", 26), SHOT / "04_nas_search.png",
                         "python -m q2_nas.evolution"),
        terminal_capture(tail(RES / "logs" / "nas_train_best.log", 8), SHOT / "05_nas_train_best.png",
                         "python -m q2_nas.train_best"),
    ]
    for fname, title in (("export_onnx.txt", "python -m inference.export_onnx --model all"),
                         ("onnx_infer.txt", "python -m inference.onnx_infer --model all"),
                         ("torch_infer.txt", "python -m inference.infer --model all"),
                         ("benchmark.txt", "python -m benchmark.benchmark")):
        p = RES / "logs" / fname
        if p.exists():
            shots.append(terminal_capture(p.read_text(encoding="utf-8", errors="ignore"),
                                          SHOT / (fname.replace(".txt", ".png")), title))

    s = []
    # ---------------- title page
    s += [Spacer(1, 4 * cm), Paragraph("Assignment # 04", styles["Title"]),
          Paragraph("Edge Computing &amp; Intelligent Systems", styles["Heading2"]),
          Spacer(1, .6 * cm),
          Paragraph("Technical Report: Knowledge Distillation &amp; Hardware-Aware Neural Architecture Search",
                    styles["Heading3"]),
          Spacer(1, 1.5 * cm),
          Paragraph(f"<b>Name:</b> {args.name}", center), Paragraph(f"<b>Roll No:</b> {args.roll}", center),
          Paragraph(f"<b>Section:</b> {args.section}", center), Spacer(1, 1.5 * cm),
          Paragraph(f"Platform: {platform.platform()} | Python {platform.python_version()} | "
                    f"PyTorch {torch.__version__}", small),
          PageBreak()]

    # ---------------- 1. overview
    s += [Paragraph("1. Overview", H1), Paragraph(
        "The repository implements two complete edge-AI pipelines: (Q1) an end-to-end knowledge-distillation "
        "(KD) framework that transfers knowledge from a ResNet-50 teacher into three lightweight students "
        "(MobileNetV2, SqueezeNet, EfficientNet-Lite0) on CIFAR-100, together with a no-KD baseline pipeline; "
        "and (Q2) a hardware-aware evolutionary Neural Architecture Search (NAS) on CIFAR-10 that measures "
        "real CPU latency, model size and peak memory, rejects architectures that violate deployment "
        "constraints, retrains the winner from scratch and deploys it. All trained models are exported to "
        "ONNX and served on the CPU with ONNX Runtime; an automated benchmark produces CSV results.", B),
        Spacer(1, 6), Paragraph("<b>Repository layout</b>", B),
        Preformatted(
            "common/          data loading (CIFAR on-device, GPU/CPU augmentation), training utilities\n"
            "q1_kd/models/    teacher.py (ResNet-50), mobilenetv2.py, squeezenet.py, efficientnet_lite.py\n"
            "q1_kd/kd.py      KD framework (soft targets, temperature softmax, hard+soft, feature hints)\n"
            "q1_kd/train_teacher.py | train_baseline.py | train_kd.py\n"
            "q2_nas/search_space.py | generator.py | hardware.py | evolution.py | train_best.py\n"
            "inference/       model_zoo.py, infer.py (unified PyTorch), export_onnx.py, onnx_infer.py\n"
            "benchmark/       benchmark.py -> results/benchmark.csv\n"
            "checkpoints/  onnx_models/  results/  docs/  samples/", code),
        Paragraph("<b>Compute note.</b> The experiments in this report were executed on a laptop CPU "
                  "(AMD Ryzen 5 5600H) because a CUDA build of PyTorch could not be installed in time. The code "
                  "is device-agnostic and uses CUDA + mixed precision automatically when available. To fit the "
                  "CPU budget the runs used reduced settings (training subsets / few epochs, a stride-2 teacher "
                  "stem and an evaluation subset of the CIFAR test set); all of these are command-line flags, "
                  "and the full-budget commands are listed in the README and in Section 6. Absolute accuracies "
                  "are therefore far below converged values; the pipelines, measurements and relative "
                  "comparisons are what this submission demonstrates.", B),
        PageBreak()]

    # ---------------- 2. Q1
    s += [Paragraph("2. Question 1 - Knowledge Distillation", H1),
          Paragraph("2.1 Teacher: ResNet-50 on CIFAR-100", H2), Paragraph(
              "torchvision's ResNet-50 is initialised with ImageNet weights. The ImageNet stem (7x7/2 conv + "
              "max-pool) would reduce a 32x32 image to 8x8 before the first stage, so it is replaced by a 3x3 "
              "convolution (initialised from the centre 3x3 of the pretrained 7x7 kernels) without max-pool. "
              "<i>stem_stride=1</i> keeps the stages at 32/16/8/4 (GPU setting); <i>stem_stride=2</i> is a 4x "
              "cheaper variant used for the CPU run. Training: SGD + Nesterov momentum, one-cycle LR, weight "
              "decay 5e-4, label smoothing 0.1, random-crop + flip augmentation. The layer3 and layer4 outputs "
              "are exposed as intermediate features for distillation.", B),
          Paragraph(f"Run configuration: {teacher_args}", small)]
    if teacher_h:
        s.append(table([["epoch", "train loss", "train acc %", "test top-1 %", "test top-5 %", "time (s)"]] +
                       [[r["epoch"], fnum(r["train_loss"], 3), fnum(r["train_acc"]), fnum(r["test_top1"]),
                         fnum(r["test_top5"]), fnum(r["epoch_time_s"], 0)] for r in teacher_h]))
    s += [Paragraph("2.2 Student architectures (implemented from scratch)", H2), table([
        ["Student", "Key building block", "CIFAR adaptation", "Mid / final feature"],
        ["MobileNetV2", "inverted residual + linear bottleneck, ReLU6", "stride-1 stem & stage 2 (total stride 8)",
         "96x8x8 / 1280x4x4"],
        ["SqueezeNet 1.1", "Fire module (squeeze 1x1, expand 1x1 + 3x3)", "3x3 stride-1 stem, BN, 3 max-pools",
         "384x8x8 / 512x4x4"],
        ["EfficientNet-Lite0", "MBConv (k3/k5), no SE, ReLU6, stochastic depth",
         "stride-1 stem & stage 2", "112x8x8 / 1280x4x4"]], [2.9 * cm, 6.3 * cm, 5.0 * cm, 3.0 * cm], fs=6.8),
          Paragraph("2.3 Distillation framework (q1_kd/kd.py)", H2), Paragraph(
              "The student minimises<br/>"
              "<b>L = (1 - &alpha;) CE(z<sub>s</sub>, y) + &alpha; T<super>2</super> KL(softmax(z<sub>t</sub>/T) || "
              "softmax(z<sub>s</sub>/T)) + &beta; &Sigma;<sub>i</sub> || n(&phi;<sub>i</sub>(F<sub>s</sub><super>i"
              "</super>)) - n(F<sub>t</sub><super>i</super>) ||<super>2</super></b><br/>"
              "&bull; <b>Soft-target KD</b>: the KL term matches the teacher's predictive distribution "
              "(teacher run in eval mode, no gradients).<br/>"
              "&bull; <b>Temperature softmax</b>: softmax(z/T) with T = 4 exposes the relative probabilities of "
              "wrong classes (\"dark knowledge\"); the T<super>2</super> factor keeps gradient scale constant.<br/>"
              "&bull; <b>Combined hard + soft labels</b>: &alpha; weights the soft term against the ground-truth "
              "cross-entropy (&alpha; = 0.7).<br/>"
              "&bull; <b>Feature-based KD</b> (FitNets-style hints): student mid-level and final feature maps are "
              "mapped by learnable 1x1 conv + BN connectors &phi;<sub>i</sub> to the channel count of teacher "
              "layer3 / layer4, spatially aligned by adaptive average pooling, L2-normalised per sample (n) and "
              "compared with a squared error (&beta; = 1). Connectors are trained jointly with the student and "
              "discarded at deployment.<br/>Each term can be switched off (--no-soft, --no-features) for ablations.", B),
          Paragraph(f"KD run configuration: {kd_args}", small),
          Paragraph("2.4 Results: baseline vs. distilled students", H2)]
    if comp:
        s.append(table([["Student", "Params (M)", "Baseline top-1 %", "KD top-1 %", "Gain (pts)", "Teacher top-1 %",
                         "Compression"]] +
                       [[r["student"], r["params_M"], r["baseline_top1"], r["kd_top1"], r["gain_pts"],
                         r["teacher_top1"], f"{r['compression_x']}x"] for r in comp]))
    s += [Spacer(1, 6), img(f_curves), img(f_bars), PageBreak()]

    # ---------------- 3. Q2
    s += [Paragraph("3. Question 2 - Hardware-Aware Neural Architecture Search", H1),
          Paragraph("3.1 Search space (q2_nas/search_space.py)", H2), table([
              ["Dimension", "Choices"],
              ["Depth", "2-4 stages x 1-3 blocks per stage (stage 1 keeps resolution, others stride 2)"],
              ["Channels", "stem 8/16/24/32; stage 16/24/32/48/64/96/128 (non-decreasing)"],
              ["Kernel size", "3, 5, 7 (per stage)"],
              ["Block type", "standard conv | depthwise-separable conv | inverted-residual MBConv (x4)"],
              ["Activation", "ReLU, ReLU6, Hardswish, SiLU, LeakyReLU (per stage)"],
              ["Input resolution", "20, 24, 28, 32 px"], ["Dropout", "0, 0.1, 0.2"]], [3.5 * cm, 13 * cm]),
          Paragraph("3.2 Generator and search strategy", H2), Paragraph(
              "<b>Generator</b> (q2_nas/generator.py) converts a JSON configuration into an nn.Module "
              "(stem -> stages of the chosen blocks with residuals where shapes match -> global pooling -> "
              "classifier). <b>Search</b> (q2_nas/evolution.py) is regularized (aging) evolution: an initial "
              "population of feasible random architectures; each cycle a tournament selects a parent, one gene "
              "is mutated, the child is evaluated and the oldest individual is removed. Candidates are "
              "proxy-trained on a CIFAR-10 training subset and validated on images held out from the training "
              "set (the test set is never used during search).", B),
          Paragraph("3.3 Hardware-aware selection (q2_nas/hardware.py)", H2), Paragraph(
              "Before any training, each candidate is profiled on the CPU: <b>latency</b> = median of 30 batch-1 "
              "forward passes with one intra-op thread (after warm-up); <b>storage size</b> = bytes of the "
              "serialised state_dict; <b>peak inference memory</b> = peak bytes of the PyTorch CPU allocator during "
              "one forward pass, recorded from torch.profiler memory events (CUDA allocator fallback, None if "
              "unsupported). A candidate violating any constraint is <b>rejected</b> and logged with the reason. "
              "Feasible candidates are ranked by a MnasNet-style reward: fitness = acc x (lat/LAT)<super>w"
              "</super> x (size/SIZE)<super>w</super>, w = -0.07.", B)]
    if hist:
        n_rej = sum(r["status"] == "rejected" for r in hist)
        s.append(Paragraph(f"<b>Search statistics:</b> {len(hist)} candidates evaluated, {len(hist) - n_rej} "
                           f"trained, {n_rej} rejected by deployment constraints (latency &le; 3 ms, size &le; 1 MB, "
                           f"peak memory &le; 4 MB).", small))
    if best:
        m = best["search_metrics"]
        s += [Paragraph("3.4 Best discovered architecture", H2),
              Paragraph(f"<b>{best['summary']}</b>", B),
              table([["proxy val acc %", "fitness", "latency (ms)", "size (MB)", "peak mem (MB)", "params"],
                     [m["val_acc"], m["fitness"], m["latency_ms"], m["size_mb"], m["peak_mem_mb"], m["params"]]]),
              Preformatted(json.dumps(best["config"], indent=1)[:1500], code)]
    s.append(img(f_nas))
    top = sorted([r for r in hist if r["status"] == "trained"], key=lambda r: -float(r["fitness"]))[:8]
    if top:
        s += [Paragraph("Top feasible candidates", H2),
              table([["#", "architecture", "val %", "fitness", "ms", "MB", "mem MB"]] +
                    [[r["id"], Paragraph(r["arch"], small), r["val_acc"], r["fitness"], r["latency_ms"], r["size_mb"],
                      r["peak_mem_mb"]] for r in top], [0.8 * cm, 9.4 * cm, 1.3 * cm, 1.4 * cm, 1.1 * cm, 1.2 * cm,
                                                         1.3 * cm])]
    if nas_h:
        s += [Paragraph("3.5 Training the best architecture from scratch (full CIFAR-10 test set)", H2),
              table([["epoch", "train loss", "train acc %", "test top-1 %", "time (s)"]] +
                    [[r["epoch"], fnum(r["train_loss"], 3), fnum(r["train_acc"]), fnum(r["test_top1"]),
                      fnum(r["epoch_time_s"], 0)] for r in nas_h])]
    s.append(PageBreak())

    # ---------------- 4. deployment + benchmark
    s += [Paragraph("4. Inference, ONNX deployment and benchmarking", H1), Paragraph(
        "<b>Unified inference</b> (inference/infer.py): a single Predictor class rebuilds any student or the NAS "
        "model from its checkpoint (model_zoo.py), applies the dataset-specific preprocessing and returns top-k "
        "labels. <b>ONNX</b> (inference/export_onnx.py): every model is exported with opset 17 and a dynamic batch "
        "axis; the ONNX Runtime output is verified against PyTorch. <b>ONNX Runtime</b> (inference/onnx_infer.py) "
        "needs only numpy, Pillow and onnxruntime and runs on the CPUExecutionProvider. <b>Benchmark</b> "
        "(benchmark/benchmark.py) records accuracy (PyTorch and ONNX), batch-1 latency with one thread, "
        "ONNX batch-32 throughput, sizes and peak memory into results/benchmark.csv.", B), Spacer(1, 6)]
    if bench:
        s.append(table([["model", "data", "params M", "ckpt MB", "onnx MB", "torch top-1", "onnx top-1",
                         "torch ms", "onnx ms", "onnx img/s (b32)", "peak mem MB"]] +
                       [[r["model"], r["dataset"], fnum(r["params_M"], 3), fnum(r["ckpt_size_mb"]),
                         fnum(r.get("onnx_size_mb")), fnum(r["torch_top1"]), fnum(r.get("onnx_top1")),
                         fnum(r["torch_latency_ms"]), fnum(r.get("onnx_latency_ms")),
                         fnum(r.get("onnx_throughput_ips_b32"), 0), fnum(r["peak_mem_mb"])] for r in bench],
                       fs=6.6))
        s.append(Paragraph("Accuracy columns use the same evaluation subset of the test set for PyTorch and ONNX; "
                           "identical values confirm a lossless export. Latency: median, batch 1, 1 CPU thread.",
                           small))
    s.append(PageBreak())

    # ---------------- 5. screenshots
    s.append(Paragraph("5. Screenshots (terminal output of the executed pipelines)", H1))
    for p in shots:
        s += [img(p), Spacer(1, 8)]
    s.append(PageBreak())

    # ---------------- 6. discussion
    s += [Paragraph("6. Discussion, limitations and reproduction", H1), Paragraph(
        "&bull; The KD pipeline trains students with the same seed, data order and schedule as the baselines, so "
        "the KD-vs-baseline gap isolates the effect of distillation. <b>Observed result:</b> under the tiny CPU "
        "budget (2 epochs on 6,000 images, about 94 optimizer steps) the KD students scored "
        "2-4 points <i>below</i> their baselines for all three architectures. This is expected in this regime: (i) the teacher itself only reached ~39% top-1 after its "
        "short fine-tuning, so its soft targets carry limited dark knowledge; (ii) with alpha = 0.7 only 30% of the "
        "loss weight is on the ground-truth labels, and the feature-hint term (which starts near its maximum of 4) "
        "dominates the first epochs - the student spends its few updates aligning features with the "
        "(randomly initialised) connectors instead of fitting labels. KD acts as a regulariser whose benefit appears "
        "with longer schedules; with the full budget (teacher ~75-80%, 30 student epochs) KD typically improves "
        "CIFAR-100 students by 1-4 points. A lower beta or a warm-up of the hint term is recommended for short "
        "schedules.<br/>"
        "&bull; NAS latency is measured on the actual target CPU, so the constraint check reflects real deployment "
        "cost rather than FLOPs. Measurements taken while other training jobs ran in parallel are slightly "
        "inflated; the final benchmark was run with the machine otherwise idle.<br/>"
        "&bull; Full-budget reproduction (GPU recommended):", B),
        Preformatted(
            "python -m q1_kd.train_teacher  --epochs 15 --lr 0.05\n"
            "python -m q1_kd.train_baseline --arch all --epochs 30\n"
            "python -m q1_kd.train_kd       --arch all --epochs 30 --T 4 --alpha 0.7 --beta 1.0\n"
            "python -m q2_nas.evolution     --population 10 --cycles 40 --proxy-epochs 3 --proxy-train 20000\n"
            "python -m q2_nas.train_best    --epochs 30\n"
            "python -m inference.export_onnx --model all && python -m benchmark.benchmark --limit 0", code)]

    out = ROOT / "docs" / "Technical_Report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Assignment 4 Technical Report")
    doc.build(s)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="&lt;Your Name&gt;")
    ap.add_argument("--roll", default="&lt;ROLL-NUM&gt;")
    ap.add_argument("--section", default="&lt;SECTION&gt;")
    build(ap.parse_args())
