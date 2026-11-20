"""
Export trained student models (Q1.7) and the NAS model (Q2.8) to ONNX.

Each model gets a dynamic batch dimension, input name 'input' and output name 'logits'.
After export the ONNX graph is checked and its output is compared against PyTorch.

Usage:
    python -m inference.export_onnx --model all
"""
import argparse
import json

import numpy as np
import torch

from inference.model_zoo import ALL_MODELS, ONNX_DIR, load_model

try:
    import onnx  # noqa: F401  (optional: only used for the graph checker)
except ImportError:
    # The TorchScript exporter imports the `onnx` package only to embed onnxscript custom
    # functions, which these models do not use.  On machines where `onnx` cannot be
    # installed/loaded (e.g. Windows Smart App Control blocks its DLLs) that step is skipped;
    # the serialized ModelProto is still produced by PyTorch's C++ exporter.
    from torch.onnx._internal.torchscript_exporter import onnx_proto_utils, utils as _ts_utils
    _ts_utils.onnx_proto_utils._add_onnxscript_fn = lambda proto, custom_opsets: proto
    onnx_proto_utils._add_onnxscript_fn = lambda proto, custom_opsets: proto


def export(name, opset=17):
    model, meta = load_model(name)
    r = meta["resolution"]
    dummy = torch.randn(1, 3, r, r)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    path = ONNX_DIR / f"{name}.onnx"
    torch.onnx.export(model, (dummy,), str(path), dynamo=False, opset_version=opset,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      do_constant_folding=True)
    # metadata needed by the ONNX Runtime pipeline (no torch required at inference time)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(path)))
    except ImportError:
        pass
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, r, r)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    err = float(np.abs(ref - out).max())
    print(f"{name:<28} -> {path.name:<32} {path.stat().st_size / 1e6:7.3f} MB | max |torch-ort| = {err:.2e}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Export trained models to ONNX")
    ap.add_argument("--model", default="all")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    for name in (ALL_MODELS if args.model == "all" else [args.model]):
        export(name, args.opset)


if __name__ == "__main__":
    main()
