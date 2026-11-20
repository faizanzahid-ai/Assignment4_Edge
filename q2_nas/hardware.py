"""
Q2.5 / Q2.6 - Hardware-aware profiling and deployment constraints.

For every candidate we *measure* (no proxies / FLOP estimates):
  * CPU inference latency  - median wall-clock time of a batch-1 forward pass,
                             fixed number of intra-op threads to emulate an edge CPU
  * model storage size     - number of bytes of the serialised state_dict
  * peak inference memory  - peak CPU memory allocated by the PyTorch allocator during one
                             forward pass (torch.profiler memory events); falls back to the
                             CUDA allocator peak, or None when not supported
Candidates violating any constraint are rejected *before* training.
"""
import io
import statistics
import time

import torch

DEFAULT_CONSTRAINTS = {
    "max_latency_ms": 4.0,      # batch-1, single-thread CPU latency
    "max_size_mb": 1.0,         # serialized fp32 weights
    "max_peak_mem_mb": 8.0,     # peak activation + workspace memory of one inference
}


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def measure_latency_ms(model, resolution, threads=1, warmup=5, runs=30):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(1, 3, resolution, resolution)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(prev)
    return statistics.median(times)


@torch.no_grad()
def measure_peak_memory_mb(model, resolution):
    """Peak bytes held by the CPU allocator during a single forward pass."""
    model = model.cpu().eval()
    x = torch.randn(1, 3, resolution, resolution)
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model(x)
        # "[memory]" events carry +bytes for allocations and -bytes for frees, in time order
        events = [e for e in prof.events() if e.name == "[memory]"]
        events.sort(key=lambda e: e.time_range.start)
        cur = peak = 0
        for e in events:
            cur += e.cpu_memory_usage
            peak = max(peak, cur)
        if peak > 0:
            return peak / 1e6
        # older versions: fall back to per-op allocation totals
        peak = max((e.cpu_memory_usage for e in prof.key_averages()), default=0)
        return peak / 1e6 if peak > 0 else None
    except Exception:
        if torch.cuda.is_available():
            m = model.cuda()
            torch.cuda.reset_peak_memory_stats()
            m(x.cuda())
            peak = torch.cuda.max_memory_allocated() / 1e6
            model.cpu()
            return peak
        return None


def profile_model(model, resolution, threads=1):
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "size_mb": model_size_mb(model),
        "latency_ms": measure_latency_ms(model, resolution, threads),
        "peak_mem_mb": measure_peak_memory_mb(model, resolution),
    }


def check_constraints(profile, constraints=DEFAULT_CONSTRAINTS):
    """Return a list of violated constraints (empty list = deployable)."""
    v = []
    if profile["latency_ms"] > constraints["max_latency_ms"]:
        v.append(f"latency {profile['latency_ms']:.2f}ms > {constraints['max_latency_ms']}ms")
    if profile["size_mb"] > constraints["max_size_mb"]:
        v.append(f"size {profile['size_mb']:.3f}MB > {constraints['max_size_mb']}MB")
    pm = profile.get("peak_mem_mb")
    if pm is not None and pm > constraints["max_peak_mem_mb"]:
        v.append(f"peak mem {pm:.2f}MB > {constraints['max_peak_mem_mb']}MB")
    return v
