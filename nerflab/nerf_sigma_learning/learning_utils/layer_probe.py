# layer_probe.py
# ---------------------------------------------------------------------
# Flexible layer statistics probes for PyTorch
#
# Features:
#   • capture = "input" | "output" | "both"
#   • hooks leaf modules by default; also hooks the root if it's a leaf
#   • bounded per-layer history
#   • robust stats: percentiles, zero/NaN/Inf rates, histogram snapshot
#   • sampling for large tensors (keeps percentiles/hists tractable)
#   • enable/disable, print throttling, clean unregister
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Union, Literal
from collections import defaultdict, deque
import math

import torch
import torch.nn as nn

CaptureMode = Literal["input", "output", "both"]
SampleMode = Literal["uniform", "random"]


# ----------------------------- Stats containers ------------------------------

@dataclass
class StatRecord:
    """One snapshot of stats for either the input or output of a layer."""
    which: Literal["input", "output"]
    shape: Tuple[int, ...]
    min: float
    p1: float
    p5: float
    mean: float
    std: float
    p95: float
    p99: float
    max: float
    frac_zero: float
    frac_nan: float
    frac_inf: float
    # Histogram snapshot (optional)
    hist_edges: Optional[List[float]] = None   # len = bins + 1
    hist_counts: Optional[List[int]] = None    # len = bins


# ----------------------------- Helper functions ------------------------------

def _percentiles(x: torch.Tensor, qs: Tuple[int, ...] = (1, 5, 50, 95, 99)) -> List[float]:
    """
    Compute percentiles on a finite-only vector (1-D float tensor).
    Returns NaNs if no finite values.
    """
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    # ignore NaNs/Infs just in case
    xf = x[torch.isfinite(x)]
    if xf.numel() == 0:
        return [float("nan")] * len(qs)
    q = torch.tensor(qs, device=xf.device, dtype=torch.float32) / 100.0
    vals = torch.quantile(xf.float(), q)
    return [float(v.item()) for v in vals]


def _tiny_hist(x: torch.Tensor, bins: int = 50) -> Tuple[List[float], List[int]]:
    """
    Build a simple histogram on finite values of a 1-D float tensor.
    Returns (edges, counts) as Python lists.
    """
    x = x.float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return [0.0, 1.0], [0]
    lo, hi = x.min().item(), x.max().item()
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        # create a non-degenerate range
        hi = lo + 1.0
    counts_t = torch.histc(x, bins=bins, min=lo, max=hi).to(torch.int64)
    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    counts = counts_t.tolist()
    return edges, counts


# --------------------------------- Probe -------------------------------------

class LayerProbe:
    """
    LayerProbe
    ----------
    Attach forward hooks to chosen modules and collect statistics with options:
      • capture: "input" | "output" | "both"
      • live printing: print_every = 0 (off) or N to print every N-th call (per tag)
      • bounded history per layer (history_max)
      • histogram snapshot (bins configurable)
      • sampling for very large tensors (max_elems, sample_mode)
      • enable()/disable(), register_on()/unregister(), clear()

    Typical use:
    >>> probe = LayerProbe(capture="both", print_every=0, history_max=3, bins=64)
    >>> probe.register_on(model, predicate=lambda n,m: isinstance(m, (nn.Linear, nn.Conv2d, nn.ReLU)))
    >>> _ = model(torch.randn(1024, 3))
    >>> print(probe.table())                     # compact text summary (latest per tag)
    >>> probe.plot_hist("layers.2:Linear", which="output")   # optional (needs matplotlib)

    Notes:
    - Inputs in forward hooks arrive as a tuple; we take the first tensor by default.
    - Outputs may be a Tensor or a tuple. Tuple handling is optional (see __init__ flag).
    - For memory safety, tensors are detached and moved to CPU for stats.
    """

    def __init__(
        self,
        capture: CaptureMode = "output",
        print_every: int = 0,
        history_max: int = 2,
        bins: int = 50,
        enabled: bool = True,
        # Sampling to avoid huge-vec quantile/hist failures:
        max_elems: int = 250_000,
        sample_mode: SampleMode = "uniform",
        # Output handling:
        handle_tuple_outputs: bool = False,  # if True, collect stats from all Tensor outputs in tuples
    ):
        self.capture: CaptureMode = capture
        self.print_every: int = max(0, int(print_every))
        self.history_max: int = max(1, int(history_max))
        self.bins: int = max(2, int(bins))
        self.enabled: bool = enabled

        self.max_elems: int = max(10_000, int(max_elems))
        self.sample_mode: SampleMode = sample_mode
        self.handle_tuple_outputs: bool = handle_tuple_outputs

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # tag -> deque[StatRecord]
        self.data: Dict[str, deque[StatRecord]] = defaultdict(lambda: deque(maxlen=self.history_max))
        # tag counters (for throttled printing)
        self._counts: Dict[str, int] = defaultdict(int)

    # ---------- public API ----------

    def enable(self) -> None:
        """Enable data collection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable data collection (hooks remain attached)."""
        self.enabled = False

    def clear(self) -> None:
        """Clear collected stats only (keeps hooks)."""
        self.data.clear()
        self._counts.clear()

    def unregister(self) -> None:
        """Remove all registered hooks (does not clear collected data)."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def register_on(
        self,
        root: nn.Module,
        predicate: Optional[Callable[[str, nn.Module], bool]] = None,
        tag_fmt: str = "{name}:{cls}",
    ) -> None:
        """
        Attach forward hooks to modules under `root` that satisfy `predicate`.
        Default predicate hooks *leaf modules* (no children).
        Also hooks the root itself if it is a leaf (covers single-module models).

        Args:
            root:      Root module to scan.
            predicate: Callable (name, module) -> bool. If None, hooks leaves.
            tag_fmt:   Format string for display tags. Exposes {name}, {cls}.
        """
        if predicate is None:
            predicate = lambda name, m: len(list(m.children())) == 0  # leaf modules

        for name, module in root.named_modules():
            # name is "" for root; give it a readable tag
            pretty_name = name or "root"
            if predicate(pretty_name, module):
                tag = tag_fmt.format(name=pretty_name, cls=module.__class__.__name__)
                handle = module.register_forward_hook(self._make_hook(tag))
                self._handles.append(handle)

    def latest(self, tag: str) -> Optional[StatRecord]:
        """Return the latest StatRecord for a given tag (or None)."""
        dq = self.data.get(tag)
        return dq[-1] if dq and len(dq) > 0 else None

    def history(self, tag: str) -> List[StatRecord]:
        """Return the full (bounded) history list for a tag."""
        dq = self.data.get(tag, [])
        return list(dq)

    def keys(self) -> List[str]:
        """Convenience: current list of tags with data."""
        return sorted(self.data.keys())

    def table(self) -> str:
        """Compact single-line summary per layer (latest record only)."""
        lines = []
        for tag in sorted(self.data.keys()):
            rec = self.data[tag][-1]
            lines.append(
                f"{tag:<28} [{rec.which:^6}] shape={str(rec.shape):<14} "
                f"min={rec.min:+.4f} p5={rec.p5:+.4f} mean={rec.mean:+.4f} "
                f"std={rec.std:+.4f} p95={rec.p95:+.4f} max={rec.max:+.4f} "
                f"zero={rec.frac_zero*100:5.1f}% nan={rec.frac_nan*100:4.1f}% inf={rec.frac_inf*100:4.1f}%"
            )
        return "\n".join(lines)

    def plot_hist(self, tag: str, which: Literal["input", "output"] = "output") -> None:
        """
        Plot the latest histogram for a given tag/stream.
        Requires matplotlib. No-op if no record/hist exists.
        """
        rec = self._latest_by_stream(tag, which)
        if rec is None or rec.hist_edges is None or rec.hist_counts is None:
            print(f"[LayerProbe] No histogram for {tag} ({which}).")
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[LayerProbe] matplotlib not available.")
            return

        edges = rec.hist_edges
        counts = rec.hist_counts
        centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]
        plt.figure()
        plt.bar(centers, counts, width=(edges[1] - edges[0]))
        plt.title(f"{tag} — {which} histogram (bins={len(counts)})")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.show()

    # ---------- internals ----------

    def _latest_by_stream(self, tag: str, which: Literal["input", "output"]) -> Optional[StatRecord]:
        dq = self.data.get(tag, [])
        for rec in reversed(dq):
            if rec.which == which:
                return rec
        return None

    def _make_hook(self, tag: str) -> Callable:
        def _hook(module: nn.Module, inp: Tuple, out: Union[torch.Tensor, Tuple, None]):
            if not self.enabled:
                return

            self._counts[tag] += 1
            should_print = self.print_every > 0 and (self._counts[tag] % self.print_every == 0)

            # --- small utilities local to the hook ---

            def _to_cpu_tensor(t: torch.Tensor) -> torch.Tensor:
                # Detach and move to CPU for lightweight stats; handle sparse
                if t.is_sparse:
                    t = t.coalesce().values()
                return t.detach().to("cpu")

            def _maybe_sample(x: torch.Tensor) -> torch.Tensor:
                # Sample large vectors to keep quantile/hist stable/fast.
                n = x.numel()
                if n <= self.max_elems:
                    return x
                if self.sample_mode == "random":
                    idx = torch.randperm(n)[: self.max_elems]
                    return x.index_select(0, idx)
                # uniform stride (deterministic)
                step = max(1, math.ceil(n / self.max_elems))
                return x[::step][: self.max_elems]

            def _collect(t: torch.Tensor, which_stream: Literal["input", "output"]) -> StatRecord:
                t = _to_cpu_tensor(t)
                x = t.reshape(-1)

                n = x.numel()
                if n == 0:
                    return StatRecord(
                        which=which_stream, shape=tuple(t.shape),
                        min=float("nan"), p1=float("nan"), p5=float("nan"),
                        mean=float("nan"), std=float("nan"),
                        p95=float("nan"), p99=float("nan"), max=float("nan"),
                        frac_zero=float("nan"), frac_nan=float("nan"), frac_inf=float("nan"),
                        hist_edges=None, hist_counts=None
                    )

                finite_mask = torch.isfinite(x)
                inf_mask = torch.isinf(x)
                nan_mask = torch.isnan(x)

                xf = x[finite_mask]
                if xf.numel() == 0:
                    # No finite values → only rates are meaningful
                    mean = std = lo = hi = float("nan")
                    p1 = p5 = p95 = p99 = float("nan")
                    frac_zero = 0.0
                    edges = counts = None
                else:
                    # Compute descriptive stats on a sampled subset of finite values
                    xf_s = _maybe_sample(xf)

                    mean = float(xf_s.mean().item())
                    std  = float(xf_s.std(unbiased=False).item())
                    lo   = float(xf_s.min().item())
                    hi   = float(xf_s.max().item())

                    p1, p5, _p50, p95, p99 = _percentiles(xf_s, (1, 5, 50, 95, 99))
                    frac_zero = float(((xf == 0).sum().item()) / max(1, xf.numel()))

                    edges, counts = _tiny_hist(xf_s, bins=self.bins)

                # Rates over the FULL vector (not sampled)
                frac_nan = float(nan_mask.sum().item() / n)
                frac_inf = float(inf_mask.sum().item() / n)

                return StatRecord(
                    which=which_stream, shape=tuple(t.shape),
                    min=lo, p1=p1, p5=p5, mean=mean, std=std,
                    p95=p95, p99=p99, max=hi,
                    frac_zero=frac_zero, frac_nan=frac_nan, frac_inf=frac_inf,
                    hist_edges=edges, hist_counts=counts
                )

            # --- gather per capture mode ---
            recs: List[StatRecord] = []

            if self.capture in ("input", "both"):
                if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    recs.append(_collect(inp[0], which_stream="input"))

            if self.capture in ("output", "both"):
                if isinstance(out, torch.Tensor):
                    recs.append(_collect(out, which_stream="output"))
                elif self.handle_tuple_outputs and isinstance(out, tuple):
                    # Collect stats from all Tensor elements in output tuples
                    for i, t in enumerate(out):
                        if isinstance(t, torch.Tensor):
                            recs.append(_collect(t, which_stream="output"))

            # Append & optionally print
            for rec in recs:
                self.data[tag].append(rec)
                if should_print:
                    print(self._format_line(tag, rec))

        return _hook

    def _format_line(self, tag: str, r: StatRecord) -> str:
        return (
            f"{tag:<28} [{r.which:^6}] shape={str(r.shape):<14} "
            f"min={r.min:+.4f} p1={r.p1:+.4f} p5={r.p5:+.4f} "
            f"mean={r.mean:+.4f} std={r.std:+.4f} p95={r.p95:+.4f} p99={r.p99:+.4f} "
            f"max={r.max:+.4f} zero={r.frac_zero*100:5.1f}% "
            f"nan={r.frac_nan*100:4.1f}% inf={r.frac_inf*100:4.1f}%"
        )
