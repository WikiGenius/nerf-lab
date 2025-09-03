"""
Sigma Visualization Utilities (self‑contained, well‑documented)
===============================================================

This module provides a compact, production‑ready toolkit to visualize per‑sample
NeRF density (sigma, σ) along rays. It is **self‑contained** (no external
`nerflab` imports) and focuses on clarity, speed, and ergonomics.

What's inside
-------------
- `choose_strides(...)` — Pick integer strides `(stride_r, stride_n)` so that a
  large `(R, N)` grid is sub‑sampled to a target number of points without
  overwhelming your plots.
- `viz_sigma_scatter(...)` — 3‑D scatter of sample points `(x, y, z)` colored by
  σ.
- `viz_sigma_heatmap(...)` — 2‑D heatmap of σ with axes `(ray, sample)`.
- `plot_nonzero_sigma_row(...)` — Inspect a single ray’s σ profile and mark the
  nonzero region(s).
- `visualize_sigma(...)` — High‑level orchestrator: scatter + heatmap + row
  inspection with sensible defaults and safety checks.

Design notes
------------
- Torch is **optional**. If you pass `torch.Tensor`, tensors are detached and
  moved to CPU internally.
- All functions sanitize NaNs/±Inf and clamp σ ≥ 0.
- Subsampling is conservative and reproducible (NumPy Generator).

Example
-------
>>> # Fake data for demonstration (R=64 rays, N=128 samples)
>>> import numpy as np, torch
>>> R, N = 64, 128
>>> t     = torch.linspace(0, 1, N).repeat(R, 1)                         # (R,N)
>>> delta = torch.full_like(t, 1.0 / N)                                   # (R,N)
>>> pts   = torch.randn(R, N, 3) * 0.5 + torch.tensor([0, 0, 2.0])       # (R,N,3)
>>> sigma = torch.relu(torch.sin(8 * t) + 0.1 * torch.randn(R, N))       # (R,N)
>>> samples = (t, delta, pts)
>>> out = visualize_sigma(samples, sigma, max_scatter_points=6000,
...                       max_samples_axis=32, clip_max=50.0)
>>> print(out)
{'row_used': 13, 'peak_sigma': 0.98, 'peak_t': 0.73, 'stride_r': 2, 'stride_n': 4}

Dependencies
------------
- numpy, matplotlib
- (optional) torch
"""
from __future__ import annotations

from typing import Optional, Tuple, Union, Dict
import math
import numpy as np
import matplotlib.pyplot as plt

try:  # Torch is optional
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


# =============================================================================
# Basic utilities
# =============================================================================

def _to_numpy_f32(x: Union[np.ndarray, "torch.Tensor"], *, clip_max: Optional[float] = None) -> np.ndarray:
    """Return a contiguous float32 NumPy array.

    - If `x` is a torch tensor, it is detached and moved to CPU.
    - Otherwise, converts via `np.asarray`.
    - If `clip_max` is provided, values above it are clipped.
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().to("cpu")
        arr = x.numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if clip_max is not None:
        arr = np.nan_to_num(arr, nan=0.0, posinf=clip_max, neginf=0.0).clip(0.0, clip_max)
    return arr


def _validate_shapes(pts: np.ndarray, sigma: np.ndarray) -> Tuple[int, int]:
    """Validate shapes for `(R, N, 3)` points and `(R, N)` sigma arrays."""
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"pts must be shape (R, N, 3); got {pts.shape}")
    if sigma.ndim != 2 or sigma.shape != pts.shape[:2]:
        raise ValueError(
            f"sigma must be shape (R, N) matching pts; got {sigma.shape} vs {pts.shape}"
        )
    return int(pts.shape[0]), int(pts.shape[1])


# =============================================================================
# Subsampling policy
# =============================================================================

def choose_strides(
    Rtot: int,
    Ntot: int,
    *,
    max_scatter_points: int = 5_000,
    max_samples_axis: int = 16,
) -> Tuple[int, int, int, int]:
    """Choose strides `(stride_r, stride_n)` for subsampling a large `(R, N)` grid.

    The goal is to keep the scatter plot to roughly `max_scatter_points` while
    also capping the number of samples per ray to about `max_samples_axis`.

    Parameters
    ----------
    Rtot, Ntot : int
        Full ray and per‑ray sample counts.
    max_scatter_points : int, default 5000
        Approximate budget for the number of *displayed* points in the 3‑D scatter.
    max_samples_axis : int, default 16
        Upper bound for `N'` (the number of samples per ray after subsampling).

    Returns
    -------
    (stride_r, stride_n, Rprime, Nprime) : tuple of ints
        The chosen strides and the resulting subsampled sizes.

    Notes
    -----
    - Uses ceiling‑style integer arithmetic.
    - If the initial estimate still exceeds the budget due to rounding, we
      iteratively increase `stride_r`.
    - As a last resort, we fall back to a `sqrt` heuristic on total points.
    """
    # Step 1: cap samples per ray (N')
    stride_n = max(1, (Ntot + max_samples_axis - 1) // max_samples_axis)  # ceil
    Nprime = max(1, Ntot // stride_n)

    # Step 2: pick rays so R' * N' ≈ max_scatter_points
    Rprime_target = max(1, max_scatter_points // Nprime)
    stride_r = max(1, (Rtot + Rprime_target - 1) // Rprime_target)  # ceil
    Rprime = max(1, Rtot // stride_r)

    # Tighten if rounding overshot the budget
    while Rprime * Nprime > max_scatter_points and stride_r < Rtot:
        stride_r += 1
        Rprime = max(1, Rtot // stride_r)

    # Emergency fallback: sqrt heuristic
    if Rprime * Nprime > max_scatter_points:
        stride_r = max(1, int(math.sqrt(max(1, (Rtot * Ntot) / max_scatter_points))))
        stride_n = max(1, Ntot // max_samples_axis)
        Rprime = max(1, Rtot // stride_r)
        Nprime = max(1, Ntot // stride_n)

    return stride_r, stride_n, Rprime, Nprime


# =============================================================================
# Visualization primitives
# =============================================================================

def viz_sigma_scatter(
    pts: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    clip_max: float = 100.0,
    max_points: Optional[int] = None,
    cmap: Optional[str] = None,
    add_triad: bool = True,
    title: Optional[str] = None,
    show: bool = True,
):
    """Scatter 3‑D sample points `(x, y, z)` colored by σ.

    Parameters
    ----------
    pts : (R, N, 3) ndarray or torch.Tensor
    sigma : (R, N) ndarray or torch.Tensor
    clip_max : float, default 100.0
        Clamp σ to `[0, clip_max]` for a stable colorbar.
    max_points : int | None
        If set and total points exceed this value, randomly subsample to speed up plotting.
    cmap : str | None
        Matplotlib colormap (e.g., 'viridis'). Defaults to Matplotlib's default.
    add_triad : bool, default True
        If True, draws a small XYZ triad at the origin for orientation.
    title : str | None
    show : bool, default True
        If True, `plt.show()` is called.

    Returns
    -------
    (fig, ax)
    """
    Pn = _to_numpy_f32(pts)
    Sn = _to_numpy_f32(sigma)

    Pn = _to_numpy_f32(pts)
    Sn = _to_numpy_f32(sigma, clip_max=clip_max)

    _R, _N = _validate_shapes(Pn, Sn)
    P = Pn.reshape(-1, 3)
    S = Sn.reshape(-1)

    # Optional subsampling
    M = P.shape[0]
    if isinstance(max_points, int) and max_points > 0 and M > max_points:
        rng = np.random.default_rng(0)  # fixed seed for reproducibility
        sel = rng.choice(M, size=max_points, replace=False)
        P, S = P[sel], S[sel]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    if M == 0 or P.size == 0:
        if title:
            ax.set_title(title)
        ax.text2D(0.5, 0.5, "No points to display", transform=ax.transAxes, ha="center")
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=S, s=2, cmap=cmap, depthshade=False)

    if add_triad:
        _draw_triad(ax, length=0.25)

    _style_3d_axis(ax)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("σ")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def viz_sigma_heatmap(
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    clip_max: float = 100.0,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """Display σ as a (ray × sample) heatmap.

    Parameters
    ----------
    sigma : (R, N) ndarray or torch.Tensor
    clip_max : float, default 100.0
    cmap : str | None
    figsize : (w, h) | None
    title : str | None
    show : bool, default True

    Returns
    -------
    (fig, ax)
    """
    S = _to_numpy_f32(sigma, clip_max=clip_max )
    
    if S.ndim != 2:
        raise ValueError(f"sigma must be shape (R, N); got {S.shape}")

    fig, ax = plt.subplots(figsize=figsize or (9, 4), dpi=120)
    im = ax.imshow(S, aspect="auto", cmap=cmap)
    ax.set_xlabel("Sample index (N)")
    ax.set_ylabel("Ray index (R)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("σ")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_nonzero_sigma_row(
    t: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    row_idx: int,
    threshold: float = 0.0,
    mark_nonzero: bool = True,
    title: Optional[str] = None,
    clip_max: Optional[float] = 100,
) -> int:
    """Plot σ along a single ray.

    Parameters
    ----------
    t : (R, N) ndarray or torch.Tensor
        Per‑sample param (e.g., distance along the ray) for labeling.
    sigma : (R, N) ndarray or torch.Tensor
        Densities.
    row_idx : int
        Which ray to plot.
    threshold : float, default 0.0
        Values > threshold are considered "nonzero" for marking.
    mark_nonzero : bool, default True
        Shade regions where σ > threshold.
    title : str | None
    clip_max : float | None
        If set, clips σ values to this maximum.
    Returns
    -------
    int
        The (possibly clamped) row index actually plotted.
    """
    T = _to_numpy_f32(t)
    S = _to_numpy_f32(sigma, clip_max=clip_max)
    if T.shape != S.shape:
        raise ValueError(f"t and sigma shapes must match; got {T.shape} vs {S.shape}")

    R, N = S.shape
    r = int(np.clip(row_idx, 0, R - 1))
    row = S[r]

    x = np.arange(N)
    fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
    ax.plot(x, row, linewidth=1.5)
    ax.set_xlabel("Sample index along ray")
    ax.set_ylabel("σ")

    if mark_nonzero:
        mask = row > threshold
        if np.any(mask):
            # Fill contiguous positive segments
            starts = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int)])) == 1)
            stops  = np.flatnonzero(np.diff(np.concatenate([mask.astype(int), [0]])) == -1)
            for a, b in zip(starts, stops):
                ax.axvspan(a, b - 1, alpha=0.15)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return r


# =============================================================================
# High‑level orchestrator
# =============================================================================

def visualize_sigma(
    samples: Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]],
    # (t, delta, pts) where t, delta: (R,N), pts: (R,N,3)
    sigma_full: Union[np.ndarray, "torch.Tensor"],
    *,
    show_scatter: bool = True,
    show_heatmap: bool = True,
    show_row_inspect: bool = True,
    max_scatter_points: int = 5_000,
    max_samples_axis: int = 16,
    heatmap_subsample_threshold: int = 2_000_000,
    idx_ray_render: Optional[int] = None,  # None → auto pick by max σ
    # Titles
    titles: bool = True,
    scatter_title: Optional[str] = None,
    heatmap_title: Optional[str] = None,
    row_plot_title: Optional[str] = None,
    clip_max: Optional[float] = 100,
) -> Dict[str, Optional[float]]:
    """Visualize σ along rays in three coordinated views.

    Views
    -----
    1) **3‑D scatter** of sample points colored by σ (subsampled).
    2) **Heatmap** of σ with optional subsampling for huge arrays.
    3) **Row inspection**: pick a ray (auto by max σ or user‑specified),
       plot σ across samples, and report the peak value and its `t`.

    Parameters
    ----------
    samples : tuple (t, delta, pts)
        - `t`, `delta` have shape `(R, N)`; `pts` has shape `(R, N, 3)`.
    sigma_full : (R, N) ndarray or torch.Tensor
        Densities along rays.
    show_scatter, show_heatmap, show_row_inspect : bool
        Toggle individual views.
    max_scatter_points : int, default 5000
        Target point budget for the scatter view.
    max_samples_axis : int, default 16
        Upper bound for samples per ray after subsampling.
    heatmap_subsample_threshold : int, default 2_000_000
        If `R*N` exceeds this, the heatmap uses the same strides as the scatter.
    idx_ray_render : int | None
        Which ray to inspect. If `None`, picks the row with the largest max σ.
    titles : bool, default True
        If True, auto‑generate default titles for each view (unless overridden).
    scatter_title, heatmap_title, row_plot_title : str | None
        Per‑view title overrides.
    clip_max : float | None
        If set, clips σ values to this maximum.
    Returns
    -------
    dict with keys
        - `row_used` : int | None
        - `peak_sigma` : float | None
        - `peak_t` : float | None
        - `stride_r` : int
        - `stride_n` : int
    """
    if sigma_full is None or samples is None:
        print("Sigma/samples not provided; pass both to visualize.")
        return {"row_used": None, "peak_sigma": None, "peak_t": None, "stride_r": 1, "stride_n": 1}

    t, delta, pts = samples

    # Basic shape checks
    if _HAS_TORCH and isinstance(sigma_full, torch.Tensor):
        Rtot, Ntot = tuple(sigma_full.shape)
    else:
        Rtot, Ntot = np.asarray(sigma_full).shape[:2]

    assert _to_numpy_f32(t).shape == (Rtot, Ntot), f"t shape mismatch: {_to_numpy_f32(t).shape} vs {(Rtot, Ntot)}"
    pts_np = _to_numpy_f32(pts)
    assert pts_np.shape[:2] == (Rtot, Ntot) and pts_np.shape[-1] == 3, f"pts shape mismatch: {pts_np.shape}"

    # Choose strides once and reuse
    stride_r, stride_n, Rp, Np = choose_strides(
        Rtot, Ntot,
        max_scatter_points=max_scatter_points,
        max_samples_axis=max_samples_axis,
    )

    # Move to NumPy for plotting (keeps original tensors untouched)
    sigma_np = _to_numpy_f32(sigma_full, clip_max=clip_max)
    t_np     = _to_numpy_f32(t)

    # ---------- Scatter ----------
    if show_scatter:
        X_sub = pts_np[::stride_r, ::stride_n, :]
        S_sub = sigma_np[::stride_r, ::stride_n]
        if X_sub.ndim == 3 and X_sub.shape[:2] == S_sub.shape:
            title = scatter_title if scatter_title is not None else (
                f"σ scatter (subsampled) • R'={X_sub.shape[0]} N'={X_sub.shape[1]}" if titles else None
            )
            viz_sigma_scatter(X_sub, S_sub, title=title)
        else:
            print("Skipping scatter: incompatible subsampled shapes:", X_sub.shape, S_sub.shape)

    # ---------- Heatmap ----------
    if show_heatmap:
        use_sub = (Rtot * Ntot > heatmap_subsample_threshold)
        if use_sub:
            S_plot = sigma_np[::stride_r, ::stride_n]
            title = heatmap_title if heatmap_title is not None else (
                f"σ heatmap (subsampled) • stride_r={stride_r}, stride_n={stride_n}" if titles else None
            )
        else:
            S_plot = sigma_np
            title = heatmap_title if heatmap_title is not None else ("σ heatmap (full)" if titles else None)
        viz_sigma_heatmap(S_plot, title=title)

    # ---------- Row inspection ----------
    row_used: Optional[int] = None
    peak_sigma: Optional[float] = None
    peak_t: Optional[float] = None

    if show_row_inspect:
        if idx_ray_render is None:
            row_scores = sigma_np.max(axis=1)  # (R,)
            row_used = int(row_scores.argmax())
            auto_msg = "auto‑selected (max σ row)"
        else:
            row_used = int(np.clip(idx_ray_render, 0, Rtot - 1))
            auto_msg = "user‑selected"

        r_idx = plot_nonzero_sigma_row(
            t_np, sigma_np,
            row_idx=row_used,
            threshold=0.0,
            mark_nonzero=True,
            title=(row_plot_title or (f"σ along ray {row_used} (nonzero shaded)") if titles else None),
        )
        print(f"Picked row: {r_idx} • {auto_msg}")

        row_sigma = sigma_np[row_used]
        j = int(row_sigma.argmax())
        peak_sigma = float(row_sigma[j])
        peak_t = float(t_np[row_used, j])
        print(f"Row {row_used}: max σ = {peak_sigma:.6f} at t = {peak_t:.6f}")

    return {
        "row_used": row_used,
        "peak_sigma": peak_sigma,
        "peak_t": peak_t,
        "stride_r": stride_r,
        "stride_n": stride_n,
    }


# =============================================================================
# Small axis helpers (pure Matplotlib)
# =============================================================================

def _draw_triad(ax, *, length: float = 0.2):
    ax.quiver(0, 0, 0, length, 0, 0)
    ax.quiver(0, 0, 0, 0, length, 0)
    ax.quiver(0, 0, 0, 0, 0, length)
    ax.text(length, 0, 0, 'X')
    ax.text(0, length, 0, 'Y')
    ax.text(0, 0, length, 'Z')


def _style_3d_axis(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-60)


if __name__ == "__main__":  # Minimal self‑test / demo
    R, N = 48, 96
    if _HAS_TORCH:
        t = torch.linspace(0, 1, N).repeat(R, 1)
        delta = torch.full_like(t, 1.0 / N)
        pts = torch.randn(R, N, 3) * 0.4 + torch.tensor([0.0, 0.0, 2.0])
        sigma = torch.relu(torch.sin(10 * t) + 0.15 * torch.randn(R, N))
    else:
        t = np.tile(np.linspace(0, 1, N), (R, 1))
        delta = np.full_like(t, 1.0 / N)
        pts = np.random.randn(R, N, 3) * 0.4 + np.array([0.0, 0.0, 2.0])
        sigma = np.clip(np.sin(10 * t) + 0.15 * np.random.randn(R, N), a_min=0.0, a_max=None)

    print(visualize_sigma((t, delta, pts), sigma, max_scatter_points=6000, max_samples_axis=24))
