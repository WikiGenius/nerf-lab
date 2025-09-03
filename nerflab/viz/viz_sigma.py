"""
Sigma Visualization — Standard, Optimized, and Well‑Documented
===============================================================

A single, cohesive module for inspecting NeRF per‑sample density (sigma, σ)
across rays. It merges and standardizes the functionality from your two
previous files into a consistent, dependency‑light API.

Highlights
----------
- **Self‑contained**: Only NumPy + Matplotlib (Torch is optional).
- **Deterministic subsampling**: Integer stride policy with safety fallback.
- **Robust I/O**: Works with NumPy arrays or torch.Tensors (auto‑CPU detach).
- **Clean plots**: 3‑D scatter, (R×N) heatmap, and single‑row inspector.
- **Utilities**: row selection by non‑zero, random/max strategies, safe
  broadcasting of `t`.

Public API
----------
- choose_strides(Rtot, Ntot, *, max_scatter_points=5000, max_samples_axis=16)
- viz_sigma_scatter(pts, sigma, *, clip_max=100.0, max_points=None, cmap=None,
                   add_triad=True, title=None, show=True)
- viz_sigma_heatmap(sigma, *, clip_max=100.0, cmap=None, figsize=None,
                    title=None, show=True)
- plot_nonzero_sigma_row(t, sigma, *, row_idx=None, clip_max=1000.0,
                         rng=None, strategy="random", ax=None, show=True,
                         mark_nonzero=False, empty="fallback") -> int
- visualize_sigma((t, delta, pts), sigma_full, *[, options...]) -> dict
- to_numpy(x) / to_numpy_f32(x, clip_max=None, clamp_nonneg=False)
- rows_with_nonzero(sigma) -> np.ndarray
- pick_random_nonzero_entry(sigma, rng=None) -> (r, c)

Example
-------
>>> import numpy as np
>>> R, N = 32, 128
>>> t = np.tile(np.linspace(0, 1, N), (R, 1))
>>> pts = np.random.randn(R, N, 3) * 0.3 + np.array([0, 0, 2.0])
>>> sigma = np.clip(np.sin(8*t) + 0.15*np.random.randn(R, N), 0, None)
>>> out = visualize_sigma((t, None, pts), sigma, max_scatter_points=6000,
...                       max_samples_axis=24, clip_max=3.0)
>>> out["stride_r"], out["stride_n"]
(2, 6)
"""
from __future__ import annotations

from typing import Optional, Tuple, Union, Dict
import math
import numpy as np
import matplotlib.pyplot as plt

try:  # Optional torch support
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]

# =============================================================================
# Conversion & validation helpers
# =============================================================================

def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert to `np.ndarray` (no dtype changes).

    - torch.Tensor → detach → cpu → numpy
    - else → `np.asarray(x)`
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        x = x.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(x)


def to_numpy_f32(
    x: ArrayLike,
    *,
    clip_max: Optional[float] = None,
    clamp_nonneg: bool = False,
) -> np.ndarray:
    """Return a contiguous `float32` NumPy array with optional clipping.

    Parameters
    ----------
    x : array_like or torch.Tensor
    clip_max : float | None
        If given, clip values to [0, clip_max] and sanitize NaN/±Inf.
    clamp_nonneg : bool
        If True, clamp negatives to 0.
    """
    arr = to_numpy(x).astype(np.float32, copy=False)
    if clip_max is not None:
        arr = np.nan_to_num(arr, nan=0.0, posinf=clip_max, neginf=0.0)
        arr = np.clip(arr, 0.0, float(clip_max))
    if clamp_nonneg:
        arr = np.maximum(arr, 0.0)
    return np.ascontiguousarray(arr)


def _validate_pts_sigma(pts: np.ndarray, sigma: np.ndarray) -> Tuple[int, int]:
    """Validate `(R,N,3)` points and `(R,N)` sigma; return `(R,N)`.
    Raises `ValueError` on mismatch.
    """
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"`pts` must be (R, N, 3); got {pts.shape}.")
    if sigma.ndim != 2 or sigma.shape != pts.shape[:2]:
        raise ValueError(
            f"`sigma` must be (R, N) matching `pts`; got {sigma.shape} vs {pts.shape}."
        )
    return int(pts.shape[0]), int(pts.shape[1])


def _broadcast_t(t: ArrayLike, shape: Tuple[int, int]) -> np.ndarray:
    """Return `t` broadcast to (R,N). Accepts (N,) or (R,N)."""
    R, N = shape
    T = to_numpy(t)
    if T.ndim == 1:
        if T.shape[0] != N:
            raise ValueError(f"`t` length {T.shape[0]} != N {N}.")
        T = np.broadcast_to(T, (R, N))
    elif T.ndim == 2:
        if T.shape != (R, N):
            raise ValueError(f"`t` shape {T.shape} must equal (R, N)=({R}, {N}).")
    else:
        raise ValueError("`t` must be 1‑D (N,) or 2‑D (R,N).")
    return T


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
    """Choose `(stride_r, stride_n)` to downsample an (R,N) grid.

    Strategy
    --------
    1) Limit per‑ray samples: choose `stride_n` so post‑subsample `N' ≤ max_samples_axis`.
    2) With `N'` fixed, pick `stride_r` s.t. `R' * N' ≈ max_scatter_points`.
    3) If rounding overshoots budget, increment `stride_r`.
    4) Final safeguard: sqrt heuristic on total points.

    Returns `(stride_r, stride_n, Rprime, Nprime)`.
    """
    # Step 1: cap samples per ray
    stride_n = max(1, (Ntot + max_samples_axis - 1) // max_samples_axis)  # ceil
    Nprime = max(1, Ntot // stride_n)

    # Step 2: approximate rays count given the point budget
    Rprime_target = max(1, max_scatter_points // Nprime)
    stride_r = max(1, (Rtot + Rprime_target - 1) // Rprime_target)  # ceil
    Rprime = max(1, Rtot // stride_r)

    # Step 3: tighten if needed
    while Rprime * Nprime > max_scatter_points and stride_r < Rtot:
        stride_r += 1
        Rprime = max(1, Rtot // stride_r)

    # Step 4: safeguard for extreme cases
    if Rprime * Nprime > max_scatter_points:
        stride_r = max(1, int(math.sqrt(max(1, (Rtot * Ntot) / max_scatter_points))))
        stride_n = max(1, Ntot // max_samples_axis)
        Rprime = max(1, Rtot // stride_r)
        Nprime = max(1, Ntot // stride_n)

    return stride_r, stride_n, Rprime, Nprime


# =============================================================================
# Small axis helpers (pure Matplotlib)
# =============================================================================

def _draw_triad(ax: plt.Axes, *, length: float = 0.2) -> None:
    ax.quiver(0, 0, 0, length, 0, 0)
    ax.quiver(0, 0, 0, 0, length, 0)
    ax.quiver(0, 0, 0, 0, 0, length)
    ax.text(length, 0, 0, 'X')
    ax.text(0, length, 0, 'Y')
    ax.text(0, 0, length, 'Z')


def _style_3d_axis(ax: plt.Axes) -> None:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-60)


# =============================================================================
# Visualization primitives
# =============================================================================

def viz_sigma_scatter(
    pts: ArrayLike,
    sigma: ArrayLike,
    *,
    clip_max: float = 100.0,
    max_points: Optional[int] = None,
    cmap: Optional[str] = None,
    add_triad: bool = True,
    title: Optional[str] = None,
    show: bool = True,
):
    """3‑D scatter of sample points `(x, y, z)` colored by σ.

    Parameters
    ----------
    pts : (R, N, 3)
    sigma : (R, N)
    clip_max : float
        Clip σ to [0, clip_max] for a stable colorbar.
    max_points : int | None
        If set and total points exceed this value, randomly subsample to speed up plotting.
    cmap : str | None
    add_triad : bool
    title : str | None
    show : bool

    Returns
    -------
    (fig, ax)
    """
    Pn = to_numpy_f32(pts)
    Sn = to_numpy_f32(sigma, clip_max=clip_max, clamp_nonneg=True)
    _validate_pts_sigma(Pn, Sn)

    P = Pn.reshape(-1, 3)
    S = Sn.reshape(-1)

    M = P.shape[0]
    if isinstance(max_points, int) and max_points > 0 and M > max_points:
        rng = np.random.default_rng(0)  # fixed seed for reproducibility
        sel = rng.choice(M, size=max_points, replace=False)
        P, S = P[sel], S[sel]
        M = max_points

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    if M == 0:
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
    sigma: ArrayLike,
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
    sigma : (R, N)
    clip_max : float
    cmap : str | None
    figsize : (w, h) | None
    title : str | None
    show : bool

    Returns
    -------
    (fig, ax)
    """
    S = to_numpy_f32(sigma, clip_max=clip_max, clamp_nonneg=True)
    if S.ndim != 2:
        raise ValueError(f"`sigma` must be (R, N); got {S.shape}.")

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


# =============================================================================
# Row utilities
# =============================================================================

def rows_with_nonzero(sigma: ArrayLike) -> np.ndarray:
    """Return sorted row indices that contain at least one non‑zero entry."""
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2‑D with shape (R, N).")
    return np.flatnonzero((S != 0).any(axis=1))


def pick_random_nonzero_entry(
    sigma: ArrayLike, rng: Optional[np.random.Generator] = None
) -> Tuple[int, int]:
    """Pick a random (row, col) where sigma != 0; raises if none exist."""
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2‑D with shape (R, N).")
    nz = np.flatnonzero(S.ravel() != 0)
    if nz.size == 0:
        raise ValueError("No non‑zero entries in `sigma`.")
    rng = rng or np.random.default_rng()
    k = int(rng.choice(nz))
    return tuple(map(int, np.unravel_index(k, S.shape)))


def _select_row(eligible: np.ndarray, S: np.ndarray, *, strategy: str, rng) -> int:
    """Select a row index from `eligible` according to a strategy."""
    if eligible.size == 0:
        raise ValueError("No rows contain non‑zero sigma entries.")
    if strategy == "random":
        rng = rng or np.random.default_rng()
        return int(rng.choice(eligible))
    if strategy == "max":
        peaks = S[eligible].max(axis=1)
        return int(eligible[int(np.argmax(peaks))])
    raise ValueError("`strategy` must be 'random' or 'max'.")


def plot_nonzero_sigma_row(
    t: ArrayLike,
    sigma: ArrayLike,
    *,
    row_idx: Optional[int] = None,
    clip_max: Optional[float] = 1000.0,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "random",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    mark_nonzero: bool = False,
    empty: str = "plot_zero",  # {"error","fallback","plot_zero"}
) -> int:
    """Plot σ for a single ray‑row that has at least one non‑zero entry.

    Returns the row index actually plotted. See module docstring for params.
    """
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2‑D with shape (R, N).")
    R, N = S.shape
    T = _broadcast_t(t, (R, N))

    eligible = rows_with_nonzero(S)

    if row_idx is None:
        row_idx = _select_row(eligible, S, strategy=strategy, rng=rng)
    else:
        row_idx = int(row_idx)
        if not (0 <= row_idx < R):
            raise ValueError(f"`row_idx` out of range [0, {R-1}].")
        if not (S[row_idx] != 0).any():
            if empty == "fallback":
                row_idx = _select_row(eligible, S, strategy=strategy, rng=rng)
            elif empty == "plot_zero":
                pass
            elif empty == "error":
                raise ValueError(f"Requested row_idx={row_idx} has no non‑zero entries.")
            else:
                raise ValueError("`empty` must be one of {'error','fallback','plot_zero' }.")

    x = T[row_idx]
    y = S[row_idx]
    if clip_max is not None:
        y = np.clip(y, 0.0, float(clip_max))

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
        created = True

    ax.plot(x, y, lw=1.6)
    if mark_nonzero:
        nz = (S[row_idx] != 0)
        ax.scatter(x[nz], y[nz], s=12, zorder=3)

    ax.set_xlabel("t")
    ax.set_ylabel("σ")
    ax.set_title(f"σ along ray row {row_idx}")
    ax.grid(True, alpha=0.25)

    if created and show:
        plt.tight_layout()
        plt.show()

    return row_idx


# =============================================================================
# High‑level orchestrator
# =============================================================================

def visualize_sigma(
    samples: Tuple[ArrayLike, Optional[ArrayLike], ArrayLike],  # (t, delta, pts)
    sigma_full: ArrayLike,
    *,
    show_scatter: bool = True,
    show_heatmap: bool = True,
    show_row_inspect: bool = True,
    max_scatter_points: int = 5_000,
    max_samples_axis: int = 16,
    heatmap_subsample_threshold: int = 2_000_000,
    idx_ray_render: Optional[int] = None,
    titles: bool = True,
    scatter_title: Optional[str] = None,
    heatmap_title: Optional[str] = None,
    row_plot_title: Optional[str] = None,
    clip_max: Optional[float] = 100,
) -> Dict[str, Optional[float]]:
    """Visualize σ along rays in three coordinated views.

    Returns a dict with: `row_used`, `peak_sigma`, `peak_t`, `stride_r`, `stride_n`.
    """
    if sigma_full is None or samples is None:
        print("Sigma/samples not provided; pass both to visualize().")
        return {"row_used": None, "peak_sigma": None, "peak_t": None, "stride_r": 1, "stride_n": 1}

    t, _delta, pts = samples

    if _HAS_TORCH and isinstance(sigma_full, torch.Tensor):
        Rtot, Ntot = tuple(sigma_full.shape)
    else:
        Rtot, Ntot = np.asarray(sigma_full).shape[:2]

    pts_np = to_numpy_f32(pts)
    if pts_np.shape[:2] != (Rtot, Ntot) or pts_np.shape[-1] != 3:
        raise ValueError(f"`pts` shape mismatch: expected (R,N,3)=({Rtot},{Ntot},3), got {pts_np.shape}.")

    # Strides (computed once)
    stride_r, stride_n, Rp, Np = choose_strides(
        Rtot, Ntot,
        max_scatter_points=max_scatter_points,
        max_samples_axis=max_samples_axis,
    )

    sigma_np = to_numpy_f32(sigma_full, clip_max=clip_max, clamp_nonneg=True)
    t_np = to_numpy_f32(_broadcast_t(t, (Rtot, Ntot)))

    # ---------- Scatter ----------
    if show_scatter:
        X_sub = pts_np[::stride_r, ::stride_n, :]
        S_sub = sigma_np[::stride_r, ::stride_n]
        title = scatter_title if scatter_title is not None else (
            f"σ scatter (subsampled) • R'={X_sub.shape[0]} N'={X_sub.shape[1]}" if titles else None
        )
        viz_sigma_scatter(X_sub, S_sub, title=title)

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
            row_used = int(np.clip(int(idx_ray_render), 0, Rtot - 1))
            auto_msg = "user‑selected"

        _ = plot_nonzero_sigma_row(
            t_np, sigma_np,
            row_idx=row_used,
            clip_max=clip_max,
            mark_nonzero=True,
            show=True,
        )
        print(f"Picked row: {row_used} • {auto_msg}")

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


__all__ = [
    # Converters
    "to_numpy", "to_numpy_f32",
    # Subsampling
    "choose_strides",
    # Plots
    "viz_sigma_scatter", "viz_sigma_heatmap", "plot_nonzero_sigma_row",
    # Orchestrator
    "visualize_sigma",
    # Row utilities
    "rows_with_nonzero", "pick_random_nonzero_entry",
]


if __name__ == "__main__":  # Minimal demo
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

    stats = visualize_sigma((t, delta, pts), sigma, max_scatter_points=6000, max_samples_axis=24)
    print(stats)
