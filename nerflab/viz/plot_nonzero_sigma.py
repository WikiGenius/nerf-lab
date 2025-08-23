
"""
sigma_plot.py — robust helpers for inspecting per-ray sigma profiles.

Key ideas
---------
- `sigma` is shaped (R, N): R ray-rows, N samples per ray along parameter t.
- `t` can be either shape (N,) and will be broadcast to (R, N), or directly (R, N).
- By default we *plot rows that have at least one non-zero entry*; if a requested
  row is empty, we *fall back* to an eligible row (configurable via `empty=`).

Provided API
------------
- to_numpy(x):        safe torch→numpy converter (public; useful elsewhere)
- rows_with_nonzero:  indices of rows that contain at least one non-zero entry
- pick_random_nonzero_entry(sigma, rng=None) -> (r, c)
- plot_nonzero_sigma_row(t, sigma, *, row_idx=None, clip_max=1000.0,
                         rng=None, strategy="random", ax=None, show=True,
                         mark_nonzero=False, empty="fallback") -> int

Change log (vs your original)
-----------------------------
- Fixed figure-show logic (no accidental plt.show() when `ax` is passed).
- Centralized shape validation/broadcasting (no duplication).
- Added `rows_with_nonzero` utility for clarity.
- Added `mark_nonzero` for highlighting non-zero samples.
- Added `empty=` policy with default "fallback" to avoid common errors.
- Clear, concise docstrings + examples; consistent error messages.
- Exported a complete __all__.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:  # Optional torch support
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Public small utilities
# --------------------------------------------------------------------------- #
def to_numpy(x) -> np.ndarray:
    """
    Convert input to NumPy array. If `x` is a torch.Tensor, detach→cpu→numpy.
    Otherwise, returns `np.asarray(x)`.

    Parameters
    ----------
    x : array_like or torch.Tensor

    Returns
    -------
    np.ndarray
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        x = x.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(x)


def rows_with_nonzero(sigma: np.ndarray) -> np.ndarray:
    """
    Return row indices that contain at least one non-zero entry.

    Parameters
    ----------
    sigma : (R, N) array_like

    Returns
    -------
    eligible : (K,) np.ndarray of dtype=int
        Sorted row indices with any non-zero.
    """
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2-D with shape (R, N).")
    return np.flatnonzero((S != 0).any(axis=1))


def pick_random_nonzero_entry(
    sigma, rng: Optional[np.random.Generator] = None
) -> Tuple[int, int]:
    """
    Pick a random (row, col) where sigma != 0.

    Parameters
    ----------
    sigma : (R, N) array_like
    rng   : np.random.Generator, optional

    Returns
    -------
    (r, c) : tuple[int, int]

    Raises
    ------
    ValueError if there are no non-zero entries.
    """
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2-D with shape (R, N).")
    nz = np.flatnonzero(S.ravel() != 0)
    if nz.size == 0:
        raise ValueError("No non-zero entries in `sigma`.")
    rng = rng or np.random.default_rng()
    k = int(rng.choice(nz))
    return tuple(map(int, np.unravel_index(k, S.shape)))  # (r, c)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _validate_and_broadcast_t(t, sigma_shape) -> np.ndarray:
    """Return t as shape (R, N), validating compatibility with sigma shape."""
    S_R, S_N = sigma_shape
    T = to_numpy(t)
    if T.ndim == 1:
        if T.shape[0] != S_N:
            raise ValueError(f"`t` length {T.shape[0]} != sigma N {S_N}.")
        T = np.broadcast_to(T, (S_R, S_N))
    elif T.ndim == 2:
        if T.shape != (S_R, S_N):
            raise ValueError(f"`t` shape {T.shape} must match sigma shape {(S_R, S_N)}.")
    else:
        raise ValueError("`t` must be 1-D (N,) or 2-D (R, N).")
    return T


def _select_row(eligible: np.ndarray, S: np.ndarray, *, strategy: str, rng) -> int:
    """Select a row index from `eligible` according to a strategy."""
    if eligible.size == 0:
        raise ValueError("No rows contain non-zero sigma entries.")
    if strategy == "random":
        rng = rng or np.random.default_rng()
        return int(rng.choice(eligible))
    if strategy == "max":
        peaks = S[eligible].max(axis=1)
        return int(eligible[int(np.argmax(peaks))])
    raise ValueError("`strategy` must be 'random' or 'max'.")


# --------------------------------------------------------------------------- #
# Core plotting helper
# --------------------------------------------------------------------------- #
def plot_nonzero_sigma_row(
    t,
    sigma,
    *,
    row_idx: Optional[int] = None,
    clip_max: Optional[float] = 1000.0,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "random",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    mark_nonzero: bool = False,
    empty: str = "fallback",  # {"error","fallback","plot_zero"}
) -> int:
    """
    Plot sigma for a single ray-row that has at least one non-zero entry.

    Parameters
    ----------
    t : array_like, shape (N,) or (R, N)
        Sample parameter values. If (N,), will be broadcast to (R, N).
    sigma : array_like, shape (R, N)
        Sigma values per ray-row and sample index.
    row_idx : int, optional
        If given, plot that row. If it is *empty* (all zeros), the behavior is
        controlled by `empty`.
    clip_max : float or None, default 1000.0
        Clip sigma to [0, clip_max] before plotting. Use None to disable clipping.
    rng : np.random.Generator, optional
        Random generator for 'random' strategy.
    strategy : {"random", "max"}, default "random"
        Row selection when `row_idx` is None, or when `empty="fallback"`.
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axes. Otherwise create a new figure/axes.
    show : bool, default True
        If we create the figure (ax is None), call plt.tight_layout() and plt.show()
        when show=True. Has no effect if ax is provided.
    mark_nonzero : bool, default False
        If True, add scatter markers at positions where sigma != 0.
    empty : {"error","fallback","plot_zero"}, default "fallback"
        What to do if `row_idx` is given but that row has no non-zero entries.
        - "error": raise ValueError
        - "fallback": pick an eligible row using `strategy`
        - "plot_zero": proceed and plot the all-zero row

    Returns
    -------
    int
        The row index that was plotted.

    Raises
    ------
    ValueError
        - if `sigma` is not (R, N)
        - if `t` shape is incompatible
        - if there are no eligible rows when one is required

    Examples
    --------
    >>> R, N = 4, 64
    >>> t = np.linspace(0, 1, N)
    >>> sigma = np.zeros((R, N)); sigma[2, 10:20] = 0.5
    >>> r = plot_nonzero_sigma_row(t, sigma)  # random eligible row (likely 2)
    >>> r = plot_nonzero_sigma_row(t, sigma, strategy="max")  # row with highest peak
    >>> r = plot_nonzero_sigma_row(np.broadcast_to(t, (R, N)), sigma, row_idx=2)

    Torch inputs also work:
    >>> import torch
    >>> t_t = torch.linspace(0, 1, N)
    >>> sigma_t = torch.zeros(R, N); sigma_t[1, 5:15] = 2.0
    >>> r = plot_nonzero_sigma_row(t_t, sigma_t, mark_nonzero=True)
    """
    # --- Convert to numpy and validate shapes ---
    S = to_numpy(sigma)
    if S.ndim != 2:
        raise ValueError("`sigma` must be 2-D with shape (R, N).")
    R, N = S.shape
    T = _validate_and_broadcast_t(t, (R, N))

    # --- Determine eligible rows ---
    eligible = rows_with_nonzero(S)

    # --- Choose the row to plot ---
    if row_idx is None:
        row_idx = _select_row(eligible, S, strategy=strategy, rng=rng)
    else:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= R:
            raise ValueError(f"`row_idx` out of range [0, {R-1}].")
        has_nonzero = bool((S[row_idx] != 0).any())
        if not has_nonzero:
            if empty == "fallback":
                row_idx = _select_row(eligible, S, strategy=strategy, rng=rng)
            elif empty == "plot_zero":
                pass  # proceed as zero row
            elif empty == "error":
                raise ValueError(f"Requested row_idx={row_idx} has no non-zero entries.")
            else:
                raise ValueError("`empty` must be one of {'error','fallback','plot_zero'}.")

    # --- Prepare data to plot ---
    x = T[row_idx]
    y = S[row_idx]
    if clip_max is not None:
        y = np.clip(y, 0.0, float(clip_max))

    # --- Plot ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        created_fig = True

    ax.plot(x, y, lw=1.6)
    if mark_nonzero:
        nz = (S[row_idx] != 0)
        ax.scatter(x[nz], y[nz], s=12, zorder=3)

    ax.set_xlabel("t")
    ax.set_ylabel("sigma")
    ax.set_title(f"Sigma along ray row {row_idx}")
    ax.grid(True, alpha=0.3)

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return row_idx


__all__ = [
    "to_numpy",
    "rows_with_nonzero",
    "pick_random_nonzero_entry",
    "plot_nonzero_sigma_row",
]


# --------------------------------------------------------------------------- #
# Demo / self-test usages (optional)
# --------------------------------------------------------------------------- #
def _demo_numpy_1d_t():
    print("[demo] numpy inputs, 1D t → broadcast")
    R, N = 5, 80
    t = np.linspace(0.0, 1.0, N)
    sigma = np.zeros((R, N))
    sigma[2, 15:30] = 1.2
    sigma[4, 50:60] = 0.8
    _ = plot_nonzero_sigma_row(t, sigma, strategy="max", mark_nonzero=True)


def _demo_numpy_2d_t_with_ax():
    print("[demo] numpy inputs, 2D t, shared axes plotting")
    R, N = 4, 64
    base_t = np.linspace(0.0, 1.0, N)
    T = np.vstack([base_t + 0.02 * i for i in range(R)])  # simple row-wise shift
    sigma = np.zeros((R, N))
    sigma[1, 5:18] = 1.0
    sigma[3, 40:55] = 2.0

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k in range(2):
        r = plot_nonzero_sigma_row(T, sigma, strategy="random", ax=ax, show=False)
        print(f"  plotted row {r}")
    ax.legend([f"random pick {i+1}" for i in range(2)], loc="upper right")
    plt.tight_layout()
    plt.show()


def _demo_torch_inputs():
    print("[demo] torch inputs also supported")
    if not _HAS_TORCH:
        print("  torch not available; skipping")
        return
    R, N = 3, 100
    t = torch.linspace(0, 2, N)  # type: ignore[attr-defined]
    sigma = torch.zeros(R, N)    # type: ignore[attr-defined]
    sigma[0, 20:35] = 3.0        # type: ignore[index]
    sigma[2, 60:80] = 1.0        # type: ignore[index]
    _ = plot_nonzero_sigma_row(t, sigma, mark_nonzero=True)


def _demo_pick_random_entry_then_plot_row():
    print("[demo] pick (r,c) where sigma!=0, then plot that row explicitly")
    R, N = 6, 90
    t = np.linspace(0.0, 1.0, N)
    sigma = np.zeros((R, N))
    sigma[1, 10:20] = 0.7
    sigma[4, 30:40] = 1.1

    r, c = pick_random_nonzero_entry(sigma)
    print(f"  picked non-zero entry at (r={r}, c={c})")
    _ = plot_nonzero_sigma_row(t, sigma, row_idx=r, clip_max=None, mark_nonzero=True)


if __name__ == "__main__":  # quick smoke demos
    _demo_numpy_1d_t()
    _demo_numpy_2d_t_with_ax()
    _demo_torch_inputs()
    _demo_pick_random_entry_then_plot_row()