# nerflab/viz/viz_sigma.py
from __future__ import annotations
from typing import Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # keeps flake8 happy

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from ..config.viz_config import viz_cfg as VCFG
from .axis import style_3d_axis


# ----------------------------- utils -------------------------------- #
def _to_numpy_f32(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Return a contiguous float32 NumPy array (moves tensors to CPU, detaches)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        # detach (we're only visualizing), move to CPU, ensure contiguous
        x = x.detach().to("cpu")
        # torch→numpy shares storage when possible; astype(copy=False) keeps it zero-copy if already f32
        arr = x.numpy()
        return np.asarray(arr, dtype=np.float32)
    # NumPy path
    return np.asarray(x, dtype=np.float32)


def _validate_shapes_scatter(pts: np.ndarray, sigma: np.ndarray) -> Tuple[int, int]:
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"`pts` must be shape (R, N, 3); got {pts.shape}")
    if sigma.ndim != 2 or sigma.shape[0] != pts.shape[0] or sigma.shape[1] != pts.shape[1]:
        raise ValueError(f"`sigma` must be shape (R, N) matching pts; got {sigma.shape} vs {pts.shape}")
    return int(pts.shape[0]), int(pts.shape[1])


# --------------------------------------------------------------------------- #
# σ-scatter -- 3-D view
# --------------------------------------------------------------------------- #
def viz_sigma_scatter(
    pts: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    max_points: Optional[int] = None,
    cmap: Optional[str] = None,
    add_triad: bool = True,
) -> None:
    """
    Scatter 3D sample points colored by σ.

    Parameters
    ----------
    pts  : (R, N, 3) np.ndarray or torch.Tensor
    sigma: (R, N)    np.ndarray or torch.Tensor
    max_points : int | None
        If set, randomly subsample this many points for faster plotting.
        Defaults to VCFG.sigma_max_points.
    cmap : str | None
        Matplotlib colormap name; defaults to VCFG.cmap.
    add_triad : bool
        If True, draw a small XYZ triad in world coordinates.
    """
    # Convert inputs → NumPy float32
    Pn = _to_numpy_f32(pts)
    Sn = _to_numpy_f32(sigma)

    # Validate shapes and flatten
    _R, _N = _validate_shapes_scatter(Pn, Sn)
    P = Pn.reshape(-1, 3)  # (R*N, 3)
    S = Sn.reshape(-1)     # (R*N,)

    # Resolve defaults
    max_points = VCFG.sigma_max_points if max_points is None else max_points
    cmap = VCFG.cmap if cmap is None else cmap

    # Optional subsampling for speed
    M = P.shape[0]
    if isinstance(max_points, int) and max_points > 0 and M > max_points:
        rng = np.random.default_rng()        # faster & thread-safe
        sel = rng.choice(M, size=max_points, replace=False)
        P, S = P[sel], S[sel]

    # Handle empty after subsample
    if P.size == 0:
        fig = plt.figure(figsize=VCFG.figsize, dpi=VCFG.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("No points to display")
        style_3d_axis(ax,
                      invert=VCFG.axis_invert,
                      elev=VCFG.axis_elev,
                      azim=VCFG.axis_azim)
        plt.tight_layout()
        plt.show()
        return

    # Figure
    fig = plt.figure(figsize=VCFG.figsize, dpi=VCFG.dpi)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                    c=S, s=VCFG.scatter_size, cmap=cmap, depthshade=False)

    if add_triad:
        # local import avoids circulars in some setups
        from nerflab.viz.axis import axis_triad
        axis_triad(ax, length=VCFG.axis_triad_len)

    # Axis styling
    style_3d_axis(ax,
                  invert=VCFG.axis_invert,
                  elev=VCFG.axis_elev,
                  azim=VCFG.axis_azim)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("σ")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# σ-heatmap -- 2-D image view
# --------------------------------------------------------------------------- #
def viz_sigma_heatmap(
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Display σ values as a (ray × sample) heatmap.

    Parameters
    ----------
    sigma  : (R, N) np.ndarray or torch.Tensor
    cmap   : str | None      → defaults to VCFG.heatmap_cmap
    figsize: (w, h) | None   → defaults to VCFG.heatmap_size or (8, 4)
    """
    S = _to_numpy_f32(sigma)
    if S.ndim != 2:
        raise ValueError(f"`sigma` must be shape (R, N); got {S.shape}")

    cmap = VCFG.heatmap_cmap if cmap is None else cmap
    figsize = VCFG.heatmap_size if figsize is None else (figsize or (8, 4))

    fig, ax = plt.subplots(figsize=figsize, dpi=VCFG.dpi)
    im = ax.imshow(S, aspect="auto", cmap=cmap)

    ax.set_xlabel("Sample index (N)")
    ax.set_ylabel("Ray index (R)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("σ")

    plt.tight_layout()
    plt.show()

__all__ = ["viz_sigma_heatmap", "viz_sigma_scatter"]
