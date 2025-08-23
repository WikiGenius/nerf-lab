# nerflab/viz/render.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import os
import builtins as _bi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm


# -----------------------------------------------------------------------------
# Intrinsics “duck type” (replace with your real `Intrinsics` type if you like)
# -----------------------------------------------------------------------------
class _HasHW:
    height: int
    width: int


# -----------------------------------------------------------------------------
# Prebuilt B/W colormap for crisp binary masks
# -----------------------------------------------------------------------------
_BINARY_CMAP = ListedColormap(["black", "white"])
_BINARY_NORM = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=_BINARY_CMAP.N)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BinaryRenderCfg:
    """
    Configuration for rendering 0/1 (binary) images.

    Attributes
    ----------
    round_ndigits : int | None
        If not None, round values before thresholding (e.g., 2). Helps stabilize visuals.
    threshold : float
        Values >= threshold -> 1 (white), else 0 (black).
    interpolation : str
        Matplotlib interpolation ('nearest', 'none', 'bilinear', ...).
    show_axis : bool
        Whether to show axis ticks/frame.
    figsize : (float, float) | None
        Figure size; None uses rcParams default.
    dpi : int | None
        Figure DPI; None uses rcParams default.
    """
    round_ndigits: Optional[int] = 2
    threshold: float = 0.5
    interpolation: str = "nearest"
    show_axis: bool = False
    figsize: Optional[Tuple[float, float]] = None
    dpi: Optional[int] = None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _as_int(x: Any) -> int:
    """Convert various numeric-like things (callables, numpy scalars, torch tensors) to Python int."""
    if callable(x):
        x = x()
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            x = x.item()
    except Exception:
        pass
    if isinstance(x, np.generic):
        x = np.asarray(x).item()
    return _bi.int(x)  # use real built-in

def _normalize_to_numpy(a: Any) -> np.ndarray:
    """Accept np.ndarray, sequences, or torch.Tensor and return a numpy array (no copy if possible)."""
    if isinstance(a, np.ndarray):
        return a
    try:
        import torch  # type: ignore
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def _get_hw_from_intr(intr: Any) -> Tuple[int, int]:
    """
    Extract (H, W) from an intrinsics-like object supporting common attribute spellings.
    Accepts .height/.width, .H/.W, .rows/.cols, or .image_height/.image_width.
    Attributes may be numeric or callables.
    """
    candidates = [
        ("height", "width"),
        ("H", "W"),
        ("rows", "cols"),
        ("image_height", "image_width"),
    ]
    for hn, wn in candidates:
        if hasattr(intr, hn) and hasattr(intr, wn):
            try:
                H = _as_int(getattr(intr, hn))
                W = _as_int(getattr(intr, wn))
                if H > 0 and W > 0:
                    return H, W
            except Exception:
                continue
    raise ValueError(
        "Could not extract (height, width) from `intr`. "
        "Ensure it exposes numeric attributes/callables like `.height`/`.width`."
    )


def _ensure_hw(
    C_flat: np.ndarray,
    intr: Optional[_HasHW] = None,
    hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Determine (H, W) robustly from `intr` or explicit `hw`, and validate C.size.
    Priority: hw > intr > infer square (last resort).
    """
    if hw is not None:
        H, W = _as_int(hw[0]), _as_int(hw[1])
    elif intr is not None:
        H, W = _get_hw_from_intr(intr)
    else:
        N = np.size(C_flat)  # no int() cast; safe even if `int` is shadowed
        r = _bi.int(round(N ** 0.5))
        if r * r != N:
            raise ValueError(
                "Cannot infer (H, W): provide `intr` or `hw`, "
                f"and ensure C.size == H*W (got size={N})."
            )
        H, W = r, r

    if np.size(C_flat) != H * W:
        raise ValueError(
            f"Size mismatch: C.size={np.size(C_flat)} but H*W={H*W}. "
            "Provide consistent `intr` or `hw`."
        )
    return H, W


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def render_binary_image(
    C: Any,
    *,
    intr: Optional[_HasHW] = None,
    hw: Optional[Tuple[int, int]] = None,
    cfg: BinaryRenderCfg = BinaryRenderCfg(),
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    return_fig_ax: bool = True,
) -> Tuple[Figure, Axes, np.ndarray] | np.ndarray:
    """
    Render a binary (0/1) image from a flattened or (H, W) array using a black/white colormap.

    Parameters
    ----------
    C : array-like or torch.Tensor
        Input array of shape (H*W,) or (H, W). Values will be binarized for display.
    intr : object with `.height` and `.width`, optional
        Intrinsics-like object to infer (H, W) when C is flat.
    hw : (H, W), optional
        Explicit image shape (overrides `intr`).
    cfg : BinaryRenderCfg
        Rendering configuration (rounding, threshold, interpolation, axis, size).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; if None, a new figure/axes is created.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, saves to this path (directories created as needed).
    return_fig_ax : bool
        If True (default), return (fig, ax, im2d). If False, return only the 2D image array.

    Returns
    -------
    (fig, ax, im2d) or im2d
        - fig : matplotlib.figure.Figure
        - ax  : matplotlib.axes.Axes
        - im2d: (H, W) np.ndarray of dtype float32 (0.0 or 1.0) used for rendering.

    Notes
    -----
    - This function does NOT call `plt.show()`.
    - Uses a fixed black/white colormap with hard boundaries for crisp binary visuals.
    """
    C_np = _normalize_to_numpy(C)
    if C_np.ndim == 2 and hw is None and intr is None:
        H, W = C_np.shape
        C_flat = C_np.reshape(-1)
    else:
        C_flat = C_np.reshape(-1)
        H, W = _ensure_hw(C_flat, intr=intr, hw=hw)

    # Optional rounding (useful after numeric ops for stable visuals)
    if cfg.round_ndigits is not None:
        C_flat = np.round(C_flat.astype(np.float32), cfg.round_ndigits)
    else:
        C_flat = C_flat.astype(np.float32, copy=False)

    # Binarize (>= threshold -> 1.0, else 0.0)
    im2d = (C_flat >= float(cfg.threshold)).astype(np.float32, copy=False).reshape(H, W)

    # Figure / axes
    if ax is None:
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    # Render
    ax.imshow(
        im2d,
        cmap=_BINARY_CMAP,
        norm=_BINARY_NORM,
        interpolation=cfg.interpolation,
        origin="upper",
    )
    if not cfg.show_axis:
        ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return (fig, ax, im2d) if return_fig_ax else im2d


# -----------------------------------------------------------------------------
# OO wrapper
# -----------------------------------------------------------------------------
class Renderer:
    """
    Lightweight renderer wrapper to keep consistent plotting style across your project.

    Example
    -------
    >>> rnd = Renderer(cfg=BinaryRenderCfg(threshold=0.5))
    >>> fig, ax, mask = rnd.binary(C, intr=intr, title="Opacity Mask")
    """
    def __init__(self, cfg: Optional[BinaryRenderCfg] = None) -> None:
        self.cfg = cfg or BinaryRenderCfg()

    def binary(
        self,
        C: Any,
        *,
        intr: Optional[_HasHW] = None,
        hw: Optional[Tuple[int, int]] = None,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = True,
    ):
        return render_binary_image(
            C,
            intr=intr,
            hw=hw,
            cfg=self.cfg,
            ax=ax,
            title=title,
            save_path=save_path,
            return_fig_ax=return_fig_ax,
        )

__all__ = ["Renderer"]
