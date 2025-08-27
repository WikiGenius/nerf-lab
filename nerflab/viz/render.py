# nerflab/viz/render.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Literal

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
        Preferred figure size in inches when not saving (or when save_via='mpl' without hw/intr).
    dpi : int | None
        Figure DPI when using Matplotlib.
    save_via : 'pil' | 'mpl'
        - 'pil' saves the rendered binary array directly via Pillow (exact pixels; fastest).
        - 'mpl' uses Matplotlib (for on-figure titles/overlays); we still pin W×H on save.
    save_verify : bool
        If True, reopen the saved file and assert its size == (W, H).
    """
    round_ndigits: Optional[int] = 2
    threshold: float = 0.5
    interpolation: str = "nearest"
    show_axis: bool = False
    figsize: Optional[Tuple[float, float]] = None
    dpi: Optional[int] = None
    save_via: Literal["pil", "mpl"] = "pil"
    save_verify: bool = False


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
        N = np.size(C_flat)
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


def _save_exact_pil(im2d: np.ndarray, save_path: str) -> None:
    """Save a (H,W) float32/float64/uint8 array as L-mode PNG with exact pixels."""
    from PIL import Image
    if im2d.dtype != np.uint8:
        im_u8 = np.clip(im2d * 255.0, 0, 255).astype(np.uint8)
    else:
        im_u8 = im2d
    Image.fromarray(im_u8, mode="L").save(save_path)


def _verify_saved_size(save_path: str, expected_hw: Tuple[int, int]) -> None:
    """Reopen image and assert (W,H) == expected (W,H)."""
    from PIL import Image
    Wexp, Hexp = expected_hw[1], expected_hw[0]
    w, h = Image.open(save_path).size
    assert (w, h) == (Wexp, Hexp), f"Saved {w}x{h}, expected {Wexp}x{Hexp} for {save_path}"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def render_binary_image(
    C: Any,
    *,
    intr: Optional[_HasHW] = None,
    hw: Optional[Tuple[int, int]] = None,   # (H, W) here to match NumPy shape semantics
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
        Explicit image shape (overrides `intr`). Note: (H,W) ordering here.
    cfg : BinaryRenderCfg
        Rendering configuration (rounding, threshold, interpolation, axis, size, saving policy).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; if None, a new figure/axes is created (only if needed).
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
        - im2d: (H, W) np.ndarray of dtype uint8 (0 or 255) if saved via PIL,
                or float32 (0.0 or 1.0) for rendering context.
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

    # Fast path: if saving via PIL and no Matplotlib axes requested
    if save_path is not None and cfg.save_via == "pil" and ax is None and not return_fig_ax:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        _save_exact_pil(im2d, save_path)
        if cfg.save_verify:
            _verify_saved_size(save_path, (H, W))
        return im2d

    # Otherwise, use Matplotlib for display (and optionally for saving)
    created_fig = False
    if ax is None:
        # If we know W,H and want exact pixels on MPL save, set inches from DPI
        dpi = cfg.dpi or plt.rcParams.get("figure.dpi", 100)
        if cfg.save_via == "mpl" and (intr is not None or hw is not None):
            fig_w_in = W / dpi
            fig_h_in = H / dpi
            fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        else:
            fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111)
        created_fig = True
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

    # IMPORTANT: don't use tight bbox (it changes pixel size on save)
    # Also avoid fig.tight_layout() here for exact-pixel saves.
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if cfg.save_via == "pil":
            # Save via PIL (exact pixels)
            _save_exact_pil(im2d, save_path)
        else:
            # Save via Matplotlib while pinning exact pixels:
            # Ensure DPI/inches are consistent with W×H
            dpi = cfg.dpi or fig.get_dpi()
            # If figure was not created with pinned inches, pin it now:
            if (intr is not None or hw is not None):
                fig.set_size_inches(W / dpi, H / dpi, forward=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
        if cfg.save_verify:
            _verify_saved_size(save_path, (H, W))

    # Return
    if return_fig_ax:
        return fig, ax, im2d
    else:
        # Clean up created figure if not needed
        if created_fig:
            plt.close(fig)
        return im2d


# -----------------------------------------------------------------------------
# OO wrapper
# -----------------------------------------------------------------------------
class Renderer:
    """
    Lightweight renderer wrapper to keep consistent plotting style across your project.

    Example
    -------
    >>> rnd = Renderer(cfg=BinaryRenderCfg(threshold=0.5, save_via="pil"))
    >>> fig, ax, mask = rnd.binary(C, intr=intr, title="Opacity Mask")
    """
    def __init__(self, cfg: Optional[BinaryRenderCfg] = None) -> None:
        self.cfg = cfg or BinaryRenderCfg()

    def binary(
        self,
        C: Any,
        *,
        intr: Optional[_HasHW] = None,
        hw: Optional[Tuple[int, int]] = None,  # (H, W)
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

__all__ = ["Renderer", "BinaryRenderCfg", "render_binary_image"]
