# viz/axis.py
from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, Iterable
from matplotlib.axes import Axes

from .viz_config import viz_cfg as VCFG

__all__ = [
    "set_equal",
    "axis_triad",
    "grid3d",
    "style_3d_axis",
]

# ----------------------------------------------------------------------------- #
# Internal helpers
# ----------------------------------------------------------------------------- #
def _max_span(a: Iterable[Tuple[float, float]]) -> float:
    """Largest interval length in a sequence of (min,max) pairs."""
    return max(high - low for low, high in a) / 2.0


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #
def set_equal(ax: Axes) -> None:
    """
    Force *data* aspect ratio to be 1 : 1 : 1 after artists are drawn.
    Equivalent to `ax.set_aspect("equal")` for 2‑D axes.
    """
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mid = tuple(map(np.mean, (xlim, ylim, zlim)))
    half = _max_span((xlim, ylim, zlim))
    ax.set_xlim3d(mid[0] - half, mid[0] + half)
    ax.set_ylim3d(mid[1] - half, mid[1] + half)
    ax.set_zlim3d(mid[2] - half, mid[2] + half)


def axis_triad(
    ax: Axes,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    length: float | None = None,
    lw: float = 2.0,
) -> None:
    """
    Draw an RGB triad (X = red, Y = green, Z = blue) at `origin`.
    """
    length = VCFG.axis_triad_len if length is None else length
    o = np.asarray(origin, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * length
    colors = ("r", "g", "b")
    for vec, c in zip(axes, colors, strict=True):
        p = o + vec
        ax.plot((o[0], p[0]), (o[1], p[1]), (o[2], p[2]), c, lw=lw)


def grid3d(
    ax: Axes,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    *,
    step: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    color: str | None = None,
    lw: float | None = None,
    alpha: float | None = None,
) -> None:
    """
    Draw a Cartesian lattice inside `bounds`.

    Parameters
    ----------
    bounds : ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    step   : Spacing (dx,dy,dz) between grid lines.
    """
    color = VCFG.grid_color if color is None else color
    lw = 0.3 if lw is None else lw
    alpha = VCFG.grid_alpha if alpha is None else alpha

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    xs = np.arange(xmin, xmax + 1e-9, step[0], dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, step[1], dtype=float)
    zs = np.arange(zmin, zmax + 1e-9, step[2], dtype=float)

    # Vectorised plotting: build segments for all three axis directions
    for y, z in np.array(np.meshgrid(ys, zs)).T.reshape(-1, 2):
        ax.plot((xmin, xmax), (y, y), (z, z), color=color, lw=lw, alpha=alpha)
    for x, z in np.array(np.meshgrid(xs, zs)).T.reshape(-1, 2):
        ax.plot((x, x), (ymin, ymax), (z, z), color=color, lw=lw, alpha=alpha)
    for x, y in np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2):
        ax.plot((x, x), (y, y), (zmin, zmax), color=color, lw=lw, alpha=alpha)


def style_3d_axis(
    ax: Axes,
    *,
    labels: Sequence[str] = ("X", "Y", "Z"),
    equal: bool = True,
    invert: Sequence[str] = (),
    elev: float | None = None,
    azim: float | None = None,
    box_aspect: Tuple[int, int, int] = (1, 1, 1),
) -> None:
    """
    Apply a project‑wide style to any 3‑D axis.

    Parameters
    ----------
    invert : iterable containing any of {"x", "y", "z"} to flip.
    """
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    if equal:
        set_equal(ax)

    for axis_name in invert:
        getattr(ax, f"invert_{axis_name}axis")()

    ax.set_box_aspect(box_aspect)
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
