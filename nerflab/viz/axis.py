# viz/axis.py
from __future__ import annotations
from typing import Sequence, Tuple, Iterable, Optional, List

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..config.viz_config import viz_cfg as VCFG

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
    Force *data* aspect ratio to be 1 : 1 : 1 after artists are drawn.
    Equivalent to `ax.set_aspect("equal")` for 2-D axes.
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
    length: Optional[float] = None,
    lw: float = 2.0,
    alpha: float = 1.0,
    zorder: int = 3,
):
    """
    Draw an RGB triad (X=red, Y=green, Z=blue) at `origin`.

    Returns
    -------
    list of Line2D artists
    """
    length = VCFG.axis_triad_len if length is None else length
    o = np.asarray(origin, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * length
    colors = ("r", "g", "b")
    lines = []
    for vec, c in zip(axes, colors, strict=True):
        p = o + vec
        ln, = ax.plot(
            (o[0], p[0]), (o[1], p[1]), (o[2], p[2]),
            c, lw=lw, alpha=alpha, zorder=zorder
        )
        lines.append(ln)
    return lines



def grid3d(
    ax: Axes,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    *,
    step: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    color: Optional[str] = None,
    lw: Optional[float] = None,
    alpha: Optional[float] = None,
    zorder: int = 2,
) -> List:
    """
    Draw a Cartesian lattice inside `bounds` as individual line primitives.

    Why lines (not a Line3DCollection)?
        Some tests count ax.lines; collections won't show up there. Using ax.plot(...)
        ensures those tests see the expected number of new line artists.

    Parameters
    ----------
    bounds : ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        Box inside which the grid is drawn (inclusive on endpoints).
    step   : (dx, dy, dz)
        Spacing between grid lines. Must be > 0 along each axis.
    color  : str, optional
        Line color. If None, falls back to Matplotlib defaults or your theme.
    lw     : float, optional
        Line width. Defaults to 0.3 if None.
    alpha  : float, optional
        Opacity in [0,1]. If None, use Matplotlib default or your theme.
    zorder : int
        Artist stacking order (larger == on top).

    Returns
    -------
    List[Line2D]
        The line artists added to the Axes (each ax.plot returns a list with one Line).
    """
    # Defaults (keep lightweight; grid lines should be subtle)
    lw = 0.3 if lw is None else lw

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dx, dy, dz = step

    # Validate
    if not (dx > 0 and dy > 0 and dz > 0):
        return []
    if not (xmax >= xmin and ymax >= ymin and zmax >= zmin):
        return []

    # Build coordinates (inclusive of upper bound)
    xs = np.arange(xmin, xmax + 1e-9, dx, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, dy, dtype=float)
    zs = np.arange(zmin, zmax + 1e-9, dz, dtype=float)

    lines: List = []

    # Lines parallel to X at each (y, z)
    for y in ys:
        for z in zs:
            lines += ax.plot(
                [xmin, xmax], [y, y], [z, z],
                color=color, linewidth=lw, alpha=alpha, zorder=zorder
            )

    # Lines parallel to Y at each (x, z)
    for x in xs:
        for z in zs:
            lines += ax.plot(
                [x, x], [ymin, ymax], [z, z],
                color=color, linewidth=lw, alpha=alpha, zorder=zorder
            )

    # Lines parallel to Z at each (x, y)
    for x in xs:
        for y in ys:
            lines += ax.plot(
                [x, x], [y, y], [zmin, zmax],
                color=color, linewidth=lw, alpha=alpha, zorder=zorder
            )

    return lines


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
    Apply a project-wide style to any 3-D axis.

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
