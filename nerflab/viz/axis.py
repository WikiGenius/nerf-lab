# viz/axis.py
from __future__ import annotations
from typing import Sequence, Tuple, Iterable, Optional

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
) -> Line3DCollection:
    """
    Draw a Cartesian lattice inside `bounds` as a single Line3DCollection.

    Parameters
    ----------
    bounds : ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    step   : spacing (dx,dy,dz) between grid lines
    color  : line color (defaults to VCFG.grid_color)
    lw     : line width (defaults to 0.3)
    alpha  : opacity 0..1 (defaults to VCFG.grid_alpha)
    zorder : artist stacking order

    Returns
    -------
    Line3DCollection
    """
    color = VCFG.grid_color if color is None else color
    lw = 0.3 if lw is None else lw
    alpha = VCFG.grid_alpha if alpha is None else alpha

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dx, dy, dz = step

    xs = np.arange(xmin, xmax + 1e-9, dx, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, dy, dtype=float)
    zs = np.arange(zmin, zmax + 1e-9, dz, dtype=float)

    segs = []

    # Lines parallel to X at each (y,z)
    YZ = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)  # (Ny*Nz,2)
    if YZ.size:
        x0 = np.full((YZ.shape[0], 1), xmin); x1 = np.full((YZ.shape[0], 1), xmax)
        segs.append(np.stack(
            [np.hstack([x0,  YZ[:, :1], YZ[:, 1:]]),
             np.hstack([x1,  YZ[:, :1], YZ[:, 1:]])],
            axis=1
        ))  # (Ny*Nz, 2, 3)

    # Lines parallel to Y at each (x,z)
    XZ = np.stack(np.meshgrid(xs, zs, indexing="ij"), axis=-1).reshape(-1, 2)  # (Nx*Nz,2)
    if XZ.size:
        y0 = np.full((XZ.shape[0], 1), ymin); y1 = np.full((XZ.shape[0], 1), ymax)
        segs.append(np.stack(
            [np.hstack([XZ[:, :1], y0, XZ[:, 1:]]),
             np.hstack([XZ[:, :1], y1, XZ[:, 1:]])],
            axis=1
        ))  # (Nx*Nz, 2, 3)

    # Lines parallel to Z at each (x,y)
    XY = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)  # (Nx*Ny,2)
    if XY.size:
        z0 = np.full((XY.shape[0], 1), zmin); z1 = np.full((XY.shape[0], 1), zmax)
        segs.append(np.stack(
            [np.hstack([XY, z0]),
             np.hstack([XY, z1])],
            axis=1
        ))  # (Nx*Ny, 2, 3)

    if segs:
        segs = np.concatenate(segs, axis=0)  # (N, 2, 3)
    else:
        segs = np.empty((0, 2, 3), dtype=float)

    coll = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    coll.set_zorder(zorder)
    ax.add_collection3d(coll)
    return coll


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
