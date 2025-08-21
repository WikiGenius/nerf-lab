from __future__ import annotations
from functools import lru_cache
from typing import Any, Tuple, Optional, Sequence

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from ..config.viz_config import viz_cfg as VCFG

__all__ = ["plot_box", "plot_sphere"]

# --------------------------------------------------------------------------- #
#  Precomputed constants (avoid realloc per call)
# --------------------------------------------------------------------------- #
# Corner selector over mins/maxs for each axis → builds 8 vertices in one go.
_BOX_VERT_SEL = np.array(
    [
        [0, 0, 0],  # (xmin, ymin, zmin)
        [1, 0, 0],  # (xmax, ymin, zmin)
        [0, 1, 0],  # (xmin, ymax, zmin)
        [1, 1, 0],  # (xmax, ymax, zmin)
        [0, 0, 1],  # (xmin, ymin, zmax)
        [1, 0, 1],  # (xmax, ymin, zmax)
        [0, 1, 1],  # (xmin, ymax, zmax)
        [1, 1, 1],  # (xmax, ymax, zmax)
    ],
    dtype=np.int8,
)

# Faces as quads (indices into the 8 vertices above)
_BOX_FACES = np.array(
    [
        [0, 1, 3, 2],  # −Z
        [4, 5, 7, 6],  # +Z
        [0, 1, 5, 4],  # −Y
        [2, 3, 7, 6],  # +Y
        [0, 2, 6, 4],  # −X
        [1, 3, 7, 5],  # +X
    ],
    dtype=np.int8,
)


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _as_float3(x: Any) -> Tuple[float, float, float]:
    """Coerce a 3-vector (torch/numpy/sequence) to 3 floats."""
    arr = np.asarray(x, dtype=np.float32).reshape(3)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _as_float(x: Any) -> float:
    """Coerce scalar (torch/numpy/python) to float."""
    return float(np.asarray(x, dtype=np.float32).reshape(()))


def _rgba(color: Any, alpha: float) -> Tuple[float, float, float, float]:
    """Convert any Matplotlib color to RGBA with given alpha."""
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, float(alpha))


@lru_cache(maxsize=8)
def _sphere_uv(res: int):
    """Cache sin/cos parameterization for a given resolution."""
    u = np.linspace(0.0, 2.0 * np.pi, res, dtype=np.float32)
    v = np.linspace(0.0, np.pi, res, dtype=np.float32)
    return np.sin(u), np.cos(u), np.sin(v), np.cos(v), np.ones_like(u, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Primitive plotters
# --------------------------------------------------------------------------- #
def plot_box(
    ax: Any,
    box: Any,
    *,
    facecolor: Any = "C0",
    edgecolor: Any = "k",
    face_alpha: Optional[float] = None,
    edge_alpha: Optional[float] = None,
    linewidth: float = 0.6,
    zorder: int = 1,
) -> Poly3DCollection:
    """
    Draw an axis-aligned box as a single Poly3DCollection.

    Parameters
    ----------
    ax          : 3D Matplotlib axis
    box         : exposes `bounds` ⇒ ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    facecolor   : face color
    edgecolor   : edge color
    face_alpha  : face opacity (defaults to VCFG.default_alpha)
    edge_alpha  : edge opacity  (defaults to face_alpha if None)
    linewidth   : edge width
    zorder      : collection z-order

    Returns
    -------
    Poly3DCollection
    """
    fa = VCFG.default_alpha if face_alpha is None else float(face_alpha)
    ea = fa if edge_alpha is None else float(edge_alpha)

    # Ensure numeric (float32) bounds of shape (3,2) and sorted per axis
    b = np.asarray(box.bounds, dtype=np.float32).reshape(3, 2)
    b = np.sort(b, axis=1)  # (min,max) ordering per axis
    mins = b[:, 0]
    maxs = b[:, 1]

    # Build 8 vertices in one vectorized step
    verts = np.where(_BOX_VERT_SEL, maxs, mins).astype(np.float32)  # (8,3)

    poly = Poly3DCollection(
        verts[_BOX_FACES],  # (6,4,3)
        facecolor=_rgba(facecolor, fa),
        edgecolor=_rgba(edgecolor, ea),
        linewidths=float(linewidth),
        zorder=int(zorder),
    )
    ax.add_collection3d(poly)
    return poly


def plot_sphere(
    ax: Any,
    sphere: Any,
    *,
    color: Any = "C1",
    face_alpha: Optional[float] = None,
    res: int = 32,
    zorder: int = 1,
    shade: bool = True,
    antialiased: bool = False,
):
    """
    Draw a sphere via Axes3D.plot_surface (returns a Poly3DCollection-like artist).

    Parameters
    ----------
    ax          : 3D Matplotlib axis
    sphere      : exposes `.center` (cx,cy,cz) and `.radius`
    color       : surface color
    face_alpha  : surface opacity (defaults to VCFG.default_alpha)
    res         : samples along u & v (>= 8 recommended)
    zorder      : artist z-order
    shade       : enable shading
    antialiased : surface antialiasing

    Returns
    -------
    artist (Poly3DCollection-like)
    """
    fa = VCFG.default_alpha if face_alpha is None else float(face_alpha)

    cx, cy, cz = _as_float3(sphere.center)
    r = _as_float(sphere.radius)
    if r <= 0.0:
        raise ValueError(f"sphere.radius must be > 0, got {r}")

    res = int(max(8, res))
    sin_u, cos_u, sin_v, cos_v, ones_u = _sphere_uv(res)

    # Broadcasted parameterization (res x res) in float32
    x = cx + r * (cos_u[:, None] * sin_v[None, :])
    y = cy + r * (sin_u[:, None] * sin_v[None, :])
    z = cz + r * (ones_u[:, None] * cos_v[None, :])

    artist = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=bool(antialiased),
        shade=bool(shade),
        color=_rgba(color, fa),
        zorder=int(zorder),
    )
    return artist
