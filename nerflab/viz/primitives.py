from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Any, Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..config.viz_config import viz_cfg as VCFG

__all__ = ["plot_box", "plot_sphere"]

# --------------------------------------------------------------------------- #
#  Pre-computed constants (avoids reallocating every call)
# --------------------------------------------------------------------------- #
# Vertex index patterns for a box, encoded as 0/1 selectors over (min,max)
# for each axis. This lets us build the 8 corners with a single np.where.
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


def _as_float3(x: Any) -> Tuple[float, float, float]:
    """Coerce a 3-vector (torch / numpy / sequence) to 3 floats."""
    arr = np.asarray(x, dtype=np.float32).reshape(3)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _as_float(x: Any) -> float:
    """Coerce scalar (torch / numpy / python) to float."""
    return float(np.asarray(x, dtype=np.float32).reshape(()))


@lru_cache(maxsize=8)
def _sphere_uv(res: int):
    """Cache sin/cos parameterization for a given resolution."""
    u = np.linspace(0.0, 2.0 * np.pi, res, dtype=np.float32)
    v = np.linspace(0.0, np.pi, res, dtype=np.float32)
    sin_v, cos_v = np.sin(v), np.cos(v)
    sin_u, cos_u = np.sin(u), np.cos(u)
    ones_u = np.ones_like(u, dtype=np.float32)
    return sin_u, cos_u, sin_v, cos_v, ones_u


# --------------------------------------------------------------------------- #
#  Primitive plotters
# --------------------------------------------------------------------------- #
def plot_box(
    ax,
    box,
    *,
    alpha: float | None = None,
    edgecolor: str = "k",
    facecolor: str = "C0",
) -> None:
    """
    Draw an axis-aligned box.

    Parameters
    ----------
    ax        : A 3-D matplotlib axis.
    box       : Object exposing `bounds` → ((xmin,xmax), (ymin,ymax), (zmin,zmax)).
    alpha     : Transparency override.  Falls back to cfg.default_alpha.
    edgecolor : Line colour of box edges.
    facecolor : Face (patch) colour.
    """
    alpha = VCFG.default_alpha if alpha is None else alpha

    # Ensure numeric (float32) bounds → mins, maxs shape (3,)
    b = np.asarray(box.bounds, dtype=np.float32)  # (3,2)
    mins = b[:, 0]
    maxs = b[:, 1]

    # Build 8 vertices in one go using selector patterns
    # verts[i, :] = where(_BOX_VERT_SEL[i], maxs, mins)
    verts = np.where(_BOX_VERT_SEL, maxs, mins).astype(np.float32)

    poly = Poly3DCollection(
        verts[_BOX_FACES],
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidths=0.6,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def plot_sphere(
    ax,
    sphere,
    *,
    alpha: float | None = None,
    res: int = 32,
    color: str = "C1",
) -> None:
    """
    Draw a sphere.

    Parameters
    ----------
    ax     : 3-D matplotlib axis.
    sphere : Object exposing `.center` (cx,cy,cz) and `.radius`.
    alpha  : Transparency override (falls back to cfg.default_alpha).
    res    : Number of samples along u & v axes. 32 is a decent default.
    color  : Surface colour.
    """
    alpha = VCFG.default_alpha if alpha is None else alpha

    cx, cy, cz = _as_float3(sphere.center)
    r = _as_float(sphere.radius)

    sin_u, cos_u, sin_v, cos_v, ones_u = _sphere_uv(int(res))

    # Broadcasted parameterization (res x res) in float32
    # x = cx + r * cos(u) * sin(v)
    # y = cy + r * sin(u) * sin(v)
    # z = cz + r * cos(v)
    x = cx + r * (cos_u[:, None] * sin_v[None, :])
    y = cy + r * (sin_u[:, None] * sin_v[None, :])
    z = cz + r * (ones_u[:, None] * cos_v[None, :])

    ax.plot_surface(
        x,
        y,
        z,
        linewidth=0,
        antialiased=False,
        facecolors=None,
        shade=True,
        color=color,
        alpha=alpha,
    )
