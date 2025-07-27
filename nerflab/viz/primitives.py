from __future__ import annotations
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .viz_config import viz_cfg as VCFG

# --------------------------------------------------------------------------- #
#  Pre‑computed constants (avoids reallocating every call)
# --------------------------------------------------------------------------- #
_BOX_FACES = np.array(
    [[0, 1, 3, 2],  # −Z face
     [4, 5, 7, 6],  # +Z
     [0, 1, 5, 4],  # −Y
     [2, 3, 7, 6],  # +Y
     [0, 2, 6, 4],  # −X
     [1, 3, 7, 5]], # +X
    dtype=np.int8,
)

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
    Draw an axis‑aligned box.

    Parameters
    ----------
    ax        : A 3‑D matplotlib axis.
    box       : Object exposing `bounds` → ((xmin,xmax), (ymin,ymax), (zmin,zmax)).
    alpha     : Transparency override.  Falls back to cfg.default_alpha.
    edgecolor : Line colour of box edges.
    facecolor : Face (patch) colour.
    """
    alpha = VCFG.default_alpha if alpha is None else alpha
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box.bounds

    # Eight vertices, laid out in (x,y,z) order
    verts = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ],
        dtype=np.float32,
    )

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
    ax     : 3‑D matplotlib axis.
    sphere : Object exposing `.center` (cx,cy,cz) and `.radius`.
    alpha  : Transparency override (falls back to cfg.default_alpha).
    res    : Number of samples along u & v axes. 32 is a decent default.
    color  : Surface colour.
    """
    alpha = VCFG.default_alpha if alpha is None else alpha
    u = np.linspace(0.0, 2.0 * np.pi, res, dtype=np.float32)
    v = np.linspace(0.0, np.pi, res, dtype=np.float32)

    sin_v, cos_v = np.sin(v), np.cos(v)
    sin_u, cos_u = np.sin(u), np.cos(u)

    cx, cy, cz = sphere.center
    r = sphere.radius

    # Broadcasted outer‑products for surface parameterisation
    x = cx + r * np.outer(cos_u, sin_v)
    y = cy + r * np.outer(sin_u, sin_v)
    z = cz + r * np.outer(np.ones_like(u), cos_v)

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
