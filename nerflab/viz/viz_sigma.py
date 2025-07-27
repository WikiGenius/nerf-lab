# viz/sigma.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # keeps flake8 happy

from nerflab.viz.viz_config import viz_cfg as VCFG
from nerflab.viz.axis import style_3d_axis


# --------------------------------------------------------------------------- #
# σ-scatter -- 3-D view
# --------------------------------------------------------------------------- #
def viz_sigma_scatter(
    pts: np.ndarray,
    sigma: np.ndarray,
    *,
    max_points: int | None = None,
    cmap: str | None = None,
    add_triad: bool = True,
) -> None:
    """
    Scatter points in 3-D coloured by density/σ value.

    Parameters
    ----------
    pts  : (R, N, 3) sample positions
    sigma: (R, N)    densities
    """
    max_points = VCFG.sigma_max_points if max_points is None else max_points
    cmap       = VCFG.cmap            if cmap       is None else cmap

    P = pts.reshape(-1, 3).astype(np.float32)
    S = sigma.reshape(-1).astype(np.float32)

    # Optional subsampling
    if P.shape[0] > max_points:
        sel = np.random.choice(P.shape[0], max_points, replace=False)
        P, S = P[sel], S[sel]

    # Figure
    fig = plt.figure(figsize=VCFG.figsize, dpi=VCFG.dpi)
    ax  = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(*P.T, c=S, s=VCFG.scatter_size, cmap=cmap, depthshade=False)

    if add_triad:
        from nerflab.viz.axis import axis_triad
        axis_triad(ax, length=VCFG.axis_triad_len)

    # Axis styling
    style_3d_axis(ax,
                  invert=VCFG.axis_invert,
                  elev=VCFG.axis_elev,
                  azim=VCFG.axis_azim)

    fig.colorbar(sc, ax=ax, label="σ")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# σ-heatmap -- 2-D image view
# --------------------------------------------------------------------------- #
def viz_sigma_heatmap(
    sigma: np.ndarray,
    *,
    cmap: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Display σ values as a (ray × sample) heat-map.

    Parameters
    ----------
    sigma: (R, N) densities
    """
    cmap     = VCFG.heatmap_cmap if cmap     is None else cmap
    figsize  = VCFG.heatmap_size if figsize  is None else figsize or (8, 4)

    fig, ax = plt.subplots(figsize=figsize, dpi=VCFG.dpi)
    im = ax.imshow(sigma, aspect="auto", cmap=cmap)

    ax.set_xlabel("Sample index (N)")
    ax.set_ylabel("Ray index (R)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("σ")

    plt.tight_layout()
    plt.show()
