# viz/world.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Iterable, Literal, Optional, Tuple

from nerflab import Camera
from .axis import style_3d_axis, axis_triad, grid3d
from .viz_config import viz_cfg as CFG

NDArray = np.ndarray


def plot_cloud_batch_with_camera(
    points: NDArray, 
    H: np.ndarray,
    *,
    # ────── styling overrides ─────────────────────────────────────────────
    point_size: int                    = 3,
    camera_scale: float               = 0.4,
    figsize: Optional[Tuple[int, int]] = None,
    elev: Optional[float]             = None,
    azim: Optional[float]             = None,
    invert_x: Optional[bool]          = None,
    grid_lines: Optional[bool]        = None,
    grid_bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    grid_step: Optional[Tuple[float, float, float]]      = None,
) -> None:
    """
    Visualise a batch of 3D point-rays and a single camera pose H (4×4 world→camera).

    Parameters
    ----------
    points     : ndarray, shape (R, N, 3)
                 R = number of rays, N = samples per ray;  
                 this will be flattened to (R·N, 3) for plotting.
    H          : ndarray, shape (4, 4)
                 Camera pose matrix H_wc.
    point_size : int
                 Scatter marker size for the points.
    camera_scale: float
                 Scale of the camera axis triad.
    figsize    : (w, h)
                 Figure size in inches.
    elev, azim : float
                 3D view elevation & azimuth (deg).
    invert_x   : bool
                 Flip X-axis for left-handed convention.
    grid_*     : optional grid-line config (see `plot_world`).
    """
    # ── cfg-driven defaults ───────────────────────────────────────────
    figsize    = CFG.figsize     if figsize    is None else figsize
    elev       = CFG.axis_elev   if elev       is None else elev
    azim       = CFG.axis_azim   if azim       is None else azim
    invert_x   = ("x" in CFG.axis_invert) if invert_x is None else invert_x
    grid_lines = CFG.grid_lines  if grid_lines is None else grid_lines
    grid_bounds = CFG.grid_bounds if grid_bounds is None else grid_bounds
    grid_step   = CFG.grid_step   if grid_step is None else grid_step

    # ── create figure / axis ─────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=CFG.dpi)
    ax  = fig.add_subplot(111, projection="3d")

    # ── plot the batch of points ──────────────────────────────────────
    pts = points.reshape(-1, 3)
    ax.scatter(*pts.T, s=point_size, c="C0", depthshade=False, label="Batch points")

    # ── plot the camera center + axes ─────────────────────────────────
    cam_pos = H[:3, 3]
    ax.scatter(*cam_pos, s=CFG.camera_marker_size, c="red", marker="o", label="Camera")
    Camera._draw_pose_axes(ax, H, scale=camera_scale)

    # ── optional lattice grid ─────────────────────────────────────────
    if grid_lines:
        grid3d(ax, bounds=grid_bounds, step=grid_step)

    # ── world axis triad & styling ────────────────────────────────────
    axis_triad(ax, length=CFG.axis_triad_len)
    style_3d_axis(
        ax,
        invert=("x",) if invert_x else (),
        elev=elev,
        azim=azim,
    )

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()
