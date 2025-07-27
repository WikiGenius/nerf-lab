# viz/world.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Iterable, Literal, Optional, Tuple

from nerflab.geometry import Box, Sphere
from nerflab import Camera
from .primitives import plot_box, plot_sphere
from .axis import style_3d_axis, axis_triad, grid3d
from .viz_config import viz_cfg as CFG

NDArray = np.ndarray


# -----------------------------------------------------------------------------
def plot_world(
    world,
    *,
    # ───────── camera / ray controls ───────────────────────────────────────
    cameras: Optional[Iterable[Camera]] = None,
    draw_rays: bool = False,
    ray_step: Optional[int] = None,
    ray_mode: Optional[Literal["lines", "quiver", "points"]] = None,
    # ───────── external point cloud (only for ray_mode=="points") ──────────
    points: Optional[NDArray] = None,        # shape (R, N, 3)
    point_size: int = 3,
    # ───────── world occupancy sampling ────────────────────────────────────
    sample_grid: bool = False,
    grid_bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    grid_res: Optional[Tuple[int, int, int]] = None,
    # ───────── lattice grid lines ──────────────────────────────────────────
    grid_lines: Optional[bool] = None,
    grid_step: Optional[Tuple[float, float, float]] = None,
    # ───────── view / style tweaks ─────────────────────────────────────────
    invert_x: Optional[bool] = None,
    figsize: Optional[Tuple[int, int]] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    # ───────── optional near / far override for ALL cameras ───────────────
    t_near: Optional[float] = None,
    t_far: Optional[float] = None,
) -> None:
    """
    Visualise a `World` plus optional `Camera` objects, rays, and a point cloud.

    Parameters
    ----------
    cameras        : Iterable of Camera objects.
    draw_rays      : If *True*, draw rays for each camera.
    ray_step       : Pixel subsampling stride.  Defaults to `CFG.ray_step`.
    ray_mode       : How to draw rays — "lines", "quiver", or "points".
                     Defaults to `CFG.ray_mode`.
                     **If "points", you MUST pass `points`.**
    points         : ndarray shaped (R, N, 3) where R = H×W pixels.
                     Only used when `ray_mode=="points"`.
    point_size     : Scatter marker size (px) for `points`.
    sample_grid    : Voxel‑sample occupancy and plot black points for ∞ density.
    grid_*         : Bounds / resolution / spacing for lattice grid.
    invert_x       : Flip X‑axis for left‑handed view (e.g. OpenGL convention).
    elev, azim     : Matplotlib 3‑D camera angles (deg).
    t_near, t_far  : Override near / far planes for *all* cameras' rays.
    """

    # ── cfg‑driven defaults ────────────────────────────────────────────────
    ray_step    = CFG.ray_step    if ray_step    is None else ray_step
    ray_mode    = CFG.ray_mode    if ray_mode    is None else ray_mode
    grid_bounds = CFG.grid_bounds if grid_bounds is None else grid_bounds
    grid_res    = CFG.grid_res    if grid_res    is None else grid_res
    grid_lines  = CFG.grid_lines  if grid_lines  is None else grid_lines
    grid_step   = CFG.grid_step   if grid_step   is None else grid_step
    figsize     = CFG.figsize     if figsize     is None else figsize
    elev        = CFG.axis_elev   if elev        is None else elev
    azim        = CFG.axis_azim   if azim        is None else azim
    invert_x    = ("x" in CFG.axis_invert) if invert_x is None else invert_x

    # ── enforce rule: points required for "points" mode ────────────────────
    if ray_mode == "points":
        if points is None:
            raise ValueError(
                "`ray_mode='points'` requires a `points` array shaped (R, N, 3)."
            )
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("`points` must have exactly three dims: (R, N, 3).")

    # ── create figure / axis ───────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=CFG.dpi)
    ax  = fig.add_subplot(111, projection="3d")

    # ── draw world shapes ──────────────────────────────────────────────────
    for s in getattr(world, "shapes", []):
        (plot_box if isinstance(s, Box) else plot_sphere)(ax, s)

    # ── optional voxel occupancy sample ────────────────────────────────────
    if sample_grid:
        xs = np.linspace(*grid_bounds[0], grid_res[0], dtype=float)
        ys = np.linspace(*grid_bounds[1], grid_res[1], dtype=float)
        zs = np.linspace(*grid_bounds[2], grid_res[2], dtype=float)
        grid_pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        occ_pts  = grid_pts[np.isinf([world.density(*p) for p in grid_pts])]
        if occ_pts.size:
            ax.scatter(*occ_pts.T, s=1, c="k", alpha=0.4, label="occ")

    # ── world axis triad ───────────────────────────────────────────────────
    axis_triad(ax, length=CFG.axis_triad_len)

    # ── cameras & rays (and external points) ───────────────────────────────
    if cameras:
        for idx, cam in enumerate(cameras):
            colour = f"C{idx % 10}"

            # ----- camera marker & axes -----------------------------------
            cam_pos = cam.H_wc[:3, 3]
            ax.scatter(*cam_pos, s=CFG.camera_marker_size,
                       c="red", marker="o", label=f"Cam {idx}")
            Camera._draw_pose_axes(ax, cam.H_wc, scale=0.4)

            if not draw_rays:
                continue

            # ----- resolve per‑camera near/far ----------------------------
            near = t_near if t_near is not None else cam.t_near
            far  = t_far  if t_far  is not None else cam.t_far

            # ----- external points branch ---------------------------------
            if ray_mode == "points":
                total_pixels = cam.intr.height * cam.intr.width
                if points.shape[0] != total_pixels:
                    raise ValueError(
                        f"`points` first dim ({points.shape[0]}) must equal "
                        f"H×W = {total_pixels} for camera {idx}."
                    )
                cloud = points[::ray_step].reshape(-1, 3)
                ax.scatter(*cloud.T, s=point_size, c=colour, depthshade=False)
                continue  # skip ray drawing for this cam

            # ----- compute / draw rays ------------------------------------
            O, D = cam.get_rays(frame="world", step=ray_step, normalize=True)

            if ray_mode == "quiver":
                ax.quiver(*O.T, *D.T, length=far,
                          normalize=True, color=colour, linewidth=0.6)

            elif ray_mode == "lines":
                segs = np.stack([O, O + far * D], axis=1)  # (N,2,3)
                ax.add_collection3d(Line3DCollection(segs, colors=colour, lw=0.6))

            else:  # should never happen due to Literal type
                raise ValueError("ray_mode must be 'lines', 'quiver', or 'points'")

    # ── lattice grid lines (optional) ──────────────────────────────────────
    if grid_lines:
        grid3d(ax, bounds=grid_bounds, step=grid_step)

    # ── axis styling & legend ──────────────────────────────────────────────
    style_3d_axis(ax,
                  invert=("x",) if invert_x else (),
                  elev=elev, azim=azim)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()
