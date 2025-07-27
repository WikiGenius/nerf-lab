from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Iterable, Literal, Tuple

from nerflab.geometry import Box, Sphere
from nerflab import Camera
from .primitives import plot_box, plot_sphere
from .axis import style_3d_axis, axis_triad, grid3d
from .viz_config import viz_cfg as VCFG 


def plot_world(
    world,
    *,
    cameras: Iterable[Camera] | None = None,
    draw_rays: bool = False,
    ray_step: int | None = None,
    ray_mode: Literal["lines", "quiver", "points"] | None = None,
    sample_grid: bool = False,
    grid_bounds: Tuple[Tuple[float, float], ...] | None = None,
    grid_res: Tuple[int, int, int] | None = None,
    grid_lines: bool | None = None,
    grid_step: Tuple[float, float, float] | None = None,
    invert_x: bool | None = None,
    figsize: Tuple[int, int] | None = None,
    elev: float | None = None,
    azim: float | None = None,
    t_near: float | None = None,
    t_far: float | None = None,
    
) -> None:
    """
    Visualise a `World` with optional `Camera` objects.
    All `None` parameters fall back to `viz.cfg` values.
    """

    # ---------- resolve defaults from cfg ------------------------------------
    ray_step = VCFG.ray_step if ray_step is None else ray_step
    ray_mode = VCFG.ray_mode if ray_mode is None else ray_mode
    grid_bounds = VCFG.grid_bounds if grid_bounds is None else grid_bounds
    grid_res = VCFG.grid_res if grid_res is None else grid_res
    grid_lines = VCFG.grid_lines if grid_lines is None else grid_lines
    grid_step = VCFG.grid_step if grid_step is None else grid_step
    figsize = VCFG.figsize if figsize is None else figsize
    elev = VCFG.axis_elev if elev is None else elev
    azim = VCFG.axis_azim if azim is None else azim
    invert_x = ("x" in VCFG.axis_invert) if invert_x is None else invert_x

    fig = plt.figure(figsize=figsize, dpi=VCFG.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # ---------- shapes -------------------------------------------------------
    for shp in getattr(world, "shapes", []):
        (plot_box if isinstance(shp, Box) else plot_sphere)(ax, shp)

    # ---------- occupancy sampling ------------------------------------------
    if sample_grid:
        xs = np.linspace(*grid_bounds[0], grid_res[0], dtype=float)
        ys = np.linspace(*grid_bounds[1], grid_res[1], dtype=float)
        zs = np.linspace(*grid_bounds[2], grid_res[2], dtype=float)
        pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        occ = pts[np.isinf([world.density(*p) for p in pts])]
        if occ.size:
            ax.scatter(*occ.T, s=1, c="k", alpha=0.4, label="occ")

    # ---------- world triad --------------------------------------------------
    axis_triad(ax, length=VCFG.axis_triad_len)

    # ---------- cameras & rays ----------------------------------------------
    if cameras:
        for i, cam in enumerate(cameras):
            t_near = cam.t_near if t_near is None else t_near
            t_far = cam.t_far if t_far is None else t_far
            
            cam_pos = cam.H_wc[:3, 3]
            ray_len = t_far
            colour = f"C{i % 10}"

            ax.scatter(
                *cam_pos,
                s=VCFG.camera_marker_size,
                c="red",
                marker="o",
                label=f"Cam {i}",
            )
            Camera._draw_pose_axes(ax, cam.H_wc, scale=0.4)

            if not draw_rays:
                continue

            O, D = cam.get_rays(frame="world", step=ray_step, normalize=True)

            if ray_mode == "quiver":
                ax.quiver(
                    *O.T,
                    *D.T,
                    length=ray_len,
                    normalize=True,
                    color=colour,
                    linewidth=0.6,
                )
            elif ray_mode == "lines":
                segs = np.stack([O, O + ray_len * D], axis=1)
                ax.add_collection3d(Line3DCollection(segs, colors=colour, lw=0.6))
            elif ray_mode == "points":
                ts = np.linspace(
                    t_near, t_far, getattr(cam, "n_points_per_ray", 20)
                )
                P = (O[:, None, :] + ts[None, :, None] * D[:, None, :]).reshape(-1, 3)
                ax.scatter(*P.T, s=2, c=colour, depthshade=False)
            else:
                raise ValueError("ray_mode must be 'lines', 'quiver', or 'points'")

    # ---------- gridlines ----------------------------------------------------
    if grid_lines:
        grid3d(ax, bounds=grid_bounds, step=grid_step)

    # ---------- unified axis style ------------------------------------------
    style_3d_axis(
        ax,
        invert=("x",) if invert_x else (),
        elev=elev,
        azim=azim,
    )

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
