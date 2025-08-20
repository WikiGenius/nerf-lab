# viz/world.py
from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import torch

from ..world.geometry import Box, Sphere
from ..camera.camera import Camera
from ..viz.primitives import plot_box, plot_sphere
from ..viz.axis import style_3d_axis, axis_triad, grid3d
from ..viz.pose import draw_pose_axes
from ..config.viz_config import viz_cfg as VCFG

NDArray = np.ndarray
CameraLike = Union[Camera, Iterable[Camera]]
CamIdxSel = Optional[Union[int, Iterable[int], Literal["all"]]]


def _to_numpy3(x) -> np.ndarray:
    """Return (...,3) as float32 numpy."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float32).numpy()
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _iter_cameras(cams: CameraLike) -> list[Camera]:
    """Normalize cameras input to a flat Python list of Camera objects."""
    if cams is None:
        return []
    if isinstance(cams, Camera):
        return [cams]
    return list(cams)


def plot_world(
    world,
    *,
    cameras: CameraLike = None,
    cam_indices: CamIdxSel = None,  # applies to any batched camera
    # ── rays / samples ─────────────────────────────────────────────────────
    draw_rays: bool = False,
    ray_mode: Literal["lines", "quiver"] = "lines",
    ray_step: int = 1,                      # pixel stride when not using rays_per_pose
    rays_per_pose: Optional[int] = None,    # overrides step if set
    draw_samples: bool = False,
    samples_rng: Optional[torch.Generator] = None,
    samples_deterministic: Optional[bool] = None,
    # ── external points (per-camera or shared) ─────────────────────────────
    external_points: Optional[Union[NDArray, torch.Tensor]] = None,
    # shapes allowed: (R,N,3) or (B,R,N,3); if provided, drawn instead of sampling
    external_point_size: float = 4.0,
    # ── lattice & view ─────────────────────────────────────────────────────
    grid_lines: Optional[bool] = None,
    grid_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    grid_step: Optional[Tuple[float, float, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    invert_axes: Optional[tuple[str, ...]] = None,  # e.g., ("x",) to flip X
    # ── optional override of near/far for visual length of rays ────────────
    t_near: Optional[float] = None,
    t_far: Optional[float] = None,
) -> None:
    """
    Render a World + (batched) Cameras + rays/samples in WORLD frame.

    Accepts a single Camera (batched or not) or a list of Cameras.
    For batched cameras, choose subset via `cam_indices` (int | list[int] | "all").

    Rays drawn either by subsampling the pixel grid (`ray_step`) or by drawing
    exactly `rays_per_pose` rays per pose (uniform random, per pose).
    If `external_points` is provided, those are plotted instead of sampling
    along rays. Otherwise, set `draw_samples=True` to sample via Camera.sample_along_rays().
    """
    # ── config defaults ────────────────────────────────────────────────────
    figsize   = VCFG.figsize    if figsize   is None else figsize
    elev      = VCFG.axis_elev  if elev      is None else elev
    azim      = VCFG.axis_azim  if azim      is None else azim
    grid_lines = VCFG.grid_lines if grid_lines is None else grid_lines
    grid_bounds = VCFG.grid_bounds if grid_bounds is None else grid_bounds
    grid_step   = VCFG.grid_step   if grid_step   is None else grid_step
    invert_axes = VCFG.axis_invert if invert_axes is None else invert_axes

    # ── figure / axis ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=VCFG.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # ── draw world shapes ─────────────────────────────────────────────────
    for s in getattr(world, "shapes", []):
        if isinstance(s, Box):
            plot_box(ax, s)
        elif isinstance(s, Sphere):
            plot_sphere(ax, s)

    # ── world axis triad ──────────────────────────────────────────────────
    axis_triad(ax, length=VCFG.axis_triad_len)

    # ── cameras & overlays ────────────────────────────────────────────────
    cams = _iter_cameras(cameras)
    color_counter = 0

    for cam in cams:
        # select subset of poses for batched cameras
        if cam._is_batched:
            sel = cam._select_cam_indices(cam_indices)
            H_wc = cam.H_wc[sel]             # (Bsel,4,4)
            O_all, D_all = cam.get_rays(
                frame="world",
                step=1 if rays_per_pose else max(1, ray_step),
                normalize=True,
            )                                 # (B, R, 3)

            # subselect the same indices from rays if needed
            O_all = O_all[sel]
            D_all = D_all[sel]
        else:
            sel = [0]
            H_wc = cam.H_wc                   # (4,4)
            O_all, D_all = cam.get_rays(
                frame="world",
                step=1 if rays_per_pose else max(1, ray_step),
                normalize=True,
            )                                 # (R, 3)

            # make them (B=1, R, 3) for uniform handling below
            O_all = O_all[None, ...]
            D_all = D_all[None, ...]

        # iterate selected poses
        for k in range(len(sel)):
            col = f"C{(color_counter % 10)}"; color_counter += 1

            # camera marker + axes
            T = H_wc[k] if cam._is_batched else H_wc
            cam_pos = _to_numpy3(T[:3, 3])
            ax.scatter(*cam_pos, s=VCFG.camera_marker_size, c=col, marker="o", label=f"Cam {color_counter-1}")
            draw_pose_axes(ax, T, scale=float((t_far if t_far is not None else cam.t_far) * 0.2))

            # near/far for visualizing ray length
            near = cam.t_near if t_near is None else t_near
            far  = cam.t_far  if t_far  is None else t_far

            # pick rays for this pose
            O = O_all[k]  # (R,3)
            D = D_all[k]  # (R,3)
            if rays_per_pose is not None:
                # sample exactly K rays
                R = O.shape[0]
                K = int(rays_per_pose)
                if not (1 <= K <= R):
                    raise ValueError(f"rays_per_pose must be in [1,{R}] (got {K})")
                # uniform random without replacement; device = CPU np for plotting
                idx = torch.randperm(R, device=O.device)[:K]
                O = O.index_select(0, idx)
                D = D.index_select(0, idx)
            else:
                # stride already applied via step in get_rays
                pass

            # external points take precedence over sampling
            if external_points is not None:
                P = torch.as_tensor(external_points)
                if cam._is_batched:
                    # For batched cameras we require (B,R,N,3)
                    if P.ndim != 4 or P.shape[-1] != 3:
                        raise ValueError(
                            "For batched cameras, external_points must have shape (B, R, N, 3)."
                        )
                    if k >= P.shape[0]:
                        raise ValueError(
                            f"external_points batch too small: B={P.shape[0]} does not cover index {k}"
                        )
                    cloud = _to_numpy3(P[k].reshape(-1, 3))
                else:
                    # For single camera we require (R,N,3)
                    if P.ndim != 3 or P.shape[-1] != 3:
                        raise ValueError(
                            "For a single camera, external_points must have shape (R, N, 3)."
                        )
                    cloud = _to_numpy3(P.reshape(-1, 3))
                ax.scatter(*cloud.T, s=external_point_size, c=col, depthshade=False, alpha=0.9)
            elif draw_samples:
                # sample along rays via Camera's sampler
                t_vals, deltas, pts = cam.sample_along_rays(
                    O, D, rng=samples_rng, deterministic=samples_deterministic
                )  # (R,N), (R,N), (R,N,3)
                P = _to_numpy3(pts.reshape(-1, 3))
                ax.scatter(*P.T, s=external_point_size, c=col, depthshade=False, alpha=0.9)

            # draw rays if requested
            if draw_rays:
                O_np = _to_numpy3(O)
                D_np = _to_numpy3(D)
                if ray_mode == "quiver":
                    ax.quiver(*O_np.T, *D_np.T, length=float(far), normalize=True, color=col, linewidth=0.6)
                elif ray_mode == "lines":
                    segs = np.stack([O_np, O_np + float(far) * D_np], axis=1)  # (N,2,3)
                    ax.add_collection3d(Line3DCollection(segs, colors=col, lw=0.6))
                else:
                    raise ValueError("ray_mode must be 'lines' or 'quiver'")

    # ── optional lattice grid ──────────────────────────────────────────────
    if grid_lines:
        grid3d(ax, bounds=grid_bounds, step=grid_step)

    # ── axis styling & legend ──────────────────────────────────────────────
    style_3d_axis(ax, invert=invert_axes, elev=elev, azim=azim)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()
