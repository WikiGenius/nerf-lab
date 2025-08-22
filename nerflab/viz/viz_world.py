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


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

@torch.no_grad()
def plot_world(
    world,
    *,
    cameras: CameraLike = None,
    cam_indices: CamIdxSel = None,  # applies to any batched camera

    # ── opacity / style controls (None -> use VCFG) ─────────────────────────
    shape_face_alpha: float | None = None,     # boxes/spheres fill
    shape_edge_alpha: float | None = None,     # box wireframe edges
    ray_alpha: float | None = None,            # rays (lines/quivers)
    ray_linewidth: float | None = None,        # lines thickness
    samples_alpha: float | None = None,        # sampled/external points

    # ── rays / samples ─────────────────────────────────────────────────────
    draw_rays: bool = False,
    ray_mode: Literal["lines", "quiver"] = "lines",
    ray_step: int | None = None,               # None -> VCFG.ray_step
    rays_per_pose: Optional[int] = None,       # overrides step if set

    draw_samples: bool = False,
    samples_rng: Optional[torch.Generator] = None,
    samples_deterministic: Optional[bool] = None,

    # ── external points (per-camera or shared) ─────────────────────────────
    # Allowed shapes: (R, N, 3) or (B, R, N, 3). If provided, drawn instead of sampling.
    external_points: Optional[Union[NDArray, torch.Tensor]] = None,
    external_point_size: float = 4.0,

    # ── lattice & view ─────────────────────────────────────────────────────
    grid_lines: Optional[bool] = None,
    grid_bounds: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None,
    grid_step: Optional[Tuple[float, float, float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    invert_axes: Optional[tuple[str, ...]] = None,  # e.g., ("x",) to flip X

    # ── optional override of near/far for visual length of rays ────────────
    t_near: Optional[float] = None,
    t_far: Optional[float] = None,
    set_labels = True
) -> None:
    """
    Render a World + (batched) Cameras + rays/samples in WORLD frame.

    Accepts a single Camera (batched or not) or a list of Cameras.
    For batched cameras, choose a subset via `cam_indices` (int | list[int] | "all").

    Rays are drawn either by subsampling the pixel grid (`ray_step`) or by drawing
    exactly `rays_per_pose` rays per pose (uniform random, per pose).
    If `external_points` is provided, those are plotted instead of sampling
    along rays. Otherwise, set `draw_samples=True` to sample via Camera.sample_along_rays().
    """
    # ── resolve config defaults ────────────────────────────────────────────
    VC = VCFG  # alias
    figsize     = VC.figsize     if figsize     is None else figsize
    elev        = VC.axis_elev   if elev        is None else elev
    azim        = VC.axis_azim   if azim        is None else azim
    grid_lines  = VC.grid_lines  if grid_lines  is None else grid_lines
    grid_bounds = VC.grid_bounds if grid_bounds is None else grid_bounds
    grid_step   = VC.grid_step   if grid_step   is None else grid_step
    invert_axes = VC.axis_invert if invert_axes is None else invert_axes
    ray_step    = VC.ray_step    if ray_step    is None else max(1, int(ray_step))

    shape_face_alpha = _clamp01(VC.shape_face_alpha if shape_face_alpha is None else float(shape_face_alpha))
    shape_edge_alpha = _clamp01(VC.shape_edge_alpha if shape_edge_alpha is None else float(shape_edge_alpha))
    ray_alpha        = _clamp01(VC.ray_alpha        if ray_alpha        is None else float(ray_alpha))
    ray_linewidth    =          VC.ray_linewidth    if ray_linewidth    is None else float(ray_linewidth)
    samples_alpha    = _clamp01(VC.samples_alpha    if samples_alpha    is None else float(samples_alpha))

    # ── figure / axis ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=VC.dpi)
    ax = fig.add_subplot(111, projection="3d")

    # ── draw world shapes (low z-order) ────────────────────────────────────
    for s in getattr(world, "shapes", []):
        if isinstance(s, Box):
            plot_box(ax, s, face_alpha=shape_face_alpha, edge_alpha=shape_edge_alpha, zorder=1)
        elif isinstance(s, Sphere):
            plot_sphere(ax, s, face_alpha=shape_face_alpha, zorder=1)

    # ── optional lattice grid ──────────────────────────────────────────────
    if grid_lines:
        grid3d(ax, bounds=grid_bounds, step=grid_step, zorder=2)

    # ── world axis triad ──────────────────────────────────────────────────
    axis_triad(ax, length=VC.axis_triad_len)  # helper sets sensible z-order (~3)

    # ── cameras & overlays ────────────────────────────────────────────────
    cams = _iter_cameras(cameras)
    if not cams:
        # Style axis anyway for consistent appearance
        style_3d_axis(ax, invert=invert_axes, elev=elev, azim=azim)
        plt.tight_layout(); plt.show()
        return

    color_counter = 0

    for cam in cams:
        is_batched = getattr(cam, "_is_batched", False)
        # Only compute rays if needed (rays, samples, or external_points)
        need_rays = draw_rays or draw_samples or (external_points is not None)

        if is_batched:
            sel = cam._select_cam_indices(cam_indices)
            H_wc = cam.H_wc[sel]  # (Bsel,4,4)

            if need_rays:
                O_all, D_all = cam.get_rays(
                    frame="world",
                    step=1 if rays_per_pose else max(1, ray_step),
                    normalize=True,
                )  # (B, R, 3)
                O_all = O_all[sel]
                D_all = D_all[sel]
        else:
            sel = [0]
            H_wc = cam.H_wc  # (4,4)
            if need_rays:
                O_all, D_all = cam.get_rays(
                    frame="world",
                    step=1 if rays_per_pose else max(1, ray_step),
                    normalize=True,
                )  # (R, 3)
                O_all = O_all[None, ...]
                D_all = D_all[None, ...]

        # Iterate selected poses
        for k in range(len(sel)):
            col = f"C{(color_counter % 10)}"; color_counter += 1

            # Camera marker + axes
            T = H_wc[k] if is_batched else H_wc
            cam_pos = _to_numpy3(T[:3, 3])
            ax.scatter(*cam_pos, s=VC.camera_marker_size, c=col, marker="o",
                       label=f"Cam {color_counter-1}", zorder=5)
            draw_pose_axes(ax, T, scale=float((t_far if t_far is not None else cam.t_far) * 0.2))

            # Visual length of rays
            far = cam.t_far if t_far is None else t_far

            # Short-circuit if no rays/samples/external points requested
            if not need_rays:
                continue

            # Pick rays for this pose
            O = O_all[k]  # (R,3)
            D = D_all[k]  # (R,3)

            if rays_per_pose is not None:
                R = int(O.shape[0])
                K = int(rays_per_pose)
                if not (1 <= K <= R):
                    raise ValueError(f"rays_per_pose must be in [1,{R}] (got {K})")
                idx = torch.randperm(R, device=O.device)[:K]
                O = O.index_select(0, idx)
                D = D.index_select(0, idx)
            # else: stride already applied in get_rays

            # External points take precedence over sampling
            if external_points is not None:
                P = torch.as_tensor(external_points)
                if is_batched:
                    # Expect (B,R,N,3)
                    if P.ndim != 4 or P.shape[-1] != 3:
                        raise ValueError("For batched cameras, external_points must have shape (B, R, N, 3).")
                    if k >= P.shape[0]:
                        raise ValueError(f"external_points batch too small: B={P.shape[0]} does not cover index {k}")
                    cloud = _to_numpy3(P[k].reshape(-1, 3))
                else:
                    # Expect (R,N,3)
                    if P.ndim != 3 or P.shape[-1] != 3:
                        raise ValueError("For a single camera, external_points must have shape (R, N, 3).")
                    cloud = _to_numpy3(P.reshape(-1, 3))
                ax.scatter(*cloud.T, s=external_point_size, c=col,
                           depthshade=False, alpha=samples_alpha, zorder=6)

            elif draw_samples:
                # Sample along rays via Camera's sampler
                t_vals, deltas, pts = cam.sample_along_rays(
                    O, D, rng=samples_rng, deterministic=samples_deterministic
                )  # (R,N), (R,N), (R,N,3)
                P = _to_numpy3(pts.reshape(-1, 3))
                ax.scatter(*P.T, s=external_point_size, c=col,
                           depthshade=False, alpha=samples_alpha, zorder=6)

            # Draw rays last (highest z-order)
            if draw_rays:
                O_np = _to_numpy3(O); D_np = _to_numpy3(D)
                if ray_mode == "quiver":
                    ax.quiver(*O_np.T, *D_np.T, length=float(far), normalize=True,
                              color=col, linewidth=ray_linewidth, alpha=ray_alpha, zorder=7)
                elif ray_mode == "lines":
                    segs = np.stack([O_np, O_np + float(far) * D_np], axis=1)  # (N,2,3)
                    coll = Line3DCollection(segs, colors=col, lw=ray_linewidth)
                    coll.set_alpha(ray_alpha)
                    coll.set_zorder(7)
                    ax.add_collection3d(coll)
                else:
                    raise ValueError("ray_mode must be 'lines' or 'quiver'")

    # ── axis styling & legend ──────────────────────────────────────────────
    style_3d_axis(ax, invert=invert_axes, elev=elev, azim=azim)

    # De-duplicate legend labels (one per camera color)
    handles, labels = ax.get_legend_handles_labels()
    if set_labels and labels:
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l); uniq_h.append(h); uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()
