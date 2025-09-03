# viz/viz_world.py (clean, deduplicated, optimized)
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


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _to_numpy3(x) -> np.ndarray:
    """Return (...,3) as float32 NumPy."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float32).numpy()
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

def _iter_cameras(cams: CameraLike) -> list[Camera]:
    """Normalize `cameras` to a list."""
    if cams is None:
        return []
    if isinstance(cams, Camera):
        return [cams]
    return list(cams)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _uniq_legend(ax) -> None:
    """Deduplicate legend entries, keep first occurrence."""
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="upper right", frameon=False)

def _viz_add_rays(ax, O_np: np.ndarray, D_np: np.ndarray, *,
                  mode: Literal["lines", "quiver"], color: str,
                  length: float, linewidth: float, alpha: float, zorder: int) -> None:
    """Add ray glyphs for one pose (NumPy arrays, shape (R,3))."""
    if mode == "quiver":
        ax.quiver(
            *O_np.T, *D_np.T,
            length=float(length), normalize=True,
            color=color, linewidth=linewidth, alpha=alpha, zorder=zorder,
        )
    elif mode == "lines":
        segs = np.stack([O_np, O_np + float(length) * D_np], axis=1)  # (R,2,3)
        coll = Line3DCollection(segs, colors=color, lw=linewidth)
        coll.set_alpha(alpha)
        coll.set_zorder(zorder)
        ax.add_collection3d(coll)
    else:
        raise ValueError("ray_mode must be 'lines' or 'quiver'")

def _viz_add_camera(ax, cam_idx, T: torch.Tensor, *, color: str, length: float, with_axes: bool) -> None:
    """Add a camera marker and optional axes for one pose."""
    cam_pos = _to_numpy3(T[:3, 3])
    ax.scatter(*cam_pos, s=VCFG.camera_marker_size, c=color, marker="o", label=f"Cam{cam_idx}", zorder=5)
    if with_axes:
        draw_pose_axes(ax, T, scale=float(length) * 0.2)


# ----------------------------------------------------------------------------- #
# Main API
# ----------------------------------------------------------------------------- #
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
    # Allowed shapes:
    #   single cam   -> (R, N, 3)
    #   batched cam  -> (B, R, N, 3)
    # If provided, drawn instead of sampling.
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

    # ── labels / title / display ───────────────────────────────────────────
    set_labels: bool = True,
    title: Optional[str] = None,               # if None or "", no title is set
    show: bool = True,                         # call plt.show() at the end
) -> tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot]:
    """
    Render a World + (batched) Cameras + rays/samples in WORLD frame.

    Accepts a single Camera (batched or not) or a list of Cameras.
    For batched cameras, choose a subset via `cam_indices` (int | list[int] | "all").

    Rays are drawn either by striding the pixel grid (`ray_step`) or by drawing
    exactly `rays_per_pose` rays per pose (unique random on the step-grid).
    If `external_points` is provided, those are plotted instead of sampling
    along rays. Otherwise, set `draw_samples=True` to sample via Camera.sample_along_rays().

    Parameters added:
      - title: Optional title for the axes (no-op if None/"").
      - show : Whether to call plt.show() before returning.

    Returns:
      (fig, ax): Matplotlib Figure and 3D Axes.
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
        if set_labels:
            _uniq_legend(ax)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    color_counter = 0
    for cam in cams:
        is_batched = getattr(cam, "_is_batched", False)
        sel = cam._select_cam_indices(cam_indices) if is_batched else [0]
        H_wc_sel = cam.H_wc[sel] if is_batched else cam.H_wc

        # Decide whether we need rays at all
        need_rays = draw_rays or draw_samples or (external_points is not None)

        # Build rays using the most appropriate API (avoid "build all then slice")
        O_all = D_all = None
        if need_rays:
            if rays_per_pose is not None:
                O_all, D_all = cam.get_rays_sampled(
                    rays_per_pose=int(rays_per_pose),
                    frame="world",
                    step=1 if rays_per_pose else max(1, ray_step),
                    normalize=True,
                    rng=samples_rng,
                )
            else:
                O_all, D_all = cam.get_rays(
                    frame="world",
                    step=max(1, ray_step),
                    normalize=True,
                )

            # Conform to (Bsel, R, 3) regardless of batched/single
            if is_batched:
                O_all = O_all[sel]          # (Bsel, R, 3)
                D_all = D_all[sel]          # (Bsel, R, 3)
            else:
                O_all = O_all[None, ...]    # (1, R, 3)
                D_all = D_all[None, ...]    # (1, R, 3)

        # Visual ray length from overrides or camera
        vis_far = float(cam.t_far if t_far is None else t_far)

        # Iterate selected poses
        for k in range(len(sel)):
            col = f"C{(color_counter % 10)}"; color_counter += 1

            # Camera marker + axes
            T = H_wc_sel[k] if is_batched else H_wc_sel
            _viz_add_camera(ax, k, T, color=col, length=vis_far, with_axes=True)

            if not need_rays:
                continue  # only cameras requested

            # Rays for this pose
            O = O_all[k]  # (R,3)
            D = D_all[k]  # (R,3)

            # External points take precedence over sampling
            if external_points is not None:
                P = torch.as_tensor(external_points)
                if is_batched:
                    # Expect (B, R, N, 3). We only index batch dimension at k.
                    if P.ndim != 4 or P.shape[-1] != 3:
                        raise ValueError("For batched cameras, external_points must have shape (B, R, N, 3).")
                    if k >= P.shape[0]:
                        raise ValueError(f"external_points batch too small: B={P.shape[0]} does not cover index {k}")
                    cloud = _to_numpy3(P[k].reshape(-1, 3))
                else:
                    # Expect (R, N, 3)
                    if P.ndim != 3 or P.shape[-1] != 3:
                        raise ValueError("For a single camera, external_points must have shape (R, N, 3).")
                    cloud = _to_numpy3(P.reshape(-1, 3))
                ax.scatter(
                    *cloud.T, s=external_point_size, c=col,
                    depthshade=False, alpha=samples_alpha, zorder=6
                )

            elif draw_samples:
                # Sample along rays via Camera's sampler (uses camera's n_points_per_ray)
                t_vals, deltas, pts = cam.sample_along_rays(
                    O, D, rng=samples_rng, deterministic=samples_deterministic
                )  # (R,N), (R,N), (R,N,3)
                P = _to_numpy3(pts.reshape(-1, 3))
                ax.scatter(
                    *P.T, s=external_point_size, c=col,
                    depthshade=False, alpha=samples_alpha, zorder=6
                )

            # Draw rays
            if draw_rays:
                _viz_add_rays(
                    ax,
                    _to_numpy3(O), _to_numpy3(D),
                    mode=ray_mode, color=col,
                    length=vis_far, linewidth=ray_linewidth,
                    alpha=ray_alpha, zorder=7,
                )

    # ── axis styling & legend / title / show ───────────────────────────────
    style_3d_axis(ax, invert=invert_axes, elev=elev, azim=azim)
    if set_labels:
        _uniq_legend(ax)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax

