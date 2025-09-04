# viz/viz_world.py (clean, optimized, well‑documented)
from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

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
    """Return (...,3) as **float32** NumPy for fast Matplotlib consumption."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float32).numpy()
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _iter_cameras(cams: CameraLike) -> list[Camera]:
    """Normalize `cameras` into a list for uniform iteration."""
    if cams is None:
        return []
    if isinstance(cams, Camera):
        return [cams]
    return list(cams)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _uniq_legend(ax) -> None:
    """Deduplicate legend entries (keep first occurrence)."""
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


def _as_long_tensor(
    x: Optional[Union[torch.Tensor, np.ndarray, Iterable[int], Iterable[Iterable[int]]]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert indices-like input to `torch.long` on `device`.

    Accepts `None`, 1‑D (K,), or 2‑D (B,K). Returns `None` if `x` is `None`.
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=torch.long)
    # generic iterables
    try:
        # Heuristically detect nested iterables → 2D
        first = next(iter(x))  # may raise StopIteration
        if isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
            return torch.as_tensor(list(x), device=device, dtype=torch.long)
        # 1D
        return torch.as_tensor(list(x), device=device, dtype=torch.long)
    except StopIteration:
        return torch.empty(0, dtype=torch.long, device=device)


def _viz_add_rays(
    ax: Axes3D,
    O_np: np.ndarray,
    D_np: np.ndarray,
    *,
    mode: Literal["lines", "quiver"],
    color: str,
    t0: float,
    length: float,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Add ray glyphs for one pose (NumPy arrays, shape (R,3)).

    Parameters
    ----------
    O_np, D_np : (R,3)
        Ray origins and (unit) directions in world frame.
    t0, length : float
        Visual segment is drawn from **O + t0·D** to **O + (t0+length)·D**.
    mode : {"lines", "quiver"}
        Rendering style; "lines" supports (t0,length) exactly.
    """
    if mode == "quiver":
        # Quiver ignores t0; we visualize at the origin with a given length.
        ax.quiver(
            *O_np.T,
            *D_np.T,
            length=float(length),
            normalize=True,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    elif mode == "lines":
        P0 = O_np + float(t0) * D_np
        P1 = O_np + float(t0 + length) * D_np
        segs = np.stack([P0, P1], axis=1)  # (R,2,3)
        coll = Line3DCollection(segs, colors=color, lw=linewidth)
        coll.set_alpha(alpha)
        coll.set_zorder(zorder)
        ax.add_collection3d(coll)
    else:
        raise ValueError("ray_mode must be 'lines' or 'quiver'")


def _viz_add_camera(
    ax: Axes3D,
    cam_idx: int,
    T: torch.Tensor,
    *,
    color: str,
    length: float,
    with_axes: bool,
) -> None:
    """Add a camera marker and optional axes for one pose."""
    cam_pos = _to_numpy3(T[:3, 3])
    ax.scatter(
        *cam_pos,
        s=VCFG.camera_marker_size,
        c=color,
        marker="o",
        label=f"Cam{cam_idx}",
        zorder=5,
    )
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
    shape_face_alpha: float | None = None,  # boxes/spheres fill
    shape_edge_alpha: float | None = None,  # box wireframe edges
    ray_alpha: float | None = None,  # rays (lines/quivers)
    ray_linewidth: float | None = None,  # lines thickness
    samples_alpha: float | None = None,  # sampled/external points

    # ── rays / samples ─────────────────────────────────────────────────────
    draw_rays: bool = False,
    ray_mode: Literal["lines", "quiver"] = "lines",
    ray_step: int | None = None,  # None -> VCFG.ray_step
    rays_per_pose: Optional[int] = None,  # overrides step if set
    # NEW: explicit selection over the step‑grid
    rays_indices: Optional[Union[torch.Tensor, np.ndarray, Iterable[int], Iterable[Iterable[int]]]] = None,

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
    title: Optional[str] = None,  # if None or "", no title is set
    show: bool = True,  # call plt.show() at the end
) -> Tuple[Figure, Axes3D]:
    """Render World + Cameras + rays/samples in **world** frame.

    • Supports **single** or **batched** `Camera`. Use `cam_indices` to select
      a subset of poses from a batched camera (int | list[int] | "all").
    • Rays can be drawn by step-grid (`ray_step`), by uniform sub-sampling
      (`rays_per_pose`), or **explicitly** via `rays_indices` which has higher
      priority (mirrors `Camera.get_rays_sampled`).
    • If `external_points` is provided, they are plotted instead of sampling
      along rays. Otherwise, pass `draw_samples=True` to call
      `Camera.sample_along_rays()` using the camera’s `n_points_per_ray`.

    `rays_indices` semantics
    ------------------------
    Unbatched camera  : (K,) step-grid indices.  
    Batched camera    : (B,K) per-pose indices **or** (K,) to broadcast to all
                        selected poses. If you pass (Bsel,K) alongside
                        `cam_indices`, this function expands to (B,K) internally.

    Returns
    -------
    (fig, ax) : Matplotlib `Figure` and 3D axes (`Axes3D`).
    """
    # ── resolve config defaults ────────────────────────────────────────────
    VC = VCFG  # alias
    figsize = VC.figsize if figsize is None else figsize
    elev = VC.axis_elev if elev is None else elev
    azim = VC.axis_azim if azim is None else azim
    grid_lines = VC.grid_lines if grid_lines is None else grid_lines
    grid_bounds = VC.grid_bounds if grid_bounds is None else grid_bounds
    grid_step = VC.grid_step if grid_step is None else grid_step
    invert_axes = VC.axis_invert if invert_axes is None else invert_axes
    ray_step = VC.ray_step if ray_step is None else max(1, int(ray_step))

    shape_face_alpha = _clamp01(VC.shape_face_alpha if shape_face_alpha is None else float(shape_face_alpha))
    shape_edge_alpha = _clamp01(VC.shape_edge_alpha if shape_edge_alpha is None else float(shape_edge_alpha))
    ray_alpha = _clamp01(VC.ray_alpha if ray_alpha is None else float(ray_alpha))
    ray_linewidth = VC.ray_linewidth if ray_linewidth is None else float(ray_linewidth)
    samples_alpha = _clamp01(VC.samples_alpha if samples_alpha is None else float(samples_alpha))

    # ── figure / axis ─────────────────────────────────────────────────────
    fig: Figure = plt.figure(figsize=figsize, dpi=VC.dpi)
    ax: Axes3D = fig.add_subplot(111, projection="3d")

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
            # Prepare indices if provided
            idx_user = _as_long_tensor(rays_indices, device=cam.H_wc.device)
            idx_for_camera = None  # what we'll pass into Camera.get_rays_sampled(...)

            if idx_user is not None:
                if is_batched:
                    if idx_user.ndim == 1:
                        # (K,) → expand to (B,K) for camera, then later slice to sel
                        K = idx_user.shape[0]
                        idx_for_camera = idx_user.view(1, K).expand(cam.B, K).contiguous()
                    elif idx_user.ndim == 2:
                        if idx_user.shape[0] == cam.B:
                            # Already (B,K): pass-through
                            idx_for_camera = idx_user
                        elif idx_user.shape[0] == len(sel):
                            # (Bsel,K): lift to (B,K) placing rows at selected indices
                            Bsel, K = idx_user.shape
                            # Use row 0 as a safe default for non-selected poses
                            default_row = idx_user[0]
                            idx_for_camera = default_row.view(1, K).expand(cam.B, K).clone()
                            for row_pos, b_idx in enumerate(sel):
                                idx_for_camera[b_idx] = idx_user[row_pos]
                        else:
                            raise ValueError(
                                f"rays_indices rows ({idx_user.shape[0]}) must equal camera.B ({cam.B}) or len(sel) ({len(sel)})."
                            )
                    else:
                        raise ValueError("For batched cameras, rays_indices must be (K,) or (B,K)")
                else:
                    if idx_user.ndim != 1:
                        raise ValueError("For a single camera, rays_indices must be 1D (K,)")
                    idx_for_camera = idx_user  # (K,)

            # Camera API call hierarchy: indices > rays_per_pose > full grid
            if idx_for_camera is not None:
                O_all, D_all = cam.get_rays_sampled(
                    rays_per_pose=None,  # ignored when indices is provided
                    frame="world",
                    step=ray_step,
                    normalize=True,
                    rng=samples_rng,
                    indices=idx_for_camera,  # (B,K) for batched, (K,) for single
                )
            elif rays_per_pose is not None:
                O_all, D_all = cam.get_rays_sampled(
                    rays_per_pose=int(rays_per_pose),
                    frame="world",
                    step=ray_step,
                    normalize=True,
                    rng=samples_rng,
                    indices=None,
                )
            else:
                O_all, D_all = cam.get_rays(
                    frame="world",
                    step=ray_step,
                    normalize=True,
                )

            # Conform to (Bsel, R, 3) regardless of batched/single
            if is_batched:
                O_all = O_all[sel]  # (Bsel, R, 3)
                D_all = D_all[sel]  # (Bsel, R, 3)
            else:
                O_all = O_all[None, ...]  # (1, R, 3)
                D_all = D_all[None, ...]  # (1, R, 3)

        # Visual segment settings from near/far or camera defaults
        t0 = float(0.0 if t_near is None else t_near)
        vis_len = float((cam.t_far if t_far is None else t_far) - t0)
        vis_len = max(0.0, vis_len)

        # Iterate selected poses
        for k in range(len(sel)):
            col = f"C{(color_counter % 10)}"; color_counter += 1

            # Camera marker + axes
            T = H_wc_sel[k] if is_batched else H_wc_sel
            _viz_add_camera(ax, k, T, color=col, length=(t0 + vis_len if vis_len > 0 else cam.t_far), with_axes=True)

            if not need_rays:
                continue  # only cameras requested

            # Rays for this pose
            O = O_all[k]  # (R,3)
            D = D_all[k]  # (R,3)

            # External points take precedence over sampling
            if external_points is not None:
                P = torch.as_tensor(external_points)
                if is_batched:
                    # Expect (B, R, N, 3) or (Bsel, R, N, 3). Index batch dim at k.
                    if P.ndim != 4 or P.shape[-1] != 3:
                        raise ValueError("For batched cameras, external_points must have shape (B, R, N, 3).")
                    if P.shape[0] == len(sel):
                        cloud = _to_numpy3(P[k].reshape(-1, 3))
                    elif P.shape[0] == getattr(cam, 'B', len(sel)):
                        cloud = _to_numpy3(P[sel[k]].reshape(-1, 3))
                    else:
                        raise ValueError(
                            f"external_points batch mismatch: got B={P.shape[0]}, expected {len(sel)} or {getattr(cam,'B', len(sel))}"
                        )
                else:
                    # Expect (R, N, 3)
                    if P.ndim != 3 or P.shape[-1] != 3:
                        raise ValueError("For a single camera, external_points must have shape (R, N, 3).")
                    cloud = _to_numpy3(P.reshape(-1, 3))
                ax.scatter(
                    *cloud.T,
                    s=external_point_size,
                    c=col,
                    depthshade=False,
                    alpha=samples_alpha,
                    zorder=6,
                )

            elif draw_samples:
                # Sample along rays via Camera's sampler (uses camera's n_points_per_ray)
                t_vals, deltas, pts = cam.sample_along_rays(
                    O, D, rng=samples_rng, deterministic=samples_deterministic
                )  # (R,N), (R,N), (R,N,3)
                P = _to_numpy3(pts.reshape(-1, 3))
                ax.scatter(
                    *P.T,
                    s=external_point_size,
                    c=col,
                    depthshade=False,
                    alpha=samples_alpha,
                    zorder=6,
                )

            # Draw rays
            if draw_rays:
                _viz_add_rays(
                    ax,
                    _to_numpy3(O),
                    _to_numpy3(D),
                    mode=ray_mode,
                    color=col,
                    t0=t0,
                    length=vis_len if vis_len > 0 else (cam.t_far if t_far is None else float(t_far)),
                    linewidth=ray_linewidth,
                    alpha=ray_alpha,
                    zorder=7,
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
