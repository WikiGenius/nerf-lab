# nerflab/viz/viz_world_interactive.py
# =============================================================================
# Interactive 3-D World Viewer (Plotly)
# =============================================================================
# This module provides a drop-in interactive counterpart to your Matplotlib
# `plot_world(...)`. It renders:
#   • World geometry: axis-aligned Box(center,size) and Sphere(center,radius)
#   • Camera poses (supports single or batched Camera)
#   • Rays and (optionally) sampled points along rays
#   • Axes triads for the world and for each camera (so you can “read” frames)
#
# Design goals
# ------------
# - API parity with your Matplotlib version (draw_rays, draw_samples,
#   rays_indices, cam_indices, t_near/t_far, etc.)
# - Zero surprises with your geometry classes (Box has `size`, not `dims`)
# - Efficient: only builds the rays you ask for, caches imports naturally
#
# Usage
# -----
# from nerflab.viz import plot_world_interactive
# fig, _ = plot_world_interactive(world, cameras=cam, draw_rays=True, save_html="view.html")
#
# Requirements
# ------------
#   pip install plotly
#
# Optional (for PNG export):
#   pip install -U kaleido
# =============================================================================

from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import plotly.graph_objects as go

from ..world.geometry import Box, Sphere, World
from ..camera.camera import Camera
from ..config.viz_config import viz_cfg as VCFG

NDArray = np.ndarray
CameraLike = Union[Camera, Iterable[Camera]]
CamIdxSel = Optional[Union[int, Iterable[int], Literal["all"]]]


# =============================================================================
# Helpers: type/array adapters, glyph builders
# =============================================================================
def _to_numpy3(x) -> np.ndarray:
    """Convert array-like or tensor to **float32** NumPy with last dim = 3."""
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif torch.is_tensor(x):
        arr = x.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {arr.shape}")
    return arr


def _iter_cameras(cams: CameraLike) -> list[Camera]:
    """Normalize CameraLike to a list for uniform iteration."""
    if cams is None:
        return []
    if isinstance(cams, Camera):
        return [cams]
    return list(cams)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _as_long_tensor(
    x: Optional[Union[torch.Tensor, np.ndarray, Iterable[int], Iterable[Iterable[int]]]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Accept None, (K,), or (B,K); return torch.long on `device`."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=torch.long)
    it = iter(x)
    try:
        first = next(it)
    except StopIteration:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
        return torch.as_tensor([first, *it], device=device, dtype=torch.long)
    return torch.as_tensor([first, *it], device=device, dtype=torch.long)


# ---- Camera frustum & axes glyphs -------------------------------------------
def _camera_frustum_lines(C: np.ndarray, R: np.ndarray, *, fov_deg: float = 60.0, scale: float = 0.2):
    """
    Build line segments for a simple camera frustum.

    Parameters
    ----------
    C : (3,) camera center in world
    R : (3,3) rotation (columns are camera basis axes in world coords)
    fov_deg : horizontal field-of-view used to spread the frustum corners
    scale   : frustum depth (in world units)
    """
    f = float(scale)
    half = np.tan(np.deg2rad(fov_deg / 2.0)) * f
    corners_cam = np.array(
        [[ half,  half, f],
         [-half,  half, f],
         [-half, -half, f],
         [ half, -half, f]], dtype=np.float32
    )
    corners_w = (R @ corners_cam.T).T + C[None, :]
    lines = [np.stack([C, q], axis=0) for q in corners_w]  # C→corner lines
    loop = [0, 1, 2, 3, 0]                                 # quad ring
    for i in range(4):
        a = corners_w[loop[i]]
        b = corners_w[loop[i + 1]]
        lines.append(np.stack([a, b], axis=0))
    return lines  # list of (2,3)


def _add_lines3d(fig: go.Figure, lines: Sequence[np.ndarray], name: str, *, width: float = 2.0, opacity: float = 1.0, color: Optional[str] = None):
    """Append line segments to a Plotly figure."""
    for seg in lines:
        seg = _to_numpy3(seg)
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines",
            line=dict(width=float(width), color=color),
            name=name, showlegend=False, opacity=float(opacity),
            hoverinfo="skip",
        ))


def _add_axes_triad_lines(
    fig: go.Figure,
    origin: np.ndarray,
    R: np.ndarray,
    *,
    L: float = 0.25,
    name_prefix: str = "",
):
    """
    Draw +x, +y, +z axes as RGB line segments of length L in **world** frame.

    Parameters
    ----------
    origin : (3,) axis origin in world coordinates
    R      : (3,3) rotation whose columns are the basis axes to draw in world
             (use identity for world axes; use camera R for camera axes)
    """
    o = origin.reshape(3)
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    ends = np.stack([o + L * ex, o + L * ey, o + L * ez], axis=0)  # (3,3)
    axes = [
        ("x", ends[0], "red"),
        ("y", ends[1], "green"),
        ("z", ends[2], "blue"),
    ]
    for label, p, color in axes:
        fig.add_trace(go.Scatter3d(
            x=[o[0], p[0]], y=[o[1], p[1]], z=[o[2], p[2]],
            mode="lines+markers+text",
            line=dict(width=5, color=color),
            marker=dict(size=2, color=color),
            text=[None, f"{name_prefix}{label}"],
            textposition="top center",
            name=f"{name_prefix}{label}",
            showlegend=False,
            hovertemplate=f"{name_prefix}{label} axis<extra></extra>",
        ))


# ---- Box & sphere meshing (matches your dataclasses) -------------------------
def _box_corners_from_bounds(bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> np.ndarray:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=np.float32)


def _box_corners_from_center_size(center, size) -> np.ndarray:
    c = _to_numpy3(center).reshape(3)
    s = _to_numpy3(size).reshape(3)
    hx, hy, hz = (s / 2.0).tolist()
    corners = np.array(
        [[-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
         [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]], dtype=np.float32
    )
    return corners + c


def _triangulated_box_vertices_faces(box: Box) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a triangulated AABB mesh for Plotly Mesh3d.

    Preference order:
      1) box.corners_world() if available,
      2) box.bounds  (your Box exposes this),
      3) (box.center, box.size)
    """
    if hasattr(box, "corners_world"):
        V = _to_numpy3(box.corners_world())  # (8,3)
    elif hasattr(box, "bounds"):
        V = _box_corners_from_bounds(box.bounds)
    else:
        V = _box_corners_from_center_size(box.center, box.size)

    F = np.array([  # two triangles per face
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    return V, F


def _sphere_mesh_grid(sphere: Sphere, *, nu: int = 40, nv: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = _to_numpy3(sphere.center).reshape(3)
    r = float(sphere.radius)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nv)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = c[0] + r * np.cos(vv) * np.cos(uu)
    y = c[1] + r * np.cos(vv) * np.sin(uu)
    z = c[2] + r * np.sin(vv)
    return x, y, z


def _grid_to_tris(nv: int, nu: int):
    """
    Convert an (nv, nu) parametric grid into triangle indices for Plotly Mesh3d.
    Returns three int arrays (i, j, k). If the grid is too small, returns empty arrays.
    """
    if nv < 2 or nu < 2:
        # Not enough cells to form triangles
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    tri_i, tri_j, tri_k = [], [], []

    def idx(a, b):  # linear index into flattened (nv, nu) grid
        return a * nu + b

    for a in range(nv - 1):
        for b in range(nu - 1):
            p00 = idx(a, b)
            p01 = idx(a, b + 1)
            p10 = idx(a + 1, b)
            p11 = idx(a + 1, b + 1)
            # two triangles per quad (p00, p01, p11) and (p00, p11, p10)
            tri_i.extend([p00, p00])
            tri_j.extend([p01, p11])
            tri_k.extend([p11, p10])

    return np.asarray(tri_i, dtype=np.int32), np.asarray(tri_j, dtype=np.int32), np.asarray(tri_k, dtype=np.int32)


# =============================================================================
# Main API
# =============================================================================
@torch.no_grad()
def plot_world_interactive(
    world: World,
    *,
    cameras: CameraLike = None,
    cam_indices: CamIdxSel = None,  # applies to any batched camera

    # ── style / opacity (defaults via VCFG) ─────────────────────────────────
    shape_face_alpha: float | None = None,
    shape_edge_alpha: float | None = None,   # edges are line overlays
    ray_alpha: float | None = None,
    ray_linewidth: float | None = None,
    samples_alpha: float | None = None,

    # ── rays / samples ─────────────────────────────────────────────────────
    draw_rays: bool = False,
    ray_mode: Literal["lines", "quiver"] = "lines",  # 'quiver'→short segments
    ray_step: int | None = None,
    rays_per_pose: Optional[int] = None,
    rays_indices: Optional[Union[torch.Tensor, np.ndarray, Iterable[int], Iterable[Iterable[int]]]] = None,

    draw_samples: bool = False,
    samples_rng: Optional[torch.Generator] = None,
    samples_deterministic: Optional[bool] = None,

    # ── external points (if given, overrides sampling) ─────────────────────
    external_points: Optional[Union[NDArray, torch.Tensor]] = None,
    external_point_size: float = 2.0,

    # ── axes / view / labels ───────────────────────────────────────────────
    draw_world_axes: bool = True,
    draw_camera_axes: bool = True,
    axes_length: float = 0.25,  # length of triad arms (world units)
    set_labels: bool = True,
    title: Optional[str] = None,
    show: bool = True,  # in notebooks, display the widget

    # ── visible ray length override ────────────────────────────────────────
    t_near: Optional[float] = None,
    t_far: Optional[float] = None,

    # ── file export ────────────────────────────────────────────────────────
    save_html: Optional[str] = None,
) -> Tuple[go.Figure, None]:
    """
    Interactive 3-D world viewer (Plotly).

    Semantics mirror the Matplotlib `plot_world(...)`:
      • single or batched `Camera`
      • `cam_indices` selection
      • `draw_rays`, `draw_samples`, and `rays_indices` control
      • optional `external_points` instead of sampling
      • `t_near`/`t_far` define visible ray segments
      • triads for **world** and for each **camera** to read axes orientation

    Returns
    -------
    (fig, None) : Use `(fig, ax)` unpacking compatibility in existing code.
    """
    # Resolve config defaults
    VC = VCFG
    shape_face_alpha = _clamp01(VC.shape_face_alpha if shape_face_alpha is None else float(shape_face_alpha))
    shape_edge_alpha = _clamp01(VC.shape_edge_alpha if shape_edge_alpha is None else float(shape_edge_alpha))
    ray_alpha        = _clamp01(VC.ray_alpha        if ray_alpha        is None else float(ray_alpha))
    ray_linewidth    = VC.ray_linewidth if ray_linewidth is None else float(ray_linewidth)
    samples_alpha    = _clamp01(VC.samples_alpha    if samples_alpha    is None else float(samples_alpha))
    ray_step         = VC.ray_step if ray_step is None else max(1, int(ray_step))

    fig = go.Figure()

    # ----- world geometry ------------------------------------------------------
    shapes = getattr(world, "shapes", [])
    for s in shapes:
        if isinstance(s, Box):
            V, F = _triangulated_box_vertices_faces(s)
            fig.add_trace(go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                i=F[:, 0], j=F[:, 1], k=F[:, 2],
                name="box",
                opacity=float(shape_face_alpha),
                flatshading=True,
                hoverinfo="skip",
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.1),
            ))
            # wireframe edges
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                _add_lines3d(fig, [np.stack([V[a], V[b]], axis=0)], "box-edge",
                             width=1.0, opacity=float(shape_edge_alpha))
        elif isinstance(s, Sphere):
            x, y, z = _sphere_mesh_grid(s, nu=40, nv=20)
            assert x.shape == y.shape == z.shape, "sphere mesh grids must have the same shape"
            nv, nu = x.shape
            pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            I, J, K = _grid_to_tris(nv, nu)
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=I, j=J, k=K,
                name="sphere",
                opacity=float(shape_face_alpha),
                flatshading=True,
                hoverinfo="skip",
                lighting=dict(ambient=0.5, diffuse=0.7, specular=0.1),
            ))

    # World frame axes triad
    if draw_world_axes:
        _add_axes_triad_lines(
            fig,
            origin=np.zeros(3, dtype=np.float32),
            R=np.eye(3, dtype=np.float32),
            L=float(axes_length),
            name_prefix="world:",
        )

    # ----- cameras + rays/samples ---------------------------------------------
    cams = _iter_cameras(cameras)
    if not cams:
        # finalize layout and return
        fig.update_layout(
            title=title or "World view",
            scene=dict(
                xaxis=dict(title="x" if set_labels else "", showgrid=True),
                yaxis=dict(title="y" if set_labels else "", showgrid=True),
                zaxis=dict(title="z" if set_labels else "", showgrid=True),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        if save_html:
            fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
        if show:
            fig.show()
        return fig, None

    color_idx = 0
    for cam in cams:
        is_batched = getattr(cam, "_is_batched", False)
        sel = cam._select_cam_indices(cam_indices) if is_batched else [0]
        H_wc_sel = cam.H_wc[sel] if is_batched else cam.H_wc

        need_rays = draw_rays or draw_samples or (external_points is not None)

        # Prepare ray indices (broadcast rules match your Matplotlib path)
        idx_user = _as_long_tensor(rays_indices, device=cam.H_wc.device)
        idx_for_camera = None
        if need_rays and idx_user is not None:
            if is_batched:
                if idx_user.ndim == 1:
                    K = idx_user.shape[0]
                    idx_for_camera = idx_user.view(1, K).expand(cam.B, K).contiguous()
                elif idx_user.ndim == 2:
                    if idx_user.shape[0] == cam.B:
                        idx_for_camera = idx_user
                    elif idx_user.shape[0] == len(sel):
                        # lift (Bsel,K) → (B,K) where non-selected rows copy row 0
                        Bsel, K = idx_user.shape
                        default_row = idx_user[0]
                        idx_for_camera = default_row.view(1, K).expand(cam.B, K).clone()
                        for row_pos, b_idx in enumerate(sel):
                            idx_for_camera[b_idx] = idx_user[row_pos]
                    else:
                        raise ValueError(
                            f"rays_indices rows ({idx_user.shape[0]}) must equal camera.B ({cam.B}) or len(sel) ({len(sel)})."
                        )
                else:
                    raise ValueError("For batched cameras, rays_indices must be (K,) or (B,K).")
            else:
                if idx_user.ndim != 1:
                    raise ValueError("For a single camera, rays_indices must be 1D (K,).")
                idx_for_camera = idx_user

        # Build rays efficiently using your Camera API
        O_all = D_all = None
        if need_rays:
            if idx_for_camera is not None:
                O_all, D_all = cam.get_rays_sampled(
                    rays_per_pose=None, frame="world", step=1, normalize=True, rng=samples_rng, indices=idx_for_camera
                )
            elif rays_per_pose is not None:
                O_all, D_all = cam.get_rays_sampled(
                    rays_per_pose=int(rays_per_pose), frame="world", step=ray_step, normalize=True, rng=samples_rng, indices=None
                )
            else:
                O_all, D_all = cam.get_rays(frame="world", step=ray_step, normalize=True)

            if is_batched:
                O_all, D_all = O_all[sel], D_all[sel]  # (Bsel,R,3)
            else:
                O_all, D_all = O_all[None, ...], D_all[None, ...]

        # Visible segment settings
        t0 = float(0.0 if t_near is None else t_near)
        vis_len = float((cam.t_far if t_far is None else t_far) - t0)
        vis_len = max(0.0, vis_len)
        seg_len_for_quiver = max(1e-6, vis_len if vis_len > 0 else float(cam.t_far))

        # Draw selected poses
        for k in range(len(sel)):
            col = f"C{(color_idx % 10)}"; color_idx += 1

            T = H_wc_sel[k] if is_batched else H_wc_sel
            R = T[:3, :3].detach().cpu().numpy().astype(np.float32)
            C = T[:3, 3].detach().cpu().numpy().astype(np.float32)

            # Camera axes triad (lets you read camera frame vs world)
            if draw_camera_axes:
                _add_axes_triad_lines(fig, origin=C, R=R, L=float(axes_length),
                                      name_prefix=f"cam{sel[k] if is_batched else 0}:")

            # Frustum
            fr = _camera_frustum_lines(C, R, fov_deg=60.0, scale=0.2 * seg_len_for_quiver)
            _add_lines3d(fig, fr, name=f"cam{sel[k] if is_batched else 0}", width=2.0, opacity=1.0)

            if not need_rays:
                continue

            O = _to_numpy3(O_all[k])  # (R,3)
            D = _to_numpy3(D_all[k])  # (R,3)

            # External point cloud has priority
            if external_points is not None:
                P = torch.as_tensor(external_points)
                if is_batched:
                    if P.ndim != 4 or P.shape[-1] != 3:
                        raise ValueError("For batched cameras, external_points must be (B,R,N,3).")
                    if P.shape[0] == len(sel):
                        cloud = _to_numpy3(P[k].reshape(-1, 3))
                    elif P.shape[0] == getattr(cam, "B", len(sel)):
                        cloud = _to_numpy3(P[sel[k]].reshape(-1, 3))
                    else:
                        raise ValueError(
                            f"external_points batch mismatch: got B={P.shape[0]}, expected {len(sel)} or {getattr(cam,'B',len(sel))}"
                        )
                else:
                    if P.ndim != 3 or P.shape[-1] != 3:
                        raise ValueError("For single camera, external_points must be (R,N,3).")
                    cloud = _to_numpy3(P.reshape(-1, 3))
                fig.add_trace(go.Scatter3d(
                    x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
                    mode="markers",
                    marker=dict(size=float(external_point_size), opacity=float(samples_alpha)),
                    name="samples", showlegend=False,
                ))

            elif draw_samples:
                # Sample along rays via Camera's sampler
                t_vals, deltas, pts = cam.sample_along_rays(
                    torch.from_numpy(O), torch.from_numpy(D),
                    rng=samples_rng, deterministic=samples_deterministic
                )  # (R,N), (R,N), (R,N,3)
                P = _to_numpy3(pts.reshape(-1, 3))
                fig.add_trace(go.Scatter3d(
                    x=P[:, 0], y=P[:, 1], z=P[:, 2],
                    mode="markers",
                    marker=dict(size=float(external_point_size), opacity=float(samples_alpha)),
                    name="samples", showlegend=False,
                ))

            if draw_rays:
                # 'lines' vs 'quiver' → both drawn as short segments [O+t0*D, O+(t0+L)*D]
                L = seg_len_for_quiver if ray_mode == "quiver" else (vis_len if vis_len > 0 else seg_len_for_quiver)
                P0 = O + t0 * D
                P1 = O + (t0 + L) * D
                for seg in np.stack([P0, P1], axis=1):  # (R,2,3)
                    _add_lines3d(fig, [seg], name="ray", width=float(ray_linewidth), opacity=float(ray_alpha))

    # ----- layout / output ------------------------------------------------------
    fig.update_layout(
        title=title or "World & Camera view",
        scene=dict(
            xaxis=dict(title="x" if set_labels else "", showgrid=True),
            yaxis=dict(title="y" if set_labels else "", showgrid=True),
            zaxis=dict(title="z" if set_labels else "", showgrid=True),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()
    return fig, None
