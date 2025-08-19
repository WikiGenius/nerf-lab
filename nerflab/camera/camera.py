# nerflab/camera.py
# nerflab/camera.py (only the changed/import bits shown)
from __future__ import annotations
from typing import Optional, Tuple, Literal, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .intrinsics import Intrinsics
from .transforms import invert_T
from ..config.config import CFG
from ..config.viz_config import viz_cfg
from .sampling import stratified_samples_batch
from ..viz.axis import style_3d_axis, axis_triad
from ..viz.pose import draw_pose_axes
import warnings

class Camera:
    """
    Minimal pinhole camera with optional batching.

    This class generates rays and sampled points for NeRF-style rendering.
    If constructor parameters are omitted, it falls back to values from the
    global `CFG` (intrinsics, near/far bounds, and samples-per-ray).

    Args:
        H_wc (Tensor | ndarray): Camera-to-World pose(s), shape (4,4) or (B,4,4).
        intr (Intrinsics, optional): Camera intrinsics. If None, uses CFG.intrinsics.
        t_bounds (tuple[float, float], optional): (t_near, t_far). If None, uses CFG.rays.
        n_points_per_ray (int, optional): Default samples per ray. If None, uses CFG.rays.N.
        device (torch.device, optional): Device for internal tensors.
        dtype (torch.dtype): Dtype for internal tensors (default: torch.float32).

    Notes:
        • Returns are batched when H_wc is batched.
        • Ray directions follow camera convention: +x right, +y up, -z forward.

    Key methods:
        get_rays(frame="world", step=1, normalize=True)
            -> origins, dirs with shapes (R,3) or (B,R,3).

        get_rays_sampled(rays_per_pose=None | K, indices=None, ...)
            -> subset of rays per pose, by count or explicit indices.

        sample_along_rays(O, D, deterministic=None)
            -> (t_vals, deltas, points) with shapes
            (R,N),(R,N),(R,N,3) or (B,R,N),(B,R,N),(B,R,N,3).

        plot_rays(...), plot_samples(...)
            -> Matplotlib helpers for quick inspection.

    Raises:
        ValueError: If input shapes are invalid or parameters are out of range.
    """


    def __init__(
        self,
        H_wc: Union[torch.Tensor, np.ndarray],
        intr: Optional[Intrinsics] = None,
        t_bounds: Optional[Tuple[float, float]] = None,
        n_points_per_ray: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # --- defaults from CFG if None ---
        if intr is None:
            ic = CFG.intrinsics
            intr = Intrinsics(
                fx=float(ic.fx), fy=float(ic.fy),
                width=int(ic.width), height=int(ic.height)
            )
        if t_bounds is None:
            rc = CFG.rays
            t_bounds = (float(rc.t_near), float(rc.t_far))
        if n_points_per_ray is None:
            n_points_per_ray = int(CFG.rays.N)

        # store intrinsics
        self.intr = intr

        self.deterministic = bool(CFG.rays.deterministic)
        
        # poses
        self.H_wc = torch.as_tensor(H_wc, device=device, dtype=dtype)
        if self.H_wc.shape[-2:] != (4, 4):
            raise ValueError(f"H_wc must have shape (...,4,4), got {self.H_wc.shape}")
        self.H_cw = invert_T(self.H_wc)

        # scalars
        self.t_near, self.t_far = map(float, t_bounds)
        self.n_points_per_ray = int(n_points_per_ray)

        # pixel-grid caches
        self._grid_cache_key: Optional[tuple[int, int, int, str, str]] = None
        self._u_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None


    # ── properties ───────────────────────────────────────────────────────
    @property
    def _is_batched(self) -> bool:
        return self.H_wc.ndim == 3

    @property
    def B(self) -> int:
        return 1 if not self._is_batched else int(self.H_wc.shape[0])

    # ==========================================================================
    # Rays — full grid
    # ==========================================================================
    def get_rays(
        self,
        frame: Literal["camera", "world"] = "world",
        step: int = 1,
        normalize: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for the (optionally strided) pixel grid.

        Returns
        -------
        origins, dirs :
            Unbatched → (R, 3)
            Batched   → (B, R, 3)
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        # default to pose's device/dtype
        device = device or self.H_wc.device
        dtype = dtype or self.H_wc.dtype

        # (R,), (R,) pixel coordinates & camera-frame directions (R,3)
        u, v = self._pixel_grid(step=step, device=device, dtype=dtype)
        dirs_c = self._dirs_cam(u, v, normalize=normalize).to(device=device, dtype=dtype)

        if frame == "camera":
            if self._is_batched:
                R = dirs_c.shape[0]
                O = torch.zeros((self.B, R, 3), device=device, dtype=dtype)
                D = dirs_c.expand(self.B, -1, -1)  # (B, R, 3)
                return O, D
            # single pose
            O = torch.zeros_like(dirs_c)  # (R, 3)
            return O, dirs_c

        # world-frame
        R_wc = self.H_wc[..., :3, :3]  # (3,3) or (B,3,3)
        t_wc = self.H_wc[..., :3, 3]   # (3,)   or (B,3)

        if self._is_batched:
            # dirs_c: (R,3) → (B,R,3)
            D = torch.einsum('bij,rj->bri', R_wc, dirs_c)  # (B,R,3)
            if normalize:
                D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            O = t_wc[:, None, :].expand(-1, D.shape[1], -1)  # (B, R, 3)
            return O.to(device=device, dtype=dtype), D.to(device=device, dtype=dtype)

        # single pose
        D = (R_wc @ dirs_c.T).T  # (R,3)
        if normalize:
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        O = t_wc.expand_as(D)    # (R,3)
        return O.to(device=device, dtype=dtype), D.to(device=device, dtype=dtype)

    # ==========================================================================
    # Rays — sampled per pose (random indices or explicit indices)
    # ==========================================================================
    @torch.no_grad()
    def get_rays_sampled(
        self,
        *,
        rays_per_pose: Optional[int] = None,         # None → all rays
        frame: Literal["camera", "world"] = "world",
        step: int = 1,
        normalize: bool = True,
        generator: Optional[torch.Generator] = None,
        indices: Optional[torch.Tensor] = None,      # (K,) or (B,K)
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return all rays or a random subset per pose (batched-safe).

        Priority:
        1) If `indices` is provided, use it directly (ignores `rays_per_pose`).
        2) Else if `rays_per_pose` is None, return all rays.
        3) Else sample exactly `rays_per_pose` per pose.
        """
        O, D = self.get_rays(frame=frame, step=step, normalize=normalize,
                             device=device, dtype=dtype)

        # ── explicit indices ──────────────────────────────────────────────────
        if indices is not None:
            if self._is_batched:
                B, R, _ = O.shape
                if indices.ndim != 2 or indices.shape[0] != B:
                    raise ValueError("indices must be (B, K) for batched sampling")
                idx = indices.to(device=O.device, dtype=torch.long)
                idx3 = idx.unsqueeze(-1).expand(-1, -1, 3)
                return torch.gather(O, 1, idx3), torch.gather(D, 1, idx3)
            # single
            if indices.ndim != 1:
                raise ValueError("indices must be (K,) for unbatched sampling")
            idx = indices.to(device=O.device, dtype=torch.long)
            return O.index_select(0, idx), D.index_select(0, idx)

        # ── decide K ──────────────────────────────────────────────────────────
        if rays_per_pose is None:
            return O, D  # all rays

        if self._is_batched:
            B, R, _ = O.shape
            K = int(rays_per_pose)
            if not (1 <= K <= R):
                raise ValueError(f"rays_per_pose must be in [1, {R}] (got {K})")
            if K == R:
                return O, D
            # Random per-pose without replacement (uniform)
            probs = torch.full((B, R), 1.0 / R, device=O.device, dtype=torch.float32)
            idx = torch.multinomial(probs, num_samples=K, replacement=False).to(torch.long)  # (B, K)
            idx3 = idx.unsqueeze(-1).expand(-1, -1, 3)
            return torch.gather(O, 1, idx3), torch.gather(D, 1, idx3)

        # single pose
        R = O.shape[0]
        K = int(rays_per_pose)
        if not (1 <= K <= R):
            raise ValueError(f"rays_per_pose must be in [1, {R}] (got {K})")
        if K == R:
            return O, D
        idx = torch.randperm(R, generator=generator, device=O.device)[:K]
        return O.index_select(0, idx), D.index_select(0, idx)

    # ==========================================================================
    # Sampling along rays — using your existing sampler (CFG-friendly)
    # ==========================================================================
    @torch.no_grad()
    def sample_along_rays(
        self,
        O: torch.Tensor,
        D: torch.Tensor,
        *,
        rng: Optional[torch.Generator] = None,
        deterministic: Optional[bool] = None,
    ):
        """
        Wrapper around nerflab.sampling.stratified_samples_batch.

        Supports:
        • Unbatched:  O,D ∈ (R, 3)    → (R, N), (R, N), (R, N, 3)
        • Batched:    O,D ∈ (B,R, 3)  → (B,R,N), (B,R,N), (B,R,N,3)
        """
        if O.shape != D.shape or O.shape[-1] != 3:
            raise ValueError(f"O and D must share shape (..., 3); got O{O.shape}, D{D.shape}")

        det = bool(deterministic) if deterministic is not None else self.deterministic

        # Pull config from *this* camera (set in __init__)
        t_near = self.t_near
        t_far  = self.t_far
        N      = self.n_points_per_ray

        if O.ndim == 2:  # (R,3)
            return stratified_samples_batch(
                O, D, t_near=t_near, t_far=t_far, N=N, rng=rng, deterministic=det
            )

        if O.ndim == 3:  # (B,R,3)
            B, R, _ = O.shape
            Of, Df = O.reshape(-1, 3), D.reshape(-1, 3)
            t, dlt, P = stratified_samples_batch(
                Of, Df, t_near=t_near, t_far=t_far, N=N, rng=rng, deterministic=det
            )
            return t.view(B, R, N), dlt.view(B, R, N), P.view(B, R, N, 3)

        raise ValueError("O and D must be (R,3) or (B,R,3)")


    # ==========================================================================
    # Visualization — rays
    # ==========================================================================
    def plot_rays(
        self,
        *,
        frame: Literal["camera", "world"] = "world",
        step: Optional[int] = None,
        mode: Literal["lines", "quiver", "points"] | None = None,
        color: Optional[str] = None,
        point_size: float = 6.0,
        points: Optional[Union[torch.Tensor, np.ndarray]] = None,  # for mode="points"
        draw_axes: bool = True,
        draw_world_axes: bool = True,
        cam_index: int = 0,
    ) -> None:
        """
        Visualize either:
          • Rays as lines or quivers, or
          • An external point cloud (mode="points").
        """
        step = viz_cfg.ray_step if step is None else step
        mode = viz_cfg.ray_mode if mode is None else mode
        color = "C0" if color is None else color

        fig = plt.figure(figsize=viz_cfg.figsize, dpi=viz_cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")

        if mode == "points":
            if points is None:
                raise ValueError("`points` must be provided when mode=='points'")
            P = torch.as_tensor(points, dtype=torch.float32)
            if P.ndim == 3:
                P = P.reshape(-1, 3)
            if P.shape[-1] != 3:
                raise ValueError("`points` last dim must be 3")
            P_np = P[::step].detach().cpu().numpy()
            ax.scatter(*P_np.T, s=point_size, c=color, depthshade=False)
        else:
            O, D = self.get_rays(frame=frame, step=step, normalize=True)
            if self._is_batched:
                O, D = O[cam_index], D[cam_index]
            ray_len = float(self.t_far)
            O_np, D_np = O.detach().cpu().numpy(), D.detach().cpu().numpy()

            if mode == "quiver":
                ax.quiver(*O_np.T, *D_np.T, length=ray_len, normalize=True, color=color, linewidth=0.6)
            elif mode == "lines":
                segs = np.stack([O_np, O_np + ray_len * D_np], axis=1)
                ax.add_collection3d(Line3DCollection(segs, colors=color, lw=0.7))
            else:
                raise ValueError("mode must be 'lines', 'quiver', or 'points'")

        if frame == "world":
            T = self.H_wc if not self._is_batched else self.H_wc[cam_index]
            cam_pos = T[:3, 3].detach().cpu().numpy()
            ax.scatter(*cam_pos, s=viz_cfg.camera_marker_size, c="red", marker="o", label="Cam")
            if draw_axes:
                draw_pose_axes(ax, T, scale=self.t_far * 0.2)

        if draw_world_axes:
            axis_triad(ax, length=viz_cfg.axis_triad_len)

        style_3d_axis(ax, invert=viz_cfg.axis_invert, elev=viz_cfg.axis_elev, azim=viz_cfg.axis_azim)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # Visualization — sampled points (uses your sampler by default)
    # ==========================================================================
    @torch.no_grad()
    def plot_samples(
        self,
        *,
        rays_per_pose: Optional[int] = None,  # None → all rays
        step: int = 1,                        # pre-grid stride for selecting rays
        frame: Literal["world", "camera"] = "world",
        show_rays: bool = True,
        samples_color: Optional[str] = None,
        samples_size: float = 6.0,
        cam_index: int = 0,
        rng: Optional[torch.Generator] = None,
        deterministic: Optional[bool] = None,  # forwarded to your sampler
    ) -> None:
        """
        Convenience visualizer:
          1) pick rays (all or subset),
          2) sample along them using your stratified sampler,
          3) scatter sampled points (and optionally draw the rays).
        """
        # 1) pick rays
        O, D = self.get_rays_sampled(
            rays_per_pose=rays_per_pose, step=step, frame=frame
        )  # (R,3) or (B,K,3)

        # 2) sample along rays via your sampler
        t_vals, deltas, pts = self.sample_along_rays(O, D, rng=rng, deterministic=deterministic)
        # shapes: (R,N),(R,N),(R,N,3) or (B,R,N),(B,R,N),(B,R,N,3)

        # 3) plot
        fig = plt.figure(figsize=viz_cfg.figsize, dpi=viz_cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")
        col = samples_color or "C1"

        if self._is_batched:
            P = pts[cam_index].reshape(-1, 3).detach().cpu().numpy()
            ax.scatter(*P.T, s=samples_size, c=col, depthshade=False)

            if show_rays:
                Oi, Di = O[cam_index], D[cam_index]
                ray_len = float(self.t_far)
                O_np, D_np = Oi.detach().cpu().numpy(), Di.detach().cpu().numpy()
                segs = np.stack([O_np, O_np + ray_len * D_np], axis=1)
                ax.add_collection3d(Line3DCollection(segs, colors="C0", lw=0.6))

            if frame == "world":
                T = self.H_wc[cam_index]
                cam_pos = T[:3, 3].detach().cpu().numpy()
                ax.scatter(*cam_pos, s=viz_cfg.camera_marker_size, c="red", marker="o", label="Cam")
                draw_pose_axes(ax, T, scale=self.t_far * 0.2)
        else:
            P = pts.reshape(-1, 3).detach().cpu().numpy()
            ax.scatter(*P.T, s=samples_size, c=col, depthshade=False)

            if show_rays:
                ray_len = float(self.t_far)
                O_np, D_np = O.detach().cpu().numpy(), D.detach().cpu().numpy()
                segs = np.stack([O_np, O_np + ray_len * D_np], axis=1)
                ax.add_collection3d(Line3DCollection(segs, colors="C0", lw=0.6))

            if frame == "world":
                cam_pos = self.H_wc[:3, 3].detach().cpu().numpy()
                ax.scatter(*cam_pos, s=viz_cfg.camera_marker_size, c="red", marker="o", label="Cam")
                draw_pose_axes(ax, self.H_wc, scale=self.t_far * 0.2)

        if frame == "world":
            axis_triad(ax, length=viz_cfg.axis_triad_len)

        style_3d_axis(ax, invert=viz_cfg.axis_invert, elev=viz_cfg.axis_elev, azim=viz_cfg.axis_azim)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # Diagnostics
    # ==========================================================================
    def print_info(self) -> None:
        print("=== Camera ===")
        print(f"Res: {self.intr.height} x {self.intr.width}")
        print(f"fx, fy: {self.intr.fx:.2f}, {self.intr.fy:.2f}")
        print(f"cx, cy: {self.intr.cx:.2f}, {self.intr.cy:.2f}")
        print(f"t_bounds: [{self.t_near}, {self.t_far}]")
        print("H_wc shape:", tuple(self.H_wc.shape))
        if self._is_batched:
            print("Batch size:", self.B)

    # ==========================================================================
    # Internals
    # ==========================================================================
    def _dirs_cam(self, u: torch.Tensor, v: torch.Tensor, normalize: bool) -> torch.Tensor:
        """
        Camera-frame directions for pixel coords (u,v).
        Convention: +x right, +y up, -z forward.
        """
        # keep scalar intrinsics as floats; cast tensor pieces explicitly
        fx, fy, cx, cy = float(self.intr.fx), float(self.intr.fy), float(self.intr.cx), float(self.intr.cy)
        one = torch.ones_like(u)
        dirs = torch.stack([
            (u - cx) / fx,
            -(v - cy) / fy,
            -one
        ], dim=-1)  # (R,3)

        if normalize:
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return dirs

    def _pixel_grid(
        self,
        step: int = 1,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flattened pixel grid (u,v) with stride = step."""
        H, W = int(self.intr.height), int(self.intr.width)
        key = (H, W, int(step), str(device), str(dtype))
        if key == self._grid_cache_key and self._u_cache is not None and self._v_cache is not None:
            return self._u_cache, self._v_cache

        v = torch.arange(0, H, step, device=device, dtype=dtype)
        u = torch.arange(0, W, step, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        self._grid_cache_key = key
        self._u_cache = uu.reshape(-1)
        self._v_cache = vv.reshape(-1)
        return self._u_cache, self._v_cache

    @staticmethod
    def _draw_pose_axes(ax, T: torch.Tensor, scale: float = 0.1):
        """Draw RGB axis triad at pose T.
        [DEPRECATED] Use nerflab.viz.pose.draw_pose_axes instead.
        """
        warnings.warn(
            "Camera._draw_pose_axes is deprecated; "
            "use nerflab.viz.pose.draw_pose_axes(ax, T, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        Tnp = T.detach().cpu().numpy()
        o = Tnp[:3, 3]
        R = Tnp[:3, :3]
        axes = np.eye(3) * scale
        colors = ["r", "g", "b"]
        for i in range(3):
            p = o + R @ axes[:, i]
            ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], colors[i], lw=2)


__all__ = ["Camera"]
