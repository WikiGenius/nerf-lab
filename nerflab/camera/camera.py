# nerflab/camera/camera.py
from __future__ import annotations

from typing import Optional, Tuple, Literal, Union, Iterable

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .intrinsics import Intrinsics
from .transforms import invert_T, validate_se3
from ..config.config import CFG
from ..config.viz_config import viz_cfg
from .sampling import stratified_samples_batch
from ..viz.axis import style_3d_axis, axis_triad
from ..viz.pose import draw_pose_axes


# ============================================================================ #
# Small helpers
# ============================================================================ #

def _ceil_div(a: int, b: int) -> int:
    """Ceiling division for positive ints: ceil(a/b) == (a + b - 1) // b."""
    return (a + b - 1) // b


def _clone_rng(rng: Optional[torch.Generator], device: torch.device) -> Optional[torch.Generator]:
    """
    Return a generator with the *same* state as `rng` (or None). Does not mutate the caller's rng.

    Notes
    -----
    - Required so that sampling functions don't advance the caller's RNG state.
    - Keeps device parity for torch.rand / randperm.
    """
    if rng is None:
        return None
    g = torch.Generator(device=device)
    g.set_state(rng.get_state())
    return g


def _quantize_uv_for_indexing(
    u: torch.Tensor, v: torch.Tensor, *, width: int, height: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize floating-point (u,v) to integer pixel coords for indexing,
    and clamp to image bounds.

    Returns
    -------
    (u_int, v_int) with same shape as inputs, dtype=torch.long.
    """
    # Round-to-nearest; swap to torch.floor for a different policy if desired.
    u_i = torch.round(u).to(torch.long)
    v_i = torch.round(v).to(torch.long)
    u_i.clamp_(0, width - 1)
    v_i.clamp_(0, height - 1)
    return u_i, v_i


def _linear_index(u_i: torch.Tensor, v_i: torch.Tensor, *, width: int) -> torch.Tensor:
    """
    Row-major linear index from (u_i, v_i).

    Works with shapes (K,) or (B,K). Returns same shape as inputs.
    """
    return v_i * width + u_i


# ============================================================================ #
# Camera
# ============================================================================ #

class Camera:
    """
    Minimal pinhole camera with optional batching.

    The class generates rays and sampled points for NeRF-style rendering.

    Args
    ----
    H_wc : (4,4) or (B,4,4)
        Camera-to-World homogeneous pose(s).
    intr : Intrinsics, optional
        Camera intrinsics; if None, uses CFG.intrinsics.
    t_bounds : (t_near, t_far), optional
        Near/Far for sampling/visualization; if None, uses CFG.rays.
    n_points_per_ray : int, optional
        Samples per ray for `sample_along_rays`; if None, uses CFG.rays.N.
    device : torch.device, optional
        Device for internal tensors.
    dtype : torch.dtype, default=torch.float32
        Dtype for internal tensors.
    repair_validate_se3 : bool, default=False
        If True, attempts to auto-fix invalid SE(3) inputs.

    Notes
    -----
    • Returns are batched when H_wc is batched.
    • Camera convention: +x right, +y up, -z forward.
    • `get_rays_sampled` samples K unique pixels on the step-grid (no replacement).
    """

    # ---------------------------------------------------------------------- #
    # Construction / state
    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        H_wc: Union[torch.Tensor, np.ndarray],
        intr: Optional[Intrinsics] = None,
        t_bounds: Optional[Tuple[float, float]] = None,
        n_points_per_ray: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        repair_validate_se3: bool = False,
    ):
        # Defaults from CFG
        if intr is None:
            ic = CFG.intrinsics
            intr = Intrinsics(
                fx=float(ic.fx),
                fy=float(ic.fy),
                width=int(ic.width),
                height=int(ic.height),
            )
        if t_bounds is None:
            rc = CFG.rays
            t_bounds = (float(rc.t_near), float(rc.t_far))
        if n_points_per_ray is None:
            n_points_per_ray = int(CFG.rays.N)

        # Store intrinsics (treated as immutable; if you change it, call clear_cache())
        self.intr = intr

        # Sampling behavior (wrapper respects this default)
        self.deterministic = bool(CFG.rays.deterministic)

        # Poses (validated)
        H_wc_t = torch.as_tensor(H_wc, device=device, dtype=dtype)
        self.H_wc = validate_se3(H_wc_t, name="H_wc", repair=repair_validate_se3)
        self.H_cw = invert_T(self.H_wc)

        # Scalars
        self.t_near, self.t_far = map(float, t_bounds)
        self.n_points_per_ray = int(n_points_per_ray)

        # Caches
        self._grid_cache_key: Optional[tuple[int, int, int, str, str]] = None
        self._u_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None

        self._rays_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pose_version: int = 0  # bump when H_wc changes

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #
    @property
    def _is_batched(self) -> bool:
        return self.H_wc.ndim == 3

    @property
    def B(self) -> int:
        return 1 if not self._is_batched else int(self.H_wc.shape[0])

    # Compact intrinsics signature for caching (immutable assumption)
    def _intr_sig(self) -> tuple:
        i = self.intr
        return (
            int(i.width),
            int(i.height),
            float(i.fx),
            float(i.fy),
            float(i.cx),
            float(i.cy),
        )

    # ---------------------------------------------------------------------- #
    # Public API — Rays (full grid or sampled)
    # ---------------------------------------------------------------------- #
    def get_rays(
        self,
        frame: Literal["camera", "world"] = "world",
        step: int = 1,
        normalize: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # NEW (opt-in):
        return_indices: bool = False,
    ):
        """
        Generate rays for the (optionally strided) pixel grid.

        Returns
        -------
        If return_indices == False (default):
            (O, D)
                Unbatched → (R, 3)
                Batched   → (B, R, 3)

        If return_indices == True:
            (O, D, idx_lin, uv_int)
                idx_lin : linear pixel indices into image (row-major)
                          Unbatched → (R,)
                          Batched   → (B, R)
                          (Broadcast case: see notes below.)
                uv_int  : integer pixel coords (u,v)
                          Unbatched → (R, 2)
                          Batched   → (B, R, 2)

        Notes
        -----
        • We cache only (O,D) for speed/memory. When return_indices=True, we skip cache.
        • For batched cameras with broadcasted (K,) u,v (internal path), idx_lin is (K,)
          and uv_int is (K,2). Duplicate them if needed per batch.
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        device = device or self.H_wc.device
        dtype = dtype or self.H_wc.dtype

        # Capacity / bounds check (required by tests)
        H, W = int(self.intr.height), int(self.intr.width)
        if step > H and step > W:
            raise ValueError(f"step={step} is too large for H={H}, W={W} (no rays).")

        # Cache lookup only when we don't need indices
        if not return_indices:
            key = self._rays_cache_key(
                frame=frame, step=step, normalize=normalize, device=device, dtype=dtype
            )
            cached = self._rays_cache.get(key)
            if cached is not None:
                return cached

        # Build pixel grid
        u, v = self._pixel_grid(step=step, device=device, dtype=dtype)  # (R,), (R,)
        if u.numel() == 0:
            raise ValueError(
                f"Empty pixel grid for step={step}, H={H}, W={W} — check intrinsics."
            )

        # Compose; ask for indices only if requested to avoid extra work
        if return_indices:
            O, D, idx_lin, uv_int = self._compose_rays_from_uv(
                u=u,
                v=v,
                frame=frame,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=True,
                image_hw=(H, W),
            )
        else:
            O, D = self._compose_rays_from_uv(
                u=u,
                v=v,
                frame=frame,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=False,
            )
            idx_lin = uv_int = None

        if not return_indices:
            key = self._rays_cache_key(
                frame=frame, step=step, normalize=normalize, device=device, dtype=dtype
            )
            self._rays_cache[key] = (O, D)
            return O, D

        return O, D, idx_lin, uv_int

    @torch.no_grad()
    def get_rays_sampled(
        self,
        *,
        rays_per_pose: Optional[int] = None,
        frame: Literal["camera", "world"] = "world",
        step: int = 1,
        normalize: bool = True,
        rng: Optional[torch.Generator] = None,
        indices: Optional[torch.Tensor] = None,  # (K,) or (B,K) on step-grid
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # NEW (opt-in):
        return_indices: bool = False,
    ):
        """
        Efficient ray sub-sampling.

        Priority:
        1) If `indices` is given → interpret over the step-grid and compute (u,v) directly.
        2) If `rays_per_pose` is None → return ALL rays via cached full-grid path.
        3) Else → sample K **unique** pixels uniformly w/o replacement on the step-grid.

        Returns
        -------
        If return_indices == False (default):
            (origins, dirs)
                Unbatched → (K,3)
                Batched   → (B,K,3)

        If return_indices == True:
            (origins, dirs, idx_lin, uv_int)
                idx_lin : (K,) or (B,K)  (row-major indices)
                uv_int  : (K,2) or (B,K,2)
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        device = device or self.H_wc.device
        dtype = dtype or self.H_wc.dtype
        B = self.B

        # Capacity / bounds check (required by tests)
        H, W = int(self.intr.height), int(self.intr.width)
        if step > H and step > W:
            raise ValueError(f"step={step} is too large for H={H}, W={W} (no rays).")

        Hw = _ceil_div(H, step)
        Ww = _ceil_div(W, step)
        R_full = Hw * Ww  # sites per pose on the step-grid
        if R_full == 0:
            raise ValueError(f"step={step} is too large for H={H}, W={W} (no rays).")

        # 1) Explicit indices over step-grid
        if indices is not None:
            idx = indices.to(device=device, dtype=torch.long)
            if self._is_batched:
                if idx.ndim != 2 or idx.shape[0] != B:
                    raise ValueError("indices must be (B,K) for batched sampling")
                v_idx = (idx // Ww) * step  # (B,K)
                u_idx = (idx % Ww) * step   # (B,K)
            else:
                if idx.ndim != 1:
                    raise ValueError("indices must be (K,) for unbatched sampling")
                v_idx = (idx // Ww) * step  # (K,)
                u_idx = (idx % Ww) * step   # (K,)

            if return_indices:
                O, D, idx_lin, uv_int = self._compose_rays_from_uv(
                    u=u_idx,
                    v=v_idx,
                    frame=frame,
                    normalize=normalize,
                    device=device,
                    dtype=dtype,
                    return_indices=True,
                    image_hw=(H, W),
                )
                return O, D, idx_lin, uv_int
            else:
                O, D = self._compose_rays_from_uv(
                    u=u_idx,
                    v=v_idx,
                    frame=frame,
                    normalize=normalize,
                    device=device,
                    dtype=dtype,
                    return_indices=False,
                )
                return O, D

        # 2) All rays
        if rays_per_pose is None:
            return self.get_rays(
                frame=frame,
                step=step,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=return_indices,
            )

        # 3) Subsample K rays (unique on the step-grid)
        K = int(rays_per_pose)
        if K < 1:
            raise ValueError("rays_per_pose must be >= 1")
        if K > R_full:
            raise ValueError(f"rays_per_pose must be in [1, {R_full}] (got {K})")
        if K == R_full:
            return self.get_rays(
                frame=frame,
                step=step,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=return_indices,
            )

        u, v = self._sample_pixel_indices(
            step=step, K=K, rng=rng, device=device, dtype=dtype, B=B
        )

        if return_indices:
            O, D, idx_lin, uv_int = self._compose_rays_from_uv(
                u=u,
                v=v,
                frame=frame,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=True,
                image_hw=(H, W),
            )
            return O, D, idx_lin, uv_int
        else:
            O, D = self._compose_rays_from_uv(
                u=u,
                v=v,
                frame=frame,
                normalize=normalize,
                device=device,
                dtype=dtype,
                return_indices=False,
            )
            return O, D

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
        Wrapper around `nerflab.sampling.stratified_samples_batch`.

        Supports
        --------
        • Unbatched:  O,D ∈ (R, 3)    → (R, N), (R, N), (R, N, 3)
        • Batched:    O,D ∈ (B,R, 3)  → (B,R,N), (B,R,N), (B,R,N,3)
        """
        if O.shape != D.shape or O.shape[-1] != 3:
            raise ValueError(
                f"O and D must share shape (..., 3); got O{O.shape}, D{D.shape}"
            )

        det = bool(deterministic) if deterministic is not None else self.deterministic
        t_near, t_far, N = self.t_near, self.t_far, self.n_points_per_ray

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

    # ---------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def plot_rays(
        self,
        *,
        frame: Literal["camera", "world"] = "world",
        step: Optional[int] = None,
        mode: Literal["lines", "quiver", "points"] | None = None,
        color: Optional[str] = None,
        point_size: float = 6.0,
        points: Optional[Union[torch.Tensor, np.ndarray]] = None,
        draw_axes: bool = True,
        draw_world_axes: bool = True,
        cam_index: int = 0,
        cam_indices: Optional[Union[int, Iterable[int], Literal["all"]]] = None,
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
            ax.scatter(*P_np.T, s=point_size, c=color or "C1", depthshade=False)
        else:
            O, D = self.get_rays(frame=frame, step=step or 1, normalize=True)
            # Loop over cameras uniformly (batched or single)
            for k, ci in enumerate(self._select_cam_indices(cam_indices or cam_index)):
                if self._is_batched:
                    Oi, Di = O[ci], D[ci]
                    col = color if color is not None else f"C{(k % 10)}"
                    self._viz_add_rays(ax, Oi, Di, mode=mode, color=col)
                    if frame == "world":
                        T = self.H_wc[ci]
                        self._viz_add_camera(ax, T, color=col, draw_axes=draw_axes)
                else:
                    self._viz_add_rays(ax, O, D, mode=mode, color=color or "C0")
                    if frame == "world":
                        self._viz_add_camera(ax, self.H_wc, color="red", draw_axes=draw_axes)
                    break  # single camera

        if frame == "world":
            axis_triad(ax, length=viz_cfg.axis_triad_len)

        style_3d_axis(
            ax,
            invert=viz_cfg.axis_invert,
            elev=viz_cfg.axis_elev,
            azim=viz_cfg.axis_azim,
        )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def plot_samples(
        self,
        *,
        rays_per_pose: Optional[int] = None,
        step: int = 1,
        frame: Literal["world", "camera"] = "world",
        show_rays: bool = True,
        samples_color: Optional[str] = None,
        samples_size: float = 6.0,
        cam_index: int = 0,
        cam_indices: Optional[Union[int, Iterable[int], Literal["all"]]] = None,
        rng: Optional[torch.Generator] = None,
        deterministic: Optional[bool] = None,
    ) -> None:
        """
        Convenience visualizer:
          1) pick rays (all or subset),
          2) sample along them,
          3) scatter sampled points (and optionally draw the rays).
        """
        O, D = self.get_rays_sampled(
            rays_per_pose=rays_per_pose, step=step, frame=frame, rng=rng
        )

        fig = plt.figure(figsize=viz_cfg.figsize, dpi=viz_cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Loop over cameras uniformly (batched or single)
        for k, ci in enumerate(self._select_cam_indices(cam_indices or cam_index)):
            if self._is_batched:
                Oi, Di = O[ci], D[ci]
                t_vals, deltas, pts = self.sample_along_rays(
                    Oi, Di, rng=rng, deterministic=deterministic
                )
                P = pts.reshape(-1, 3).detach().cpu().numpy()
                col = samples_color or f"C{(k+1) % 10}"
                ax.scatter(*P.T, s=samples_size, c=col, depthshade=False, label=f"samples cam {ci}")
                if show_rays:
                    self._viz_add_rays(ax, Oi, Di, mode="lines", color="C0")
                if frame == "world":
                    self._viz_add_camera(ax, self.H_wc[ci], color=col, draw_axes=True)
            else:
                t_vals, deltas, pts = self.sample_along_rays(
                    O, D, rng=rng, deterministic=deterministic
                )
                P = pts.reshape(-1, 3).detach().cpu().numpy()
                ax.scatter(*P.T, s=samples_size, c=samples_color or "C1", depthshade=False)
                if show_rays:
                    self._viz_add_rays(ax, O, D, mode="lines", color="C0")
                if frame == "world":
                    self._viz_add_camera(ax, self.H_wc, color="red", draw_axes=True)
                break  # single camera

        if frame == "world":
            axis_triad(ax, length=viz_cfg.axis_triad_len)

        style_3d_axis(
            ax,
            invert=viz_cfg.axis_invert,
            elev=viz_cfg.axis_elev,
            azim=viz_cfg.axis_azim,
        )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------- #
    # Diagnostics / device / pose updates
    # ---------------------------------------------------------------------- #
    def print_info(self) -> None:
        print("=== Camera ===")
        print(f"Res: {self.intr.height} x {self.intr.width}")
        print(f"fx, fy: {self.intr.fx:.2f}, {self.intr.fy:.2f}")
        print(f"cx, cy: {self.intr.cx:.2f}, {self.intr.cy:.2f}")
        print(f"t_bounds: [{self.t_near}, {self.t_far}]")
        print("H_wc shape:", tuple(self.H_wc.shape))
        if self._is_batched:
            print("Batch size:", self.B)

    def clear_cache(self) -> None:
        """Clear all per-instance caches (rays + pixel grid)."""
        self._rays_cache.clear()
        self._grid_cache_key = None
        self._u_cache = None
        self._v_cache = None

    def set_poses(
        self, H_wc: Union[torch.Tensor, np.ndarray], *, repair: bool = False
    ) -> None:
        """
        Replace camera pose(s) and invalidate caches.
        Accepts (4,4) or (B,4,4).
        """
        H_wc_t = torch.as_tensor(H_wc, device=self.H_wc.device, dtype=self.H_wc.dtype)
        self.H_wc = validate_se3(H_wc_t, name="H_wc", repair=repair)
        self.H_cw = invert_T(self.H_wc)
        self._pose_version += 1
        # Pixel-grid cache still valid if (W,H,step,device,dtype) unchanged
        self._rays_cache.clear()

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """
        Move internal tensors to device/dtype.
        Invalidates ray cache because device/dtype participate in the key.
        """
        device = device or self.H_wc.device
        dtype = dtype or self.H_wc.dtype
        self.H_wc = self.H_wc.to(device=device, dtype=dtype)
        self.H_cw = self.H_cw.to(device=device, dtype=dtype)
        # Pixel grid cache depends on device/dtype too
        self._rays_cache.clear()
        self._grid_cache_key = None
        self._u_cache = None
        self._v_cache = None
        return self

    # ---------------------------------------------------------------------- #
    # Internals (math / composition / sampling / viz)
    # ---------------------------------------------------------------------- #
    def _dirs_cam(
        self, u: torch.Tensor, v: torch.Tensor, normalize: bool
    ) -> torch.Tensor:
        """
        Camera-frame direction vectors for pixel coords (u,v).
        Convention: +x right, +y up, -z forward.

        Notes
        -----
        • This function expects `u` and `v` to already be tensors on
          the correct device/dtype.
        """
        inv_fx = 1.0 / float(self.intr.fx)
        inv_fy = 1.0 / float(self.intr.fy)
        cx = float(self.intr.cx)
        cy = float(self.intr.cy)

        one = torch.ones_like(u)
        dirs = torch.stack(
            [(u - cx) * inv_fx, -(v - cy) * inv_fy, -one], dim=-1
        )  # (...,3)
        if normalize:
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return dirs

    def _pixel_grid(
        self,
        step: int = 1,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flattened pixel grid (u,v) with stride = step.

        Returns
        -------
        (u, v)
            Each is shape (R,), where R = ceil(H/step) * ceil(W/step).
        """
        H, W = int(self.intr.height), int(self.intr.width)
        key = (H, W, int(step), str(device), str(dtype))
        if (
            key == self._grid_cache_key
            and self._u_cache is not None
            and self._v_cache is not None
        ):
            return self._u_cache, self._v_cache

        v = torch.arange(0, H, step, device=device, dtype=dtype)
        u = torch.arange(0, W, step, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        self._grid_cache_key = key
        self._u_cache = uu.reshape(-1)
        self._v_cache = vv.reshape(-1)
        return self._u_cache, self._v_cache

    def _sample_pixel_indices(
        self,
        *,
        step: int,
        K: int,
        rng: Optional[torch.Generator],
        device: torch.device,
        dtype: torch.dtype,
        B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample K unique pixel coordinates (u, v) per pose from a step-strided grid.

        Overview
        --------
        - Define coarse lattice using `step`:
              Hw = ceil(H / step),  Ww = ceil(W / step),  Rg = Hw * Ww
        - Draw K **distinct** sites (without replacement) uniformly at random
          from the Rg lattice sites for each pose.
        - Map flat site indices -> (u, v):
              v = (idx // Ww) * step
              u = (idx %  Ww) * step
        - Return float tensors (to `dtype`) for downstream ray math.

        Shapes
        ------
        - Unbatched camera   : u, v are (K,)
        - Batched (B poses)  : u, v are (B, K)

        Determinism
        -----------
        Pass a `torch.Generator` (rng) for reproducible results.
        Uses a generator-aware top-K trick (no `multinomial`) in batched mode.
        """
        if step < 1:
            raise ValueError("step must be >= 1")
        if K < 1:
            raise ValueError("K (rays_per_pose) must be >= 1")

        H, W = int(self.intr.height), int(self.intr.width)
        if step > H and step > W:
            raise ValueError(f"step={step} too large for H={H}, W={W} (no grid sites).")

        Hw = _ceil_div(H, step)
        Ww = _ceil_div(W, step)
        Rg = Hw * Ww
        if K > Rg:
            raise ValueError(
                f"Requested K={K} > grid sites {Rg} (H={H}, W={W}, step={step})."
            )

        gen = _clone_rng(rng, device)
        if B > 1:  # batched
            # i.i.d. U(0,1) scores → take top-K smallest (shuffle semantics w/ generator)
            scores = torch.rand((B, Rg), generator=gen, device=device)
            idx_flat = scores.topk(K, dim=1, largest=False).indices  # (B, K)
            v_idx = (idx_flat // Ww) * step  # (B, K)
            u_idx = (idx_flat % Ww) * step   # (B, K)
        else:  # unbatched
            idx_flat = torch.randperm(Rg, generator=gen, device=device)[:K]  # (K,)
            v_idx = (idx_flat // Ww) * step  # (K,)
            u_idx = (idx_flat % Ww) * step   # (K,)

        return u_idx.to(dtype=dtype), v_idx.to(dtype=dtype)

    def _compose_rays_from_uv(
        self,
        *,
        u: torch.Tensor,
        v: torch.Tensor,
        frame: Literal["camera", "world"],
        normalize: bool,
        device: torch.device,
        dtype: torch.dtype,
        # NEW (opt-in for caller):
        return_indices: bool = False,
        image_hw: Optional[Tuple[int, int]] = None,  # (H, W) when return_indices=True
    ):
        """
        Build (O,D) for given (u,v) without building the full grid again.

        Inputs
        ------
        • Unbatched u,v : (K,)
        • Batched u,v   : (K,) (broadcast across B)  OR (B,K)

        Returns
        -------
        If return_indices == False (default)
            O, D : (K,3) or (B,K,3)

        If return_indices == True
            O, D, idx_lin, uv_int
              idx_lin : None or (K,) or (B,K)   [linear image indices]
              uv_int  : None or (K,2) or (B,K,2)[integer pixel coords]

        Notes
        -----
        • For broadcasted (K,) in batched cameras, idx/uv are returned as (K,) / (K,2)
          (identical across batch). Duplicate if needed.
        """
        # Normalize devices
        u = u.to(device=device)
        v = v.to(device=device)

        need_norm_cam = frame == "camera"
        want_idx = bool(return_indices)

        if want_idx and image_hw is None:
            raise ValueError("image_hw=(H,W) must be provided when return_indices=True.")
        if want_idx:
            H_img, W_img = int(image_hw[0]), int(image_hw[1])

        # --------------------- Unbatched camera ---------------------
        if not self._is_batched:
            K = int(u.shape[0])
            # Directions in camera frame
            dirs_c = self._dirs_cam(u.to(dtype=dtype), v.to(dtype=dtype), normalize=need_norm_cam)  # (K,3)

            if frame == "camera":
                D = dirs_c
                O = torch.zeros_like(D, device=device, dtype=dtype)

                if not want_idx:
                    return O, D

                u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)
                idx_lin = _linear_index(u_i, v_i, width=W_img)
                uv_int = torch.stack([u_i, v_i], dim=-1)  # (K,2)
                return O, D, idx_lin, uv_int

            # world frame
            R_wc = self.H_wc[:3, :3]
            t_wc = self.H_wc[:3, 3]
            D = (R_wc @ dirs_c.T).T  # (K,3)
            if normalize:
                D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            O = t_wc.expand_as(D).clone().to(dtype=dtype)
            D = D.to(dtype=dtype)

            if not want_idx:
                return O, D

            u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)
            idx_lin = _linear_index(u_i, v_i, width=W_img)
            uv_int = torch.stack([u_i, v_i], dim=-1)  # (K,2)
            return O, D, idx_lin, uv_int

        # --------------------- Batched cameras ---------------------
        B = self.B

        # Case A: broadcast same (K,) u,v to all B views
        if u.ndim == 1:
            K = int(u.shape[0])
            dirs_c = self._dirs_cam(u.to(dtype=dtype), v.to(dtype=dtype), normalize=need_norm_cam)  # (K,3)

            if frame == "camera":
                D = dirs_c.expand(B, K, 3).contiguous()
                O = torch.zeros_like(D, device=device, dtype=dtype)

                if not want_idx:
                    return O, D

                # Indices identical for every batch item (returned once)
                u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)  # (K,)
                idx_lin = _linear_index(u_i, v_i, width=W_img)  # (K,)
                uv_int = torch.stack([u_i, v_i], dim=-1)       # (K,2)
                return O, D, idx_lin, uv_int

            R_wc = self.H_wc[..., :3, :3]           # (B,3,3)
            t_wc = self.H_wc[..., :3, 3]            # (B,3)
            D_c = dirs_c[None, ...].expand(B, K, 3) # (B,K,3)
            D = torch.einsum("bij,bkj->bki", R_wc, D_c)
            if normalize:
                D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            O = t_wc[:, None, :].expand(B, K, 3).clone().to(dtype=dtype)
            D = D.to(dtype=dtype)

            if not want_idx:
                return O, D

            u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)  # (K,)
            idx_lin = _linear_index(u_i, v_i, width=W_img)                          # (K,)
            uv_int = torch.stack([u_i, v_i], dim=-1)                                # (K,2)
            return O, D, idx_lin, uv_int

        # Case B: per-batch (B,K) u,v
        if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape or u.shape[0] != B:
            raise ValueError("For batched cameras, u and v must be (K,) or (B,K) with matching shapes.")
        _, K = u.shape

        uc = u.reshape(-1).to(dtype=dtype)  # (B*K,)
        vc = v.reshape(-1).to(dtype=dtype)
        dirs_c = self._dirs_cam(uc, vc, normalize=need_norm_cam)  # (B*K,3)

        if frame == "camera":
            D = dirs_c.view(B, K, 3)
            O = torch.zeros_like(D, device=device, dtype=dtype)

            if not want_idx:
                return O, D

            u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)  # (B,K)
            idx_lin = _linear_index(u_i, v_i, width=W_img)                          # (B,K)
            uv_int = torch.stack([u_i, v_i], dim=-1)                                # (B,K,2)
            return O, D, idx_lin, uv_int

        R_wc = self.H_wc[..., :3, :3]  # (B,3,3)
        t_wc = self.H_wc[..., :3, 3]   # (B,3)
        D = torch.einsum("bij,bkj->bki", R_wc, dirs_c.view(B, K, 3))
        if normalize:
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        O = t_wc[:, None, :].expand(B, K, 3).clone().to(dtype=dtype)
        D = D.to(dtype=dtype)

        if not want_idx:
            return O, D

        u_i, v_i = _quantize_uv_for_indexing(u, v, width=W_img, height=H_img)  # (B,K)
        idx_lin = _linear_index(u_i, v_i, width=W_img)                          # (B,K)
        uv_int = torch.stack([u_i, v_i], dim=-1)                                # (B,K,2)
        return O, D, idx_lin, uv_int

    def _select_cam_indices(
        self,
        cam_indices: Optional[Union[int, Iterable[int], Literal["all"]]],
    ) -> list[int]:
        """
        Normalize camera selection to a list of indices.
        - None  -> [0]
        - int   -> [that index]
        - "all" -> [0..B-1]
        - iterable -> list(int)
        """
        if not self._is_batched:
            return [0]
        if cam_indices is None:
            return [0]
        if cam_indices == "all":
            return list(range(self.B))
        if isinstance(cam_indices, int):
            cam_indices = [cam_indices]
        idxs = [int(i) for i in cam_indices]
        for i in idxs:
            if not (0 <= i < self.B):
                raise ValueError(f"cam index {i} out of range [0, {self.B-1}]")
        return idxs

    # ------------------------------- #
    # Visualization helpers (DRY)
    # ------------------------------- #
    def _viz_add_rays(
        self,
        ax,
        O: torch.Tensor,  # (R,3)
        D: torch.Tensor,  # (R,3)
        *,
        mode: Literal["lines", "quiver"],
        color: str,
    ) -> None:
        """Add ray glyphs for a single camera (unbatched O,D)."""
        ray_len = float(self.t_far)
        O_np = O.detach().cpu().numpy()
        D_np = D.detach().cpu().numpy()

        if mode == "quiver":
            ax.quiver(
                *O_np.T, *D_np.T,
                length=ray_len, normalize=True,
                color=color, linewidth=0.6,
            )
        elif mode == "lines":
            segs = np.stack([O_np, O_np + ray_len * D_np], axis=1)
            ax.add_collection3d(Line3DCollection(segs, colors=color, lw=0.7))
        else:
            raise ValueError("mode must be 'lines' or 'quiver'")

    def _viz_add_camera(self, ax, T: torch.Tensor, *, color: str, draw_axes: bool) -> None:
        """Add camera marker (and optional axes) for a single camera pose."""
        cam_pos = T[:3, 3].detach().cpu().numpy()
        ax.scatter(*cam_pos, s=viz_cfg.camera_marker_size, c=color, marker="o", label="Cam")
        if draw_axes:
            draw_pose_axes(ax, T, scale=self.t_far * 0.2)

    # ---------------------------------------------------------------------- #
    # Cache key includes pose version AND intrinsics signature
    # ---------------------------------------------------------------------- #
    def _rays_cache_key(
        self,
        *,
        frame: str,
        step: int,
        normalize: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """Key that uniquely identifies a rays-computation state."""
        return (
            id(self.H_wc),            # tensor identity
            self._pose_version,       # bump on set_poses()
            self._intr_sig(),         # intrinsics signature
            frame,
            int(step),
            bool(normalize),
            str(device),
            str(dtype),
        )


__all__ = ["Camera"]
