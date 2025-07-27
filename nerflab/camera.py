from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Literal
from nerflab.utils import invert_T

# ---------- Intrinsics ----------
@dataclass(frozen=True)
class Intrinsics:
    """Simple pinhole intrinsics."""
    fx: float; fy: float
    width: int; height: int
    cx = property(lambda self: self.width / 2)
    cy = property(lambda self: self.height / 2)
    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        K = np.array([[self.fx, 0,       self.cx],
                      [0,       self.fy, self.cy],
                      [0,       0,       1     ]], dtype=np.float32)
        return K

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> "Intrinsics":
        """Create from a 3×3 K."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return cls(fx, fy, cx, cy, width, height)



# ---------- Camera ----------
class Camera:
    """
    Minimal pinhole camera.

    You give:
        - intr: Intrinsics
        - H_wc: 4×4 camera→world transform
        - t_bounds: (near, far) used only for point-mode plotting/default sampling

    You get:
        - get_rays(frame="camera"/"world")
        - plot_rays()
    """
    def __init__(self,
                 intr: Intrinsics,
                 H_wc: np.ndarray,
                 t_bounds: Tuple[float, float] = (0.0, 1.0),
                 n_points_per_ray: int = 200):
        self.intr = intr
        self.H_wc = H_wc.astype(np.float32)
        self.H_cw = invert_T(self.H_wc)
        self.t_near, self.t_far = t_bounds
        self.n_points_per_ray = int(n_points_per_ray)

    # ---------- Rays ----------
    def get_rays(self,
                 frame: Literal["camera", "world"] = "world",
                 step: int = 1,
                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ray origins & directions.

        Parameters
        ----------
        frame : {"camera", "world"}
            Coordinate frame for the returned rays.
        step : int
            Subsample stride over pixels (>=1).
        normalize : bool
            If True, directions are unit length.

        Returns
        -------
        origins : (R,3)
        dirs    : (R,3)
        """
        u, v = self._pixel_grid(step)

        # Directions in camera frame (convention: +x right, +y up, -z forward)
        dirs_c = self._dirs_cam(u, v, normalize)

        if frame == "camera":
            origins_c = np.zeros_like(dirs_c)
            return origins_c, dirs_c

        # ---- world frame ----
        R_wc = self.H_wc[:3, :3]
        t_wc = self.H_wc[:3, 3]
        dirs_w = (R_wc @ dirs_c.T).T
        if normalize:
            dirs_w /= np.linalg.norm(dirs_w, axis=-1, keepdims=True)
        origins_w = np.broadcast_to(t_wc, dirs_w.shape)
        return origins_w, dirs_w

    # ---------- Plotting ----------
    def plot_rays(self,
                  frame: Literal["camera", "world"] = "world",
                  step: int = 40,
                  mode: Literal["lines", "quiver", "points"] = "lines",
                  color: str = "C0",
                  point_size: float = 6.0,
                  draw_axes: bool = True,
                  draw_world_axes: bool = True) -> None:
        """
        Visualize rays. Marks camera in world frame.

        Parameters
        ----------
        frame : "camera" | "world"
        step : int
            Subsampling stride.
        mode : "lines" | "quiver" | "points"
        color : str
        point_size : float
        draw_axes : bool
        """
        O, D = self.get_rays(frame=frame, step=step, normalize=True)

        # Ray length (for quiver/lines).
        t = self.t_far # float

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        if mode == "quiver":
            ax.quiver(O[:, 0], O[:, 1], O[:, 2],
                      D[:, 0], D[:, 1], D[:, 2],
                      length=t, normalize=True, color=color, linewidth=0.6)
        elif mode == "lines":
            P = O + t * D
            for s, e in zip(O, P):
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color, lw=0.7)
        elif mode == "points":
            ts = np.linspace(self.t_near, self.t_far, self.n_points_per_ray)
            P = (O[:, None, :] + ts[None, :, None] * D[:, None, :]).reshape(-1, 3)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=point_size, c=color, depthshade=False)
        else:
            raise ValueError("mode must be 'lines', 'quiver', or 'points'")

        # mark camera if world frame
        ax.scatter(*[0,0,0], s=50, c='blue', marker='x', label='world')
        if frame == "world":
            cam_pos = self.H_wc[:3, 3]
            ax.scatter(*cam_pos, s=50, c='red', marker='o', label='Camera')
            if draw_axes:
                self._draw_pose_axes(ax, self.H_wc, scale=t * 0.2)
        if draw_world_axes:                      # <--- NEW
            self._draw_pose_axes(ax, np.eye(4, dtype=np.float32), scale=t * 0.3)

        # show legend only if something is labeled
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    def print_info(self) -> None:
        """Pretty print camera parameters."""
        print("=== Camera ===")
        print(f"Res: {self.intr.height} x {self.intr.width}")
        print(f"fx, fy: {self.intr.fx:.2f}, {self.intr.fy:.2f}")
        print(f"cx, cy: {self.intr.cx:.2f}, {self.intr.cy:.2f}")
        print(f"t_bounds: [{self.t_near}, {self.t_far}]")
        print("H_wc:\n", self.H_wc)

    # ---------- Utils ----------
    def _dirs_cam(self, u: np.ndarray, v: np.ndarray, normalize: bool) -> np.ndarray:
        """Camera-frame directions for pixel coords (u,v)."""
        dirs = np.stack([
            (u - self.intr.cx) / self.intr.fx,
            -(v - self.intr.cy) / self.intr.fy,
            -np.ones_like(u)
        ], axis=-1).astype(np.float32)

        if normalize:
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs
    def _pixel_grid(self, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Return flattened (u,v) grid with given step."""
        v, u = np.mgrid[0:self.intr.height:step, 0:self.intr.width:step]
        return u.reshape(-1), v.reshape(-1)

    @staticmethod
    def _draw_world_axes(ax, scale: float = 0.3):
        T_w = np.eye(4, dtype=np.float32)
        Camera._draw_pose_axes(ax, T_w, scale)

    @staticmethod
    def _draw_pose_axes(ax, T: np.ndarray, scale: float = 0.1):
        """Draw RGB axes at pose T (world)."""
        o = T[:3, 3]
        R = T[:3, :3]
        axes = np.eye(3) * scale
        colors = ['r', 'g', 'b']
        for i in range(3):
            p = o + R @ axes[:, i]
            ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], colors[i], lw=2)


