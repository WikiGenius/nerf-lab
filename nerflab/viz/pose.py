# nerflab/viz/pose.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union, List
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

NDArray = np.ndarray
PoseLike = Union[torch.Tensor, NDArray]          # (4,4) or (B,4,4)
PoseList = Union[PoseLike, Sequence[PoseLike]]   # single or list/tuple of poses


def _to_numpy_pose_list(T: PoseList) -> List[NDArray]:
    """
    Normalize input poses to a list of (4,4) NumPy arrays on CPU.
    Supports:
      • torch.Tensor with shape (4,4) or (B,4,4)
      • np.ndarray  with shape (4,4) or (B,4,4)
      • list/tuple of individual poses
    """
    def _to_np(x: Union[torch.Tensor, NDArray]) -> NDArray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if isinstance(T, (list, tuple)):
        return [_to_np(p).reshape(4, 4) for p in T]

    Tnp = _to_np(T)
    if Tnp.ndim == 2:
        return [Tnp.reshape(4, 4)]
    if Tnp.ndim == 3:
        return [t.reshape(4, 4) for t in Tnp]
    raise ValueError(f"Pose array must be (4,4) or (B,4,4); got shape {Tnp.shape}")


def draw_pose_axes(
    ax,
    T: PoseList,
    *,
    scale: float = 0.1,
    colors: Sequence[str] = ("r", "g", "b"),
    linewidth: float = 2.0,
    alpha: float = 1.0,
    labels: bool = False,
    label_prefix: str = "",
) -> List:
    """
    Draw an RGB axis triad at one or more 4×4 poses.

    Args:
        ax: Matplotlib 3D axis.
        T:  Pose(s) as (4,4) or (B,4,4) torch/np arrays, or a list of (4,4).
        scale: Length of each axis vector in world units.
        colors: (x,y,z) axis colors. Default ('r','g','b').
        linewidth: Line width for the axes.
        alpha: Alpha for the axis lines.
        labels: If True, add tiny 'x','y','z' text labels.
        label_prefix: Optional prefix for labels (useful when drawing many poses).

    Returns:
        A list of added Matplotlib artists (Line3DCollection or Line2D/Text objects).
    """
    poses = _to_numpy_pose_list(T)
    artists: List = []

    # Efficient line batch via Line3DCollection when many poses:
    use_collection = len(poses) > 16

    if use_collection:
        # Build three collections (one per axis) to preserve RGB coloring.
        for axis_idx, col in enumerate(colors):
            segs = []
            for Tnp in poses:
                o = Tnp[:3, 3]
                R = Tnp[:3, :3]
                p = o + R @ (np.eye(3)[:, axis_idx] * scale)
                segs.append(np.stack([o, p], axis=0))
            lc = Line3DCollection(segs, colors=col, lw=linewidth, alpha=alpha)
            ax.add_collection3d(lc)
            artists.append(lc)
        if labels:
            for i, Tnp in enumerate(poses):
                o = Tnp[:3, 3]
                R = Tnp[:3, :3]
                for axis_idx, txt in enumerate(("x", "y", "z")):
                    p = o + R @ (np.eye(3)[:, axis_idx] * scale)
                    t = ax.text(p[0], p[1], p[2], f"{label_prefix}{txt}", color=colors[axis_idx])
                    artists.append(t)
        return artists

    # For small counts, simple per-pose plotting
    for i, Tnp in enumerate(poses):
        o = Tnp[:3, 3]
        R = Tnp[:3, :3]
        axes = np.eye(3) * scale
        for axis_idx, col in enumerate(colors):
            p = o + R @ axes[:, axis_idx]
            line, = ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]],
                            color=col, lw=linewidth, alpha=alpha)
            artists.append(line)
            if labels:
                t = ax.text(p[0], p[1], p[2], f"{label_prefix}{('x','y','z')[axis_idx]}",
                            color=col)
                artists.append(t)
    return artists
