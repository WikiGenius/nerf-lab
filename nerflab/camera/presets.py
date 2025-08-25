# nerflab/camera/presets.py
"""
Camera pose presets and spherical distributions (optimized, batch-friendly)

Example
-------
>>> from nerflab.camera.presets import make_pose_cases, pose_H, make_spherical_poses
>>> cases = make_pose_cases(axis_dist=5.0)
>>> H = pose_H(cases["+X"])                 # (4,4) torch.Tensor

# Batched slice of a large sweep (e.g., stream in chunks of 20):
>>> H_b = make_spherical_poses(c=1000, batch_size=20, offset=0, method="fibonacci")
>>> H_b.shape
torch.Size([20, 4, 4])
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Union, Literal
from math import pi
import numpy as np
import torch

# Keep this import to avoid API breakage; we’ll *prefer* our batched impl below.
from .transforms import look_at as _look_at_single

# New utilities (local module)
from .pose_utils import (
    fibonacci_points,
    ring_points,
    look_at_batch,  # robust + fast batched look_at
)

Vec3 = Tuple[float, float, float]
PoseCase = Dict[str, Vec3]


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _to_torch_dtype(dtype: Optional[Union[torch.dtype, np.dtype, type]]) -> torch.dtype:
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (np.float32, np.dtype("float32"), float):
        return torch.float32
    if dtype in (np.float64, np.dtype("float64")):
        return torch.float64
    if dtype in (np.int64, np.dtype("int64"), int):
        return torch.int64
    if dtype in (np.int32, np.dtype("int32")):
        return torch.int32
    return torch.float32


def _to_tensor3(x, *, dtype, device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t.reshape(3)


def _deg2rad(a: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(a * pi / 180.0, dtype=dtype, device=device)


# ----------------------------------------------------------------------------- #
# Axis-aligned and diagonal presets (unchanged API, minor cleanups)
# ----------------------------------------------------------------------------- #
def make_pose_cases(
    *,
    axis_dist: float = 5.0,
    up_y: Vec3 = (0.0, 1.0, 0.0),
    up_z: Vec3 = (0.0, 0.0, 1.0),
) -> Dict[str, PoseCase]:
    """Build human-readable axis and cube-corner look-at cases."""
    cases: Dict[str, PoseCase] = {}

    # Axis-aligned
    cases["+X"] = dict(eye=(axis_dist, 0.0, 0.0), target=(0.0, 0.0, 0.0), up=up_y)
    cases["-X"] = dict(eye=(-axis_dist, 0.0, 0.0), target=(0.0, 0.0, 0.0), up=up_y)

    cases["+Y"] = dict(eye=(0.0, axis_dist, 0.0), target=(0.0, 0.0, 0.0), up=up_z)
    cases["-Y"] = dict(eye=(0.0, -axis_dist, 0.0), target=(0.0, 0.0, 0.0), up=up_z)

    cases["+Z"] = dict(eye=(0.0, 0.0, axis_dist), target=(0.0, 0.0, 0.0), up=up_y)
    cases["-Z"] = dict(eye=(0.0, 0.0, -axis_dist), target=(0.0, 0.0, 0.0), up=up_y)

    # Cube corners
    for sx in (axis_dist, -axis_dist):
        for sy in (axis_dist, -axis_dist):
            for sz in (axis_dist, -axis_dist):
                name = "corner_" + ("p" if sx > 0 else "n") + ("p" if sy > 0 else "n") + ("p" if sz > 0 else "n")
                up = up_y if abs(sy) < 0.99 else up_z
                cases[name] = dict(eye=(sx, sy, sz), target=(0.0, 0.0, 0.0), up=up)
    return cases


def pose_H(
    case: PoseCase,
    *,
    forward_is_neg_z: bool = True,
    dtype: Optional[Union[torch.dtype, np.dtype, type]] = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert one case dict to a 4×4 H_wc."""
    tdtype = _to_torch_dtype(dtype)
    device = device if device is not None else torch.device("cpu")
    H = _look_at_single(
        eye=case["eye"],
        target=case["target"],
        up=case["up"],
        forward_is_neg_z=forward_is_neg_z,
        dtype=tdtype,
        device=device,
    )
    return torch.as_tensor(H, dtype=tdtype, device=device).reshape(4, 4)


# ----------------------------------------------------------------------------- #
# Spherical pose generation (optimized, vectorized, batch-friendly)
# ----------------------------------------------------------------------------- #
def make_spherical_poses(
    *,
    c: int,
    radius: float = 5.0,
    center: Vec3 = (0.0, 0.0, 0.0),
    method: Literal["fibonacci", "ring"] = "fibonacci",
    # Limit latitude to avoid poles for fibonacci method
    lat_range_deg: Tuple[float, float] = (-80.0, 80.0),
    # Ring-only: choose elevation in degrees (0 = equator)
    ring_elev_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
    forward_is_neg_z: bool = True,
    dtype: Optional[Union[torch.dtype, np.dtype, type]] = torch.float32,
    device: Optional[torch.device] = None,
    # NEW: generate only a chunk [offset : offset+batch_size)
    offset: int = 0,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate spherical camera-to-world transforms H_wc.

    - Vectorized: no Python loop when building H.
    - Batch-friendly: use `offset` and `batch_size` to create slices directly.
      Useful when C is large (e.g., 1000) but you render/process in micro-batches (e.g., 20).

    Parameters
    ----------
    c : int
        Total number of cameras in the conceptual sweep.
    radius : float
        Sphere radius (>0).
    center : (3,)
        World-space target point (all cameras look at this).
    method : {"fibonacci","ring"}
        Distribution method.
    lat_range_deg : (lo, hi)
        Latitude band for "fibonacci" (deg).
    ring_elev_deg : float
        Elevation for "ring" (deg).
    yaw_offset_deg : float
        Global yaw spin (deg).
    forward_is_neg_z : bool
        OpenGL/NeRF convention if True.
    dtype, device : torch dtype/device
    offset : int
        Start index (0-based) of the slice to generate.
    batch_size : Optional[int]
        If provided, generate at most this many poses from `offset`. If None, generate the full set.

    Returns
    -------
    H : (n, 4, 4) torch.Tensor
        Where n = batch_size if given else c.
    """
    if not isinstance(c, int) or c < 1:
        raise ValueError(f"`c` must be a positive int, got {c!r}.")
    if not (isinstance(radius, (int, float)) and radius > 0):
        raise ValueError(f"`radius` must be > 0, got {radius!r}.")
    if method not in ("fibonacci", "ring"):
        raise ValueError("`method` must be 'fibonacci' or 'ring'.")
    if offset < 0 or offset >= c:
        raise ValueError(f"`offset` must be in [0, {c-1}], got {offset}.")

    tdtype = _to_torch_dtype(dtype)
    device = device if device is not None else torch.device("cpu")
    center_t = _to_tensor3(center, dtype=tdtype, device=device)
    yaw_offset = _deg2rad(yaw_offset_deg, dtype=tdtype, device=device)

    # Slice length
    n = c - offset if batch_size is None else min(batch_size, max(c - offset, 0))

    # Points on the sphere (vectorized for the slice only)
    if method == "fibonacci":
        pts = fibonacci_points(
            c_total=c,
            start=offset,
            count=n,
            lat_range_deg=lat_range_deg,
            yaw_offset=yaw_offset,
            radius=radius,
            center=center_t,
            dtype=tdtype,
            device=device,
        )
    else:
        pts = ring_points(
            c_total=c,
            start=offset,
            count=n,
            ring_elev_deg=ring_elev_deg,
            yaw_offset=yaw_offset,
            radius=radius,
            center=center_t,
            dtype=tdtype,
            device=device,
        )

    # Build H_wc with a **batched** look_at (fast + robust)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=tdtype, device=device).expand_as(pts)
    H = look_at_batch(
        eye=pts, target=center_t.expand_as(pts), up=up, forward_is_neg_z=forward_is_neg_z
    )
    return H


__all__ = ["make_pose_cases", "pose_H", "make_spherical_poses"]
