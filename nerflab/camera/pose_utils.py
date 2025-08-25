# nerflab/camera/pose_utils.py
from __future__ import annotations
from typing import Tuple
import math
import torch

# ------------------------------ Angles ------------------------------------- #
def _deg2rad(a: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(a * math.pi / 180.0, dtype=dtype, device=device)


# ------------------------------ Distributions ------------------------------ #
@torch.no_grad()
def fibonacci_points(
    *,
    c_total: int,
    start: int,
    count: int,
    lat_range_deg: Tuple[float, float],
    yaw_offset: torch.Tensor,   # radians tensor
    radius: float,
    center: torch.Tensor,       # (3,)
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Vectorized Fibonacci-sphere slice: indices in [start, start+count).
    Returns (count, 3) points in world coordinates.
    """
    # Clamp/swap lat range
    lat_lo_deg, lat_hi_deg = lat_range_deg
    lat_lo_deg = max(-90.0, min(90.0, float(lat_lo_deg)))
    lat_hi_deg = max(-90.0, min(90.0, float(lat_hi_deg)))
    if lat_lo_deg > lat_hi_deg:
        lat_lo_deg, lat_hi_deg = lat_hi_deg, lat_lo_deg

    # Convert band to y in [-1,1] via sin(latitude)
    y_min = torch.sin(_deg2rad(lat_lo_deg, dtype=dtype, device=device))
    y_max = torch.sin(_deg2rad(lat_hi_deg, dtype=dtype, device=device))

    # Indices for this slice
    i = torch.arange(start, start + count, dtype=dtype, device=device)  # float for math below
    # Base y in (-1, 1), avoiding exact poles
    y = 1.0 - (2.0 * (i + 0.5) / float(c_total))
    # Map into [y_min, y_max]
    y = y_min + (y + 1.0) * 0.5 * (y_max - y_min)
    r_xy = torch.clamp(1.0 - y * y, min=0.0).sqrt()

    # Golden angle
    phi = math.pi * (3.0 - math.sqrt(5.0))
    theta = i * phi + yaw_offset  # stable phase

    x = r_xy * torch.cos(theta)
    z = r_xy * torch.sin(theta)

    pts = torch.stack((x, y, z), dim=-1) * float(radius) + center  # (count,3)
    return pts


@torch.no_grad()
def ring_points(
    *,
    c_total: int,
    start: int,
    count: int,
    ring_elev_deg: float,
    yaw_offset: torch.Tensor,   # radians tensor
    radius: float,
    center: torch.Tensor,       # (3,)
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Vectorized single-latitude ring slice. Returns (count,3).
    """
    elev = _deg2rad(ring_elev_deg, dtype=dtype, device=device)
    y = torch.sin(elev)
    r_xy = torch.clamp(1.0 - y * y, min=0.0).sqrt()

    k = torch.arange(start, start + count, dtype=dtype, device=device)
    theta = (2.0 * math.pi * k / float(c_total)) + yaw_offset

    x = r_xy * torch.cos(theta)
    z = r_xy * torch.sin(theta)
    yv = torch.full_like(x, y)

    pts = torch.stack((x, yv, z), dim=-1) * float(radius) + center
    return pts


# ------------------------------ Batched look_at ---------------------------- #
@torch.no_grad()
def look_at_batch(
    *,
    eye: torch.Tensor,       # (B,3)
    target: torch.Tensor,    # (B,3)
    up: torch.Tensor,        # (B,3) (per-camera up)
    forward_is_neg_z: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Robust batched look-at transform. Returns H_wc of shape (B,4,4).

    - Handles near-colinear up by re-orthogonalization.
    - Columns are [right, true_up, forward] (camera frame basis in world coords).
    - If `forward_is_neg_z` is True (OpenGL/NeRF), forward points from eye to target.
    """
    assert eye.shape == target.shape == up.shape and eye.shape[-1] == 3
    device, dtype = eye.device, eye.dtype
    B = eye.shape[0]

    fwd = (target - eye)
    fwd = fwd / (fwd.norm(dim=-1, keepdim=True) + eps)

    if forward_is_neg_z:
        fwd = -fwd  # flip to +Z-forward convention if desired

    # Remove the up component parallel to fwd to avoid degeneracy
    up = up / (up.norm(dim=-1, keepdim=True) + eps)
    up_proj = (up * fwd).sum(dim=-1, keepdim=True) * fwd
    up_ortho = up - up_proj
    up_ortho = up_ortho / (up_ortho.norm(dim=-1, keepdim=True) + eps)

    right = torch.linalg.cross(up_ortho, fwd)
    right = right / (right.norm(dim=-1, keepdim=True) + eps)

    true_up = torch.linalg.cross(fwd, right)  # already orthogonal

    R = torch.stack([right, true_up, fwd], dim=-1)  # (B,3,3)

    H = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).clone()
    H[:, :3, :3] = R
    H[:, :3, 3] = eye
    return H


# ------------------------------ Streaming API ------------------------------ #
@torch.no_grad()
def iter_spherical_pose_batches(
    *,
    c_total: int,
    batch_size: int,
    radius: float = 3.0,
    center=(0.0, 0.0, 0.0),
    method: str = "fibonacci",
    lat_range_deg: Tuple[float, float] = (-80.0, 80.0),
    ring_elev_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
    forward_is_neg_z: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    """
    Lazily yield (B,4,4) H_wc batches across a conceptual total of `c_total` poses.
    Never allocates the full (C,4,4).
    """
    device = torch.device(device)
    center_t = torch.as_tensor(center, dtype=dtype, device=device).reshape(3)
    yaw_offset = _deg2rad(yaw_offset_deg, dtype=dtype, device=device)

    for start in range(0, c_total, batch_size):
        count = min(batch_size, c_total - start)
        if method == "fibonacci":
            pts = fibonacci_points(
                c_total=c_total, start=start, count=count,
                lat_range_deg=lat_range_deg, yaw_offset=yaw_offset,
                radius=radius, center=center_t, dtype=dtype, device=device
            )
        else:
            pts = ring_points(
                c_total=c_total, start=start, count=count,
                ring_elev_deg=ring_elev_deg, yaw_offset=yaw_offset,
                radius=radius, center=center_t, dtype=dtype, device=device
            )
        up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(pts)
        H = look_at_batch(eye=pts, target=center_t.expand_as(pts), up=up, forward_is_neg_z=forward_is_neg_z)
        yield H
