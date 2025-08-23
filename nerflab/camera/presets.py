# nerflab/camera/presets.py
"""
Camera-pose presets: axis-aligned and diagonal 'look-at-origin' views
====================================================================

Example
-------
>>> from nerflab.camera.presets import make_pose_cases, pose_H
>>> cases = make_pose_cases(axis_dist=5.0)
>>> H = pose_H(cases["+X"])        # 4×4 H_wc as a torch.Tensor
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Union, Literal
from math import pi
import numpy as np
import torch

# Re-use your existing helper. Keep this local (not package-level) to avoid
# import cycles: do *not* import from nerflab.__init__.
from .transforms import look_at

Vec3 = Tuple[float, float, float]
PoseCase = Dict[str, Vec3]



def make_pose_cases(
    *,
    axis_dist: float = 5.0,
    up_y: Vec3 = (0.0, 1.0, 0.0),
    up_z: Vec3 = (0.0, 0.0, 1.0),
) -> Dict[str, PoseCase]:
    """
    Build a dictionary of pose descriptors (eye/target/up) covering:
      • ±X, ±Y, ±Z axis directions
      • all 8 cube corners at (±d, ±d, ±d)

    Parameters
    ----------
    axis_dist : float
        Distance of the camera eye from the origin (the "d" above).
    up_y, up_z : Vec3
        Alternative 'up' vectors to avoid colinearity when the view
        direction matches the default up.

    Returns
    -------
    cases : dict[str, dict]
        Keyed by human-readable names (e.g. '+X', 'corner_ppn').
        Each value has keys: 'eye', 'target', 'up'.
    """
    cases: Dict[str, PoseCase] = {}

    # --- axis-aligned six views --------------------------------------------
    cases["+X"] = dict(eye=(axis_dist, 0.0, 0.0), target=(0.0, 0.0, 0.0), up=up_y)
    cases["-X"] = dict(eye=(-axis_dist, 0.0, 0.0), target=(0.0, 0.0, 0.0), up=up_y)

    cases["+Y"] = dict(eye=(0.0, axis_dist, 0.0), target=(0.0, 0.0, 0.0), up=up_z)
    cases["-Y"] = dict(eye=(0.0, -axis_dist, 0.0), target=(0.0, 0.0, 0.0), up=up_z)

    cases["+Z"] = dict(eye=(0.0, 0.0, axis_dist), target=(0.0, 0.0, 0.0), up=up_y)
    cases["-Z"] = dict(eye=(0.0, 0.0, -axis_dist), target=(0.0, 0.0, 0.0), up=up_y)

    # --- cube-corner diagonals ---------------------------------------------
    for sx in (axis_dist, -axis_dist):
        for sy in (axis_dist, -axis_dist):
            for sz in (axis_dist, -axis_dist):
                # name like 'corner_ppp', 'corner_pnp', ...
                name = (
                    "corner_"
                    + ("p" if sx > 0 else "n")
                    + ("p" if sy > 0 else "n")
                    + ("p" if sz > 0 else "n")
                )
                # choose an 'up' that isn't parallel to the view direction
                up = up_y if abs(sy) < 0.99 else up_z
                cases[name] = dict(eye=(sx, sy, sz), target=(0.0, 0.0, 0.0), up=up)

    return cases


def _to_torch_dtype(dtype: Optional[Union[torch.dtype, np.dtype, type]]) -> torch.dtype:
    """Coerce common NumPy/Python dtypes to a valid torch.dtype."""
    if dtype is None:
        return torch.get_default_dtype()  # typically torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    # Map a few common numpy dtypes -> torch
    if dtype in (np.float32, np.dtype("float32"), float):
        return torch.float32
    if dtype in (np.float64, np.dtype("float64")):
        return torch.float64
    if dtype in (np.int64, np.dtype("int64"), int):
        return torch.int64
    if dtype in (np.int32, np.dtype("int32")):
        return torch.int32
    return torch.float32


def pose_H(
    case: PoseCase,
    *,
    forward_is_neg_z: bool = True,
    dtype: Optional[Union[torch.dtype, np.dtype, type]] = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a single pose-case dict to a 4×4 camera-to-world matrix H_wc (Torch).

    Parameters
    ----------
    case : dict
        Must contain 'eye', 'target', and 'up' (each a 3-tuple).
    forward_is_neg_z : bool
        If True (default), the camera looks along −Z in its local frame
        (OpenGL/NeRF convention). Set False for +Z-forward cameras.
    dtype : torch.dtype | numpy dtype | type, optional
        Desired tensor dtype (default: torch.float32).
    device : torch.device, optional
        Target device for the resulting tensor.

    Returns
    -------
    H_wc : (4,4) torch.Tensor
        Camera-to-world homogeneous transform.
    """
    tdtype = _to_torch_dtype(dtype)
    H_wc = look_at(
        eye=case["eye"],
        target=case["target"],
        up=case["up"],
        forward_is_neg_z=forward_is_neg_z,
        dtype=tdtype,
        device=device,
    )
    # Ensure (4,4), dtype/device as requested
    if not isinstance(H_wc, torch.Tensor):
        H_wc = torch.as_tensor(H_wc, dtype=tdtype, device=device)
    return H_wc.reshape(4, 4)


def _to_tensor3(x, *, dtype, device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t.reshape(3)


def make_spherical_poses(
    *,
    c: int,
    radius: float = 5.0,
    center: Vec3 = (0.0, 0.0, 0.0),
    method: Literal["fibonacci", "ring"] = "fibonacci",
    # Limit latitude to avoid poles: (-90, +90) degrees
    lat_range_deg: Tuple[float, float] = (-80.0, 80.0),
    # Ring-only: choose elevation for the ring in degrees (0 = equator)
    ring_elev_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,  # global spin around center (degrees)
    forward_is_neg_z: bool = True,  # NeRF/OpenGL convention
    dtype: Optional[Union[torch.dtype, np.dtype, type]] = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate `c` camera-to-world transforms H_wc on a sphere around `center`.

    All cameras look toward `center` via `look_at`. You can distribute the
    camera "eyes" either quasi-uniformly using a Fibonacci spiral or along a
    single-latitude ring (orbit shot).

    Parameters
    ----------
    c : int
        Number of camera poses to generate (must be >= 1).
    radius : float, default 5.0
        Sphere radius (must be > 0).
    center : (3,)
        World-space point the cameras look at (target).
    method : {"fibonacci", "ring"}, default "fibonacci"
        - "fibonacci": quasi-uniform points over latitude band.
        - "ring":      all points share the same latitude (orbit).
    lat_range_deg : (float, float), default (-80, 80)
        Inclusive latitude limits in degrees for "fibonacci" method.
        Useful to avoid numerical issues near the poles; values are clamped
        to [-90, 90], and swapped if given in reverse order.
        Ignored for "ring".
    ring_elev_deg : float, default 0.0
        Elevation (latitude) in degrees for the "ring" method only.
        0° = equator, +90° = north pole, -90° = south pole.
    yaw_offset_deg : float, default 0.0
        Global spin (yaw) applied to all poses, in degrees.
    forward_is_neg_z : bool, default True
        If True, camera's local -Z points toward the target (OpenGL/NeRF).
        Set False if your convention is +Z-forward.
    dtype : torch.dtype | numpy dtype | type, optional
        dtype for generated tensors (default: torch.float32).
    device : torch.device, optional
        Target device for the result (e.g., torch.device("cuda")).

    Returns
    -------
    H : (c, 4, 4) torch.Tensor
        Camera-to-world homogeneous transforms.

    Notes
    -----
    - Uses an up-vector of (0, 1, 0) for look_at. Avoiding exact poles
      (via `lat_range_deg`) reduces risk of up parallel to view direction.
    - The function does not require gradients; it’s designed for dataset
      generation and deterministic sweeps.

    Examples
    --------
    >>> H = make_spherical_poses(c=24, radius=4.0, method="fibonacci")
    >>> H.shape
    torch.Size([24, 4, 4])

    >>> H_ring = make_spherical_poses(c=36, radius=5.0, method="ring", ring_elev_deg=20)
    """
    # ---------------------------- Validation -------------------------------- #
    if not isinstance(c, int) or c < 1:
        raise ValueError(f"`c` must be a positive int, got {c!r}.")
    if not (isinstance(radius, (int, float)) and radius > 0):
        raise ValueError(f"`radius` must be > 0, got {radius!r}.")
    if method not in ("fibonacci", "ring"):
        raise ValueError("`method` must be 'fibonacci' or 'ring'.")

    # ----------------------- Dtype/Device normalization ---------------------- #
    tdtype = _to_torch_dtype(dtype)
    device = device if device is not None else torch.device("cpu")
    center_t = _to_tensor3(center, dtype=tdtype, device=device)

    # -------------------------- Angle utilities ----------------------------- #
    def deg2rad(a: float) -> torch.Tensor:
        return torch.tensor(a * pi / 180.0, dtype=tdtype, device=device)

    yaw_offset = deg2rad(yaw_offset_deg)

    # ---------------------- Latitude band (fibonacci) ------------------------ #
    # Clamp and order latitudes; convert to y = sin(lat) band.
    lat_lo_deg, lat_hi_deg = lat_range_deg
    lat_lo_deg = max(-90.0, min(90.0, float(lat_lo_deg)))
    lat_hi_deg = max(-90.0, min(90.0, float(lat_hi_deg)))
    if lat_lo_deg > lat_hi_deg:
        lat_lo_deg, lat_hi_deg = lat_hi_deg, lat_lo_deg

    y_min = torch.sin(deg2rad(lat_lo_deg))
    y_max = torch.sin(deg2rad(lat_hi_deg))

    # ----------------------- Generate unit-sphere points --------------------- #
    # We produce (x, y, z) on the unit sphere; later scale/shift by radius/center.
    if method == "fibonacci":
        # Golden-angle spiral points, roughly uniform in (y, theta).
        i = torch.arange(c, dtype=tdtype, device=device)  # [0..c-1]
        # Base y in (-1, 1), avoiding exact poles by +0.5 shift
        y = 1.0 - (2.0 * (i + 0.5) / float(c))
        # Linearly map y from (-1,1) into [y_min, y_max]
        y = y_min + (y + 1.0) * 0.5 * (y_max - y_min)
        r_xy = torch.clamp(1.0 - y * y, min=0.0).sqrt()

        # Golden angle (constant)
        phi = pi * (3.0 - torch.sqrt(torch.tensor(5.0, dtype=tdtype, device=device)))
        theta = i * phi + yaw_offset

        x = r_xy * torch.cos(theta)
        z = r_xy * torch.sin(theta)

    else:  # method == "ring"
        elev = deg2rad(ring_elev_deg)
        y = torch.sin(elev).expand(c)
        r_xy = torch.clamp(1.0 - y * y, min=0.0).sqrt()

        k = torch.arange(c, dtype=tdtype, device=device)
        theta = (2.0 * pi * k / float(c)) + yaw_offset

        x = r_xy * torch.cos(theta)
        z = r_xy * torch.sin(theta)

    # ------------------------ Scale/shift to sphere -------------------------- #
    pts = torch.stack((x, y, z), dim=-1) * radius + center_t  # (c, 3)

    # -------------------- Build H_wc via look-at for each -------------------- #
    # We keep this loop explicit because `look_at` may not support batching.
    up_nominal = torch.tensor([0.0, 1.0, 0.0], dtype=tdtype, device=device)
    H_list = []
    for p in pts:
        H = look_at(
            eye=p,
            target=center_t,
            up=up_nominal,
            forward_is_neg_z=forward_is_neg_z,
            dtype=tdtype,
            device=device,
        )
        # Ensure exactly (4,4) torch.Tensor
        H = torch.as_tensor(H, dtype=tdtype, device=device).reshape(4, 4)
        H_list.append(H)

    H = torch.stack(H_list, dim=0)  # (c, 4, 4)
    return H


__all__ = ["make_pose_cases", "pose_H", "make_spherical_poses"]
