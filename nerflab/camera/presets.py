# nerflab/presets.py
"""
Camera-pose presets: axis-aligned and diagonal 'look-at-origin' views
====================================================================

Example
-------
>>> from nerflab.presets import make_pose_cases, pose_H
>>> cases = make_pose_cases(axis_dist=5.0)
>>> H = pose_H(cases["+X"])        # 4×4 H_wc as a torch.Tensor
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch

# Re-use your existing helper. Keep this local (not package-level) to avoid
# import cycles: do *not* import from nerflab.__init__.
from .transforms import look_at

Vec3 = Tuple[float, float, float]
PoseCase = Dict[str, Vec3]

__all__ = ["make_pose_cases", "pose_H"]


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
    cases["+X"] = dict(eye=( axis_dist, 0.0,       0.0), target=(0.0, 0.0, 0.0), up=up_y)
    cases["-X"] = dict(eye=(-axis_dist, 0.0,       0.0), target=(0.0, 0.0, 0.0), up=up_y)

    cases["+Y"] = dict(eye=(0.0,  axis_dist, 0.0),      target=(0.0, 0.0, 0.0), up=up_z)
    cases["-Y"] = dict(eye=(0.0, -axis_dist, 0.0),      target=(0.0, 0.0, 0.0), up=up_z)

    cases["+Z"] = dict(eye=(0.0, 0.0,  axis_dist),      target=(0.0, 0.0, 0.0), up=up_y)
    cases["-Z"] = dict(eye=(0.0, 0.0, -axis_dist),      target=(0.0, 0.0, 0.0), up=up_y)

    # --- cube-corner diagonals ---------------------------------------------
    for sx in ( axis_dist, -axis_dist):
        for sy in ( axis_dist, -axis_dist):
            for sz in ( axis_dist, -axis_dist):
                # name like 'corner_ppp', 'corner_pnp', ...
                name = "corner_" + ("p" if sx > 0 else "n") + ("p" if sy > 0 else "n") + ("p" if sz > 0 else "n")
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
    if dtype in (np.int64,  np.dtype("int64"), int):
        return torch.int64
    if dtype in (np.int32,  np.dtype("int32")):
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
