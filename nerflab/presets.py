# nerflab/presets.py
"""
Camera‑pose presets: axis‑aligned and diagonal 'look‑at‑origin' views
====================================================================

Example
-------
>>> from nerflab.presets import make_pose_cases, pose_H
>>> cases = make_pose_cases(axis_dist=5.)
>>> H = pose_H(cases["+X"])            # 4×4 H_wc for +X view
"""

from __future__ import annotations
from typing import Sequence, Dict, Tuple
import numpy as np

from .utils import look_at            # re‑use your existing helper

Vec3 = Tuple[float, float, float]

def make_pose_cases(
    *,
    axis_dist: float = 5.0,
    up_y: Vec3 = (0, 1, 0),
    up_z: Vec3 = (0, 0, 1),
) -> Dict[str, Dict[str, Vec3]]:
    """
    Build a dict of pose descriptors (eye/target/up) covering:
    - ±X, ±Y, ±Z axis directions
    - all 8 cube corners (±axis_dist, ±axis_dist, ±axis_dist)

    Returns
    -------
    cases : dict[str, dict]
        Keyed by a human‑readable name (e.g. '+X', 'corner_+++').
        Each value has keys: 'eye', 'target', 'up'.
    """
    cases: Dict[str, Dict[str, Vec3]] = {}

    # --- axis‑aligned six views --------------------------------------------
    cases["+X"] = dict(eye=( axis_dist,  0.0,  0.0), target=(0, 0, 0), up=up_y)
    cases["-X"] = dict(eye=(-axis_dist,  0.0,  0.0), target=(0, 0, 0), up=up_y)

    cases["+Y"] = dict(eye=( 0.0,  axis_dist, 0.0), target=(0, 0, 0), up=up_z)
    cases["-Y"] = dict(eye=( 0.0, -axis_dist, 0.0), target=(0, 0, 0), up=up_z)

    cases["+Z"] = dict(eye=( 0.0,  0.0,  axis_dist), target=(0, 0, 0), up=up_y)
    cases["-Z"] = dict(eye=( 0.0,  0.0, -axis_dist), target=(0, 0, 0), up=up_y)

    # --- cube‑corner diagonals ---------------------------------------------
    for sx in ( axis_dist, -axis_dist):
        for sy in ( axis_dist, -axis_dist):
            for sz in ( axis_dist, -axis_dist):
                name = f"corner_{'p' if sx>0 else 'n'}{'p' if sy>0 else 'n'}{'p' if sz>0 else 'n'}"
                up = up_y if abs(sy) < 0.99 else up_z  # avoid colinearity
                cases[name] = dict(eye=(sx, sy, sz), target=(0, 0, 0), up=up)

    return cases


def pose_H(
    case: Dict[str, Vec3],
    *,
    forward_is_neg_z: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert a single pose‑case dict to a 4×4 camera‑to‑world matrix.

    Parameters
    ----------
    case : dict with 'eye'/'target'/'up'
    forward_is_neg_z : bool
        If True (default), the camera looks along -Z in its local frame
        (the NeRF / OpenGL convention).  Set False for +Z‑forward cameras.
    """
    return look_at(
        eye=case["eye"],
        target=case["target"],
        up=case["up"],
        forward_is_neg_z=forward_is_neg_z,
        dtype=dtype,
    )
