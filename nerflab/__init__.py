"""
nerflab – lightweight helpers for NeRF and ray‑based experiments.

Import convenience re‑exports live here so users can simply do:
>>> from nerflab import Camera, Box, stratified_samples_batch
"""
from __future__ import annotations

# ------------------------------------------------------------------ #
# Core utility & math helpers
# ------------------------------------------------------------------ #
from .utils import homogenize, dehomogenize, invert_T, distance, look_at   # noqa: F401
from .camera import Intrinsics, Camera                                     # noqa: F401
from .geometry import Box, Sphere, World                                   # noqa: F401
from .sampling import stratified_samples_batch, cartesian_to_spherical     # noqa: F401
from .render import nerf_opacity, nerf_opacity_single, sigma_from_world    # noqa: F401

# ------------------------------------------------------------------ #
# Global configuration dataclasses
# ------------------------------------------------------------------ #
from .config import CFG, IntrinsicsCfg, RaySampleCfg              # noqa: F401

# ------------------------------------------------------------------ #
# Visualisation façade  (lazy‑loaded via nerflab.viz.__getattr__)
# ------------------------------------------------------------------ #
from .viz import (
    plot_world,
    viz_sigma_scatter,
    viz_sigma_heatmap,
    viz_cfg as VizCFG,         # expose viz‑specific defaults separately
)

# ------------------------------------------------------------------ #
# Ready‑made demo / preset helpers
# ------------------------------------------------------------------ #
from .presets import make_pose_cases, pose_H                               # noqa: F401

# ------------------------------------------------------------------ #
# Back‑compat shim: keep old  `nerflab.vis` import path working
# ------------------------------------------------------------------ #
import sys as _sys, importlib as _imp
_vis_mod = _imp.import_module("nerflab.viz")
_sys.modules["nerflab.vis"] = _vis_mod
_sys.modules["nerflab.vis.plotting"] = _vis_mod  # earlier hard‑coded path

# ------------------------------------------------------------------ #
# Public API symbol list
# ------------------------------------------------------------------ #
__all__ = [
    # utils
    "homogenize", "dehomogenize", "invert_T", "distance", "look_at",
    # core classes / functions
    "Intrinsics", "Camera",
    "Box", "Sphere", "World",
    "stratified_samples_batch", "cartesian_to_spherical",
    "nerf_opacity", "nerf_opacity_single", "sigma_from_world",
    # visualisation
    "plot_world", "viz_sigma_scatter", "viz_sigma_heatmap",
    # configs
    "CFG", "VizCFG", "IntrinsicsCfg", "RaySampleCfg",
    # presets
    "make_pose_cases", "pose_H",
]
