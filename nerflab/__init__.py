"""
nerflab: minimal, modular tools for cameras, worlds, NeRF-style ops, and visualization.
Public API stays compact; internals live under subpackages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple
import importlib

# ---- Version (tries to sync with project metadata; falls back) ----
try:
    from importlib.metadata import version as _pkg_version  # py3.8+
    __version__ = _pkg_version("nerflab")
except Exception:
    __version__ = "0.1.0"

# -----------------------------------------------------------------------------
# Public API surface (lazy)
# Map public names -> (module_path, attribute_name)
# This keeps import time low while giving a single, curated entry point.
# -----------------------------------------------------------------------------
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # camera
    "Camera": ("nerflab.camera.camera", "Camera"),
    "Intrinsics": ("nerflab.camera.intrinsics", "Intrinsics"),
    "fx_from_fov": ("nerflab.camera.intrinsics", "fx_from_fov"),
    "fy_from_fov": ("nerflab.camera.intrinsics", "fy_from_fov"),
    "pose_H": ("nerflab.camera.presets", "pose_H"),
    "make_pose_cases": ("nerflab.camera.presets", "make_pose_cases"),
    "stratified_samples_batch": ("nerflab.camera.sampling", "stratified_samples_batch"),
    "cartesian_to_spherical": ("nerflab.camera.sampling", "cartesian_to_spherical"),
    "look_at": ("nerflab.camera.transforms", "look_at"),
    "distance": ("nerflab.camera.transforms", "distance"),

    # world
    "World": ("nerflab.world.geometry", "World"),
    "Box": ("nerflab.world.geometry", "Box"),
    "Sphere": ("nerflab.world.geometry", "Sphere"),
    "save_world": ("nerflab.world.world_json", "save_world"),
    "load_world": ("nerflab.world.world_json", "load_world"),
    "query_density_field": ("nerflab.world.world_json", "query_density_field"),
    
    # viz
    "plot_world": ("nerflab.viz.viz_world", "plot_world"),
    "viz_sigma_heatmap": ("nerflab.viz.viz_sigma", "viz_sigma_heatmap"),
    "viz_sigma_scatter": ("nerflab.viz.viz_sigma", "viz_sigma_scatter"),

    # learning
    "nerf_opacity": ("nerflab.learning.forward_sigma", "nerf_opacity"),
    "compute_opacity_simple": ("nerflab.learning.forward_sigma", "compute_opacity_simple"),

    # config
    "CFG": ("nerflab.config.config", "CFG"),
    "Cfg": ("nerflab.config.config", "Cfg"),
    "IntrinsicsCfg": ("nerflab.config.config", "IntrinsicsCfg"),
    "RaySampleCfg": ("nerflab.config.config", "RaySampleCfg"),
}

# Explicit submodule exposure (import-on-access)
_SUBMODULES = {
    "camera": "nerflab.camera",
    "world": "nerflab.world",
    "viz": "nerflab.viz",
    "learning": "nerflab.learning",
    "config": "nerflab.config",
}

# What `from nerflab import *` provides
__all__ = [
    "__version__",
    # camera
    "Camera", "Intrinsics", "fx_from_fov", "fy_from_fov", "pose_H", "make_pose_cases",
    "stratified_samples_batch", "cartesian_to_spherical", "look_at", "distance",
    "camera",
    # world
    "World", "Box", "Sphere", "save_world", "load_world", "world",
    # viz
    "plot_world", "viz_sigma_heatmap", "viz_sigma_scatter", "viz",
    # learning
    "nerf_opacity", "compute_opacity_simple", "learning",
    # config
    "CFG", "Cfg", "IntrinsicsCfg", "RaySampleCfg", "config",
]

def __getattr__(name: str):
    """Lazy attribute loader for the public API and submodules."""
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value  # cache for future accesses
        return value
    if name in _SUBMODULES:
        module = importlib.import_module(_SUBMODULES[name])
        globals()[name] = module  # cache
        return module
    raise AttributeError(f"module 'nerflab' has no attribute '{name}'")

def __dir__():
    return sorted(set(globals().keys()) | set(__all__))

# Help IDEs/type-checkers resolve symbols without eager imports
if TYPE_CHECKING:
    from nerflab.camera.camera import Camera  # type: ignore
    from nerflab.camera.intrinsics import Intrinsics, fx_from_fov, fy_from_fov  # type: ignore
    from nerflab.camera.presets import pose_H, make_pose_cases  # type: ignore
    from nerflab.camera.sampling import stratified_samples_batch, cartesian_to_spherical  # type: ignore
    from nerflab.camera.transforms import look_at, distance  # type: ignore

    from nerflab.world.geometry import World, Box, Sphere  # type: ignore
    from nerflab.world.world_json import save_world, load_world  # type: ignore

    from nerflab.viz.viz_world import plot_world  # type: ignore
    from nerflab.viz.viz_sigma import viz_sigma_heatmap, viz_sigma_scatter  # type: ignore

    from nerflab.learning.forward_sigma import nerf_opacity, compute_opacity_simple  # type: ignore

    from nerflab.config.config import CFG, Cfg, IntrinsicsCfg, RaySampleCfg  # type: ignore

    import nerflab.camera as camera  # type: ignore
    import nerflab.world as world  # type: ignore
    import nerflab.viz as viz  # type: ignore
    import nerflab.learning as learning  # type: ignore
    import nerflab.config as config  # type: ignore
