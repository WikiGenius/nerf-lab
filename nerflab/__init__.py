"""
nerflab: minimal, modular tools for cameras, worlds, NeRF-style ops, and visualization.
Public API stays compact; internals live under subpackages.
"""

# ---- Version (optionally sync with pyproject.toml) ----
__version__ = "0.1.0"

# ---- Camera exports ----
from .camera.camera import Camera
from .camera.intrinsics import Intrinsics
from . import camera as camera  # expose submodule (e.g., camera.presets)

# ---- World exports ----
from .world.geometry import Box, Sphere
from .world.render import save_world, load_world

# ---- Learning (Torch ops) ----
from .learning.forward_sigma import nerf_opacity, compute_opacity_simple

# ---- Config ----
from .config.config import CFG, Cfg, IntrinsicsCfg, RaySampleCfg

__all__ = [
    # version
    "__version__",
    # camera
    "Camera", "Intrinsics", "fx_from_fov", "fy_from_fov", "camera",
    # world
    "Box", "Sphere", "save_world", "load_world"
    # learning
    "nerf_opacity",
    # config
    "CFG", "Cfg", "IntrinsicsCfg", "RaySampleCfg",
]
