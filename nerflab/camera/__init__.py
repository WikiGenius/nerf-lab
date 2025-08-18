"""
Camera subpackage: camera model, intrinsics/FOV helpers, sampling, and SE(3) transforms.
Keep Torch out of here unless absolutely necessary.
"""

from .camera import Camera
from .intrinsics import Intrinsics
from .sampling import stratified_samples_batch  # framework-agnostic if possible
from .transforms import invert_T, homogenize, dehomogenize
from . import presets as presets  # allow: camera.presets.make_pose_cases, etc.

__all__ = [
    "Camera",
    "Intrinsics", "fx_from_fov", "fy_from_fov",
    "stratified_samples_batch",
    "invert_T", "homogenize", "dehomogenize",
    "presets",
]
