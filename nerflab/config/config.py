# --------------------- config.py ---------------------
from dataclasses import dataclass
import math

def fov_to_focal_length(fov_deg: float, size_px: int) -> float:
    """
    Convert a field-of-view (degrees) and image size (pixels)
    into a focal length in pixels.
    
    Args:
        fov_deg: Field of view angle in **degrees**.
        size_px: Sensor dimension in pixels (width or height).
    Returns:
        f (float): focal length in pixels.
    """
    # convert degrees → radians
    fov_rad = math.radians(fov_deg)
    # apply formula
    return size_px / (2.0 * math.tan(fov_rad / 2.0))


# --------------------- config.py ---------------------
"""
Configuration module defining camera intrinsics and ray sampling parameters
for NeRF-style rendering or ray-based simulations.

Classes:
    IntrinsicsCfg: Camera intrinsic parameters.
    RaySampleCfg:  Sampling parameters for rays through the scene.
    Cfg:          Top-level container bundling all sub-configs.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class IntrinsicsCfg:
    """
    Camera intrinsic parameters.

    Attributes:
        fx (float):   Focal length in pixels along the x-axis.
        fy (float):   Focal length in pixels along the y-axis.
        width (int):  Image sensor width in pixels.
        height (int): Image sensor height in pixels.
        # cx (float | None): Optional principal point x-coordinate (pixels).
        # cy (float | None): Optional principal point y-coordinate (pixels).
    """
    width: int = 640                                                   # image width  (px)
    height: int = 480                                                  # image height (px)
    # fx: float = 1200                                                 
    fx: float = fov_to_focal_length(fov_deg = 30, size_px = width)     # horizontal focal length (px)
    # fy: float = 1200                                                 
    fy: float = fx * (height / width)                                  # vertical focal length (px)
    # cx: float | None = None                                          # principal point x-offset (px)
    # cy: float | None = None                                          # principal point y-offset (px)


@dataclass(frozen=True)
class RaySampleCfg:
    """
    Parameters controlling how rays are sampled through the volume.

    Attributes:
        t_near (float):        Distance from the ray origin to start sampling.
        t_far (float):         Distance from the ray origin to stop sampling.
        N (int):               Number of samples per ray.
        R (int):               Total number of rays (e.g., pixels) to process.
        deterministic (bool):  If True, use fixed sampling positions; 
                               if False, jitter samples for anti-aliasing.
    """
    t_near: float = 2.5        # start of ray sampling (scene units)
    t_far: float = 3.5         # end of ray sampling (scene units)
    # N: int = 20                # samples per ray
    N: int = 40                # samples per ray
    R: int = 4096              # total number of rays (e.g., batch size)
    deterministic: bool = False  # toggle deterministic vs. randomized sampling


@dataclass(frozen=True)
class Cfg:
    """
    Top-level configuration object aggregating sub-configurations.

    Attributes:
        intrinsics (IntrinsicsCfg): Camera intrinsic parameters.
        rays (RaySampleCfg):       Ray sampling parameters.
    """
    intrinsics: IntrinsicsCfg = IntrinsicsCfg()
    rays: RaySampleCfg = RaySampleCfg()


# Instantiate a global configuration object for easy import elsewhere
CFG = Cfg()
