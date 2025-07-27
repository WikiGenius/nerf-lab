"""
nerflab – lightweight helpers for NeRF and ray‑based experiments.

Import convenience re‑exports live here so users can do:
>>> from nerflab import Camera, Box, stratified_samples_batch
"""
from .utils import homogenize, dehomogenize, invert_T, distance, look_at
from .camera import Intrinsics, Camera
from .geometry import Box, Sphere, World
from .sampling import stratified_samples_batch, cartesian_to_spherical
from .render import nerf_opacity, nerf_opacity_single, sigma_from_world
from .vis.plotting import plot_world, viz_sigma_scatter, viz_sigma_heatmap
from .config import CFG, IntrinsicsCfg, RaySampleCfg, PlotCfg
from .presets import make_pose_cases, pose_H

__all__ = [
    # utils
    "homogenize", "dehomogenize", "invert_T", "distance", "look_at",
    # core classes / fx
    "Intrinsics", "Camera",
    "Box", "Sphere", "World",
    "stratified_samples_batch", "cartesian_to_spherical",
    "nerf_opacity", "nerf_opacity_single", "sigma_from_world",
    # visualisation
    "plot_world", "viz_sigma_scatter", "viz_sigma_heatmap",
    # config
    "CFG", "IntrinsicsCfg", "RaySampleCfg", "PlotCfg",
    # presets
    "make_pose_cases", "pose_H", 
]
