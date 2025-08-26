"""
World subpackage: simple analytic geometry and rendering helpers (no plotting).
"""

from .geometry import Box, Sphere
from .world_json import save_world, load_world

__all__ = [
    "Box", "Sphere",
    "save_world", "load_world", "query_density_field"
]
