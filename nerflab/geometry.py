# nerflab/geometry.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np

# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Shape(ABC):
    """Abstract geometric primitive."""

    @abstractmethod
    def density(self, x: float, y: float, z: float) -> float:
        """Return +∞ inside the object, 0 outside (NeRF style)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Concrete primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box(Shape):
    center: Sequence[float]  # (cx, cy, cz)
    size:   Sequence[float]  # (width, height, depth)

    def __post_init__(self):
        cx, cy, cz = self.center
        w,  h,  d  = self.size
        object.__setattr__(self, "bounds", (
            (cx - w / 2, cx + w / 2),
            (cy - h / 2, cy + h / 2),
            (cz - d / 2, cz + d / 2),
        ))
    
    # Public API
    # --------------------------------------------------------------------- #
    def density(self, x: float, y: float, z: float) -> float:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        inside = xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax
        return float("inf") if inside else 0.0


@dataclass(frozen=True)
class Sphere(Shape):
    center: Sequence[float]  # (cx, cy, cz)
    radius: float

    # --------------------------------------------------------------------- #
    def density(self, x: float, y: float, z: float) -> float:
        cx, cy, cz = self.center
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius ** 2:
            return float("inf")
        return 0.0


# --------------------------------------------------------------------------- #
# Scene container
# --------------------------------------------------------------------------- #
class World:
    """Collection of Shape objects with an aggregated density query."""

    def __init__(self):
        self.shapes: list[Shape] = []

    # fluent interface – allows chaining: world.add_shape(...).add_shape(...)
    def add_shape(self, shape: Shape) -> "World":
        self.shapes.append(shape)
        return self

    def density(self, x: float, y: float, z: float) -> float:
        """Return +∞ if any shape occupies (x,y,z), else 0."""
        return float("inf") if any(
            np.isinf(s.density(x, y, z)) for s in self.shapes
        ) else 0.0
