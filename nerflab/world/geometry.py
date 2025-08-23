# nerflab/world/geometry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Union, Optional, Tuple

import torch

TensorLike = Union[torch.Tensor, Sequence[float]]


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Shape:
    """
    Abstract geometric primitive.

    Contract:
      - Implement `density(xyz)` which returns +inf inside the shape and 0 outside.
      - `xyz` can be (..., 3); the return must be broadcastable to xyz[..., 0].
    """

    def density(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Concrete primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box(Shape):
    """
    Axis-aligned box defined by center and size.

    Notes
    -----
    - `density(xyz)` returns +inf inside the box, 0 otherwise.
    - `bounds` provides ((xmin,xmax), (ymin,ymax), (zmin,zmax)) as plain floats
      on CPU for visualization utilities.
    """
    center: TensorLike
    size: TensorLike

    # cached bounds as contiguous tensors: (mins, maxs), each shape (3,)
    _bounds: Optional[tuple[torch.Tensor, torch.Tensor]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def _ensure_bounds(self, *, device=None, dtype=None) -> None:
        """Ensure internal (mins,maxs) tensors exist and match device/dtype."""
        if self._bounds is not None:
            mins, _maxs = self._bounds
            if ((device is None or mins.device == device) and
                (dtype  is None or mins.dtype  == dtype)):
                return

        c = torch.as_tensor(self.center, device=device, dtype=dtype)  # (3,)
        s = torch.as_tensor(self.size,   device=device, dtype=dtype)  # (3,)
        half = s / 2
        mins = c - half
        maxs = c + half
        object.__setattr__(self, "_bounds", (mins, maxs))

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Public bounds for viz: ((xmin,xmax), (ymin,ymax), (zmin,zmax)) as floats.
        Computed on CPU; independent of the cached tensors used by `density`.
        """
        c = torch.as_tensor(self.center, device="cpu", dtype=torch.float32)
        s = torch.as_tensor(self.size,   device="cpu", dtype=torch.float32)
        half = s / 2
        mins = c - half
        maxs = c + half
        return (
            (float(mins[0]), float(maxs[0])),
            (float(mins[1]), float(maxs[1])),
            (float(mins[2]), float(maxs[2])),
        )

    def density(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : (..., 3) tensor

        Returns
        -------
        dens : (...,) tensor
            +inf inside, 0 outside (keeps input device & dtype).
        """
        assert xyz.shape[-1] == 3, "xyz must have last dim = 3"
        self._ensure_bounds(device=xyz.device, dtype=xyz.dtype)
        mins, maxs = self._bounds  # type: ignore[assignment]

        # Vectorized comparisons over the last coord dim; reduce with all()
        inside = ((xyz >= mins) & (xyz <= maxs)).all(dim=-1)

        # Allocate once and fill; avoids building two full tensors via where()
        dens = torch.zeros(xyz.shape[:-1], device=xyz.device, dtype=xyz.dtype)
        dens[inside] = float("inf")
        return dens


@dataclass(frozen=True)
class Sphere(Shape):
    """
    Solid sphere defined by center and radius.

    Notes
    -----
    Returns +inf inside the sphere, 0 otherwise.
    """
    center: TensorLike
    radius: Union[float, torch.Tensor]

    def density(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : (..., 3) tensor

        Returns
        -------
        dens : (...,) tensor
            +inf inside, 0 outside
        """
        assert xyz.shape[-1] == 3, "xyz must have last dim = 3"
        c = torch.as_tensor(self.center, device=xyz.device, dtype=xyz.dtype)  # (3,)
        r = torch.as_tensor(self.radius, device=xyz.device, dtype=xyz.dtype)  # ()
        diff = xyz - c
        inside = (diff * diff).sum(dim=-1) <= (r * r)

        dens = torch.zeros(xyz.shape[:-1], device=xyz.device, dtype=xyz.dtype)
        dens[inside] = float("inf")
        return dens


# --------------------------------------------------------------------------- #
# Scene container
# --------------------------------------------------------------------------- #
class World:
    """
    Collection of Shape objects with an aggregated density query.

    Notes
    -----
    - `density(xyz)` returns +inf where ANY shape contains the point, else 0.
    - Vectorized over arbitrary leading batch dims in `xyz`.
    """

    def __init__(self, shapes: Optional[Iterable[Shape]] = None) -> None:
        self.shapes: List[Shape] = list(shapes) if shapes is not None else []

    def add_shape(self, shape: Shape) -> "World":
        """Fluent API: world.add_shape(...).add_shape(...)"""
        self.shapes.append(shape)
        return self


    def density(
        self,
        xyz: torch.Tensor,
        *,
        chunk_points: int | None = None,
    ) -> torch.Tensor:
        """
        Vectorized occupancy OR across all shapes.

        Input
        -----
        xyz : (..., 3) tensor

        Output
        ------
        dens : (...,) tensor
            +inf if any shape occupies the point, else 0.
        """
        assert xyz.shape[-1] == 3, "xyz must have last dim = 3"

        out_shape = xyz.shape[:-1]
        device, dtype = xyz.device, xyz.dtype

        if not self.shapes:
            return torch.zeros(out_shape, device=device, dtype=dtype)

        def _occ_mask(points_2d: torch.Tensor) -> torch.Tensor:
            # Boolean occupancy (M,), early-exit when fully covered
            occ = torch.zeros(points_2d.shape[0], device=device, dtype=torch.bool)
            for s in self.shapes:
                # shapes return 0 outside / +inf inside → isinf gives mask
                occ |= torch.isinf(s.density(points_2d))
                if occ.all():
                    break
            return occ

        flat = xyz.reshape(-1, 3)

        if not chunk_points or chunk_points <= 0:
            occ = _occ_mask(flat)                       # (M,)
            dens = torch.zeros_like(occ, dtype=dtype)   # (M,)
            dens[occ] = float("inf")
            return dens.view(out_shape)

        # chunked path (same semantics, lower peak memory)
        M = flat.shape[0]
        dens = torch.zeros(M, device=device, dtype=dtype)
        for i in range(0, M, chunk_points):
            j = i + chunk_points
            occ = _occ_mask(flat[i:j])
            dens[i:j][occ] = float("inf")
        return dens.view(out_shape)



__all__ = ["Shape", "Box", "Sphere", "World"]
