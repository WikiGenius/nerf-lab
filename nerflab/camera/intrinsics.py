# nerflab/camera.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import torch
import numpy as np



# ==========================================================================
# Intrinsics (torch-native, no CFG dependency)
# ==========================================================================
@dataclass(frozen=True)
class Intrinsics:
    """
    Minimal pinhole intrinsics. Principal point is centered by default.
    """
    fx: float
    fy: float
    width: int
    height: int

    cx = property(lambda self: self.width / 2.0)
    cy = property(lambda self: self.height / 2.0)

    def K(self, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return 3×3 intrinsics matrix as torch tensor."""
        return torch.tensor(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            device=device, dtype=dtype
        )

    @classmethod
    def from_K(cls, K: Union[torch.Tensor, np.ndarray], width: int, height: int) -> "Intrinsics":
        """Build from a 3×3 K (torch or numpy)."""
        Kt = torch.as_tensor(K, dtype=torch.float32)
        fx, fy = float(Kt[0, 0]), float(Kt[1, 1])
        return cls(fx=fx, fy=fy, width=width, height=height)


__all__ = ["Intrinsics"]
