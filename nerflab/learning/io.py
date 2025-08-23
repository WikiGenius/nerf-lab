# nerflab/learning/io.py
from __future__ import annotations

import torch
from ..world.world_json import load_world

@torch.no_grad()
def query_density_field(
    pts: torch.Tensor,
    path_world: str = "../data/world.json",
) -> torch.Tensor:
    """
    Vectorized density query.

    Expects World.density(xyz) with xyz of shape (M, 3) and returns (M,)
    (or broadcastable to it). Exactly ONE call into world.density.

    Args:
        pts: (R, N, 3) tensor of sample positions.

    Returns:
        (R, N) tensor on same device/dtype as `pts`.
    """
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"`pts` must be (R, N, 3); got {tuple(pts.shape)}")

    R, N, _ = pts.shape
    device, dtype = pts.device, pts.dtype

    world = load_world(path_world)
    xyz = pts.reshape(-1, 3)                      # (R*N, 3)
    dens = world.density(xyz)                     # (R*N,)

    if not isinstance(dens, torch.Tensor):
        dens = torch.as_tensor(dens, device=device, dtype=dtype)
    dens = dens.to(device=device, dtype=dtype)
    if dens.ndim == 2 and dens.shape[1] == 1:
        dens = dens.squeeze(1)
    if dens.ndim != 1 or dens.numel() != xyz.shape[0]:
        raise RuntimeError("World.density must return shape (R*N,) or broadcastable to it.")

    return dens.view(R, N)
