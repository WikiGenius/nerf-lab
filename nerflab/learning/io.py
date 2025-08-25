# nerflab/learning/io.py
from __future__ import annotations
import torch
from ..world.world_json import load_world

@torch.no_grad()
def query_density_field(
    pts: torch.Tensor,
    path_world: str = "../data/world.json",
    *,
    world=None,
) -> torch.Tensor:
    """
    Vectorized density query for batched or unbatched points.

    Accepts:
        - pts: (R, N, 3)
        - pts: (B, R, N, 3)

    Returns:
        - (R, N)        for input (R, N, 3)
        - (B, R, N)     for input (B, R, N, 3)

    Notes:
        - Exactly ONE call into `world.density(xyz)` with xyz flattened to (M, 3).
        - Preserves device/dtype of `pts`.
        - You can pass a preloaded `world` to avoid re-loading per call.
    """
    if pts.ndim not in (3, 4) or pts.shape[-1] != 3:
        raise ValueError(f"`pts` must be (R, N, 3) or (B, R, N, 3); got {tuple(pts.shape)}")

    device, dtype = pts.device, pts.dtype
    leading = pts.shape[:-1]           # (..., 3) -> ... = (R, N) or (B, R, N)
    M = int(torch.prod(torch.tensor(leading)))  # total sample count

    if world is None:
        world = load_world(path_world)

    # Flatten to (M, 3) without unnecessary copies.
    xyz = pts.reshape(-1, 3)  # works for both contiguous and non-contiguous

    dens = world.density(xyz)  # expect (M,) or broadcastable to (M,)

    # Normalize to a 1-D tensor of length M on the right device/dtype.
    if not isinstance(dens, torch.Tensor):
        dens = torch.as_tensor(dens, device=device, dtype=dtype)
    else:
        dens = dens.to(device=device, dtype=dtype)

    # Squeeze common (M,1) returns, but keep shape checks strict.
    if dens.ndim == 2 and dens.shape[1] == 1:
        dens = dens.squeeze(1)

    # Allow broadcasting (e.g., scalar) but validate final size.
    try:
        dens = dens.expand(M) if dens.ndim == 0 else dens
    except Exception:
        pass  # if expand fails, we’ll check size below

    if dens.ndim != 1 or dens.numel() != M:
        # final attempt: try to broadcast explicitly
        dens = torch.as_tensor(dens, device=device, dtype=dtype).reshape(-1)
        if dens.numel() not in (1, M):
            raise RuntimeError(
                f"World.density must return shape (M,) or broadcastable to it. "
                f"Got {tuple(dens.shape)} for M={M}."
            )
        if dens.numel() == 1:
            dens = dens.expand(M)

    return dens.view(*leading)
