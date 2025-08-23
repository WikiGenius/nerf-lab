# nerflab/camera/sampling.py
import torch
from typing import Tuple, Optional, Union

def stratified_samples_batch(
    origins: torch.Tensor,                         # (R, 3)
    dirs:    torch.Tensor,                         # (R, 3)
    *,
    t_near: Union[float, torch.Tensor],            # float or (R,)
    t_far:  Union[float, torch.Tensor],            # float or (R,)
    N:      int,                                   # samples per ray
    rng:    Optional[torch.Generator] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratified (or mid-bin) sampling along many rays, in PyTorch.

    Args:
        origins: (R,3) ray origins in world frame
        dirs:    (R,3) ray unit directions in world frame
        t_near:  float or (R,) near plane per ray
        t_far:   float or (R,) far  plane per ray
        N:       number of samples per ray
        rng:     optional torch.Generator for reproducibility
        deterministic: if True, sample mid-bin; else uniform within each bin

    Returns:
        t_vals: (R, N)    sample depths along each ray
        deltas: (R, N)    spacing to next sample (last = t_far − t_vals[:,-1])
        pts:    (R, N, 3) 3D points = origins[:,None,:] + t_vals[...,None]*dirs[:,None,:]
    """
    if origins.ndim != 2 or origins.shape[-1] != 3 or dirs.shape != origins.shape:
        raise ValueError(f"origins/dirs must be (R,3); got {origins.shape=}, {dirs.shape=}")
    if N <= 0:
        raise ValueError("N must be > 0")

    R, _ = origins.shape
    device = origins.device
    dtype  = origins.dtype

    # --- make t_near/t_far tensors of shape (R,) on the right device/dtype
    t_near = torch.as_tensor(t_near, device=device, dtype=dtype)
    t_far  = torch.as_tensor(t_far,  device=device, dtype=dtype)
    if t_near.ndim == 0: t_near = t_near.expand(R)
    if t_far.ndim  == 0: t_far  = t_far.expand(R)
    if t_near.shape != (R,) or t_far.shape != (R,):
        raise ValueError(f"t_near/t_far must broadcast to (R,); got {t_near.shape=}, {t_far.shape=}")

    # --- build per-ray bin positions u ∈ [0,1]
    if deterministic:
        # midpoints: (i + 0.5)/N
        u = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N
        u = u.unsqueeze(0).expand(R, N)                    # (R,N)
    else:
        # uniform within each bin: (i + rand)/N
        base = torch.arange(N, device=device, dtype=dtype).unsqueeze(0).expand(R, N)
        if rng is None:
            rand = torch.rand((R, N), device=device, dtype=dtype)
        else:
            rand = torch.rand((R, N), generator=rng, device=device, dtype=dtype)
        u = (base + rand) / N                              # (R,N)

    # --- map u to depths in [t_near, t_far]
    span   = (t_far - t_near).unsqueeze(1)                 # (R,1)
    t_vals = t_near.unsqueeze(1) + span * u                # (R,N)

    # --- compute deltas (spacing)
    deltas = torch.empty_like(t_vals)
    deltas[:, :-1] = t_vals[:, 1:] - t_vals[:, :-1]
    deltas[:, -1]  = t_far - t_vals[:, -1]

    # --- sample 3D points
    pts = origins.unsqueeze(1) + t_vals.unsqueeze(2) * dirs.unsqueeze(1)  # (R,N,3)

    return t_vals, deltas, pts



def cartesian_to_spherical(
    points: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert cartesian (...,3) → spherical (r, theta, phi)
      r     = ||(x,y,z)||  
      theta = arccos(z/r)      polar angle from +Z (0..π)  
      phi   = atan2(y, x)      azimuth from +X toward +Y (-π..π)  

    Args:
        points: (...,3) tensor
        eps:    small epsilon to avoid divide-by-zero

    Returns:
        r:     (...,) radii  
        theta: (...,) polar angles  
        phi:   (...,) azimuth angles
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    r = torch.linalg.norm(points, dim=-1)
    # clamp z / r into [-1,1]
    cos_theta = torch.clamp(z / (r + eps), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    phi   = torch.atan2(y, x)

    return r, theta, phi
