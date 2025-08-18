import torch
from typing import Tuple, Optional, Union
from ..config.config import CFG


def stratified_samples_batch(
    origins: torch.Tensor,         # (R, 3)
    dirs:    torch.Tensor,         # (R, 3)
    rng:     Optional[torch.Generator] = None,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratified (or mid-bin) sampling along many rays, in PyTorch.

    Args:
        origins:      (R,3) ray origins in world frame
        dirs:         (R,3) ray unit directions in world frame
        rng:          optional torch.Generator for reproducibility
        deterministic: if True, sample at bin centers; else random in each bin

    Returns:
        t_vals: (R, N) sample depths along each ray  
        delta:  (R, N) intervals to next sample (last = t_far − t_vals[:,-1])  
        pts:    (R, N, 3) 3D points = origins[:,None,:] + t_vals[...,None]*dirs[:,None,:]
    """

    t_near = CFG.rays.t_near
    t_far = CFG.rays.t_far  # float
    N = CFG.rays.N  # number of samples per ray
    if N <= 0:
        raise ValueError("N must be > 0")

    R = origins.shape[0]
    device = origins.device
    dtype  = origins.dtype

    # make t_near, t_far into (R,) tensors
    t_near = torch.as_tensor(t_near, device=device, dtype=dtype)
    t_far  = torch.as_tensor(t_far,  device=device, dtype=dtype)
    if t_near.ndim == 0:
        t_near = t_near.expand(R)
    if t_far.ndim == 0:
        t_far = t_far.expand(R)

    # build uniform bins [0,1] of width 1/N
    if deterministic:
        # midpoints: (i + 0.5)/N
        u = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N
        u = u.unsqueeze(0).expand(R, N)
    else:
        # uniform within each bin: (i + rand)/N
        base = torch.arange(N, device=device, dtype=dtype).unsqueeze(0).expand(R, N)
        if rng is None:
            rand = torch.rand((R, N), device=device, dtype=dtype)
        else:
            rand = torch.rand((R, N), generator=rng, device=device, dtype=dtype)
        u = (base + rand) / N

    # map to [t_near, t_far]
    t_vals = t_near.unsqueeze(1) + (t_far - t_near).unsqueeze(1) * u  # (R,N)

    # compute deltas: diff plus final far interval
    # delta[:, :-1] = t_vals[:,1:] - t_vals[:,:-1]
    # delta[:, -1]  = t_far - t_vals[:,-1]
    deltas = torch.empty_like(t_vals)
    deltas[:, :-1] = t_vals[:, 1:] - t_vals[:, :-1]
    deltas[:, -1]  = t_far - t_vals[:, -1]

    # compute 3D points
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
