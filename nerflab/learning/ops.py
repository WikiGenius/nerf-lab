# nerflab/learning/ops.py
import numpy as np
import torch
from typing import Callable, Optional, Tuple
from ..camera.camera import Intrinsics, Camera                           
from ..config.config import CFG

def compute_tau(
    sigma: torch.Tensor,
    delta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the cumulative optical thickness (tau) along ray samples.

    Args:
        sigma: Tensor of shape (..., N), density values for each sample.
        delta: Tensor of shape (..., N), distance intervals between samples.

    Returns:
        tau: Tensor of shape (..., N), where tau[..., i] = \sum_{j<i} sigma[..., j] * delta[..., j].
    """
    # Cumulative sum of sigma * delta, shifting right by one and prepending zeros
    cum_sigma_delta = torch.cumsum(sigma[..., :-1] * delta[..., :-1], dim=-1)
    tau = torch.cat([torch.zeros_like(cum_sigma_delta[..., :1]), cum_sigma_delta], dim=-1)
    return tau


def compute_transmittance(
    sigma: torch.Tensor,
    delta: torch.Tensor
) -> torch.Tensor:
    """
    Compute transmittance T_i = exp(-tau_i) before each sample.

    Args:
        sigma: Tensor of shape (..., N), density values.
        delta: Tensor of shape (..., N), distance intervals.

    Returns:
        T: Tensor of shape (..., N), transmittance before each sample.
    """
    tau = compute_tau(sigma, delta)
    return torch.exp(-tau)


def compute_weights(
    sigma: torch.Tensor,
    delta: torch.Tensor,
    T: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-sample opacity weights:
        w_i = T_i * (1 - exp(-sigma_i * delta_i)).

    Args:
        sigma: Tensor of shape (..., N), density values.
        delta: Tensor of shape (..., N), distance intervals.
        T: (Optional) precomputed transmittance tensor of shape (..., N).
           If None, will be computed internally.

    Returns:
        w: Tensor of shape (..., N), weight of each sample.
    """
    if T is None:
        T = compute_transmittance(sigma, delta)
    alpha = 1 - torch.exp(-sigma * delta)
    w = T * alpha
    return w


def render_opacity(
    sigma: torch.Tensor,
    delta: torch.Tensor,
    w: Optional[torch.Tensor] = None

) -> torch.Tensor:
    """
    Integrate per-sample weights to get total opacity per ray:
        C = sum_i w_i.

    Returns:
        C: Tensor of shape (...), total opacity in [0, 1].
    """
    if w is None:
      w = compute_weights(sigma, delta)

    return torch.sum(w, dim=-1)


def render_depth(
    sigma: torch.Tensor,
    delta: torch.Tensor,
    z_vals: torch.Tensor
) -> torch.Tensor:
    """
    Compute expected depth per ray:
        depth = sum_i w_i * z_i.

    Args:
        sigma: Tensor of shape (..., N), density values.
        delta: Tensor of shape (..., N), distance intervals.
        z_vals: Tensor of shape (..., N), sample depths along the ray.

    Returns:
        depth: Tensor of shape (...), expected termination depth.
    """
    w = compute_weights(sigma, delta)
    depth = torch.sum(w * z_vals, dim=-1)
    return depth


# Example utility that ties sampling and rendering together
def sample_and_render(
    pts: torch.Tensor,
    delta: torch.Tensor,
    t_vals: torch.Tensor,
    query_density_field: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render opacity and depth given 3D sample points and ray distances.

    Args:
        pts: Tensor of shape (R, N, 3), 3D sample points along each ray.
        delta: Tensor of shape (R, N), distance intervals between samples.
        t_vals: Tensor of shape (R, N), absolute distances along each ray.
        query_density_field: Function mapping pts -> sigma of shape (R, N).

    Returns:
        C: Tensor of shape (R,), total opacity per ray.
        depth: Tensor of shape (R,), expected depth per ray.
        sigma: Tensor of shape (R, N), density values per sample.
    """
    # Query densities (can be ground truth or model predictions)
    sigma = query_density_field(pts)

    # Compute opacity and depth
    C = render_opacity(sigma, delta)
    depth = render_depth(sigma, delta, t_vals)

    return C, depth, sigma

def get_batch_rays(
    H: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ray origins and directions given camera intrinsics and pose.

    Args:
        H: (4,4) camera-to-world transform.

    Returns:
        O_w: (R,3) ray origins.
        D_w: (R,3) ray directions.
    """
    intr  = Intrinsics(**CFG.intrinsics.__dict__)
    cam = Camera(intr, H, t_bounds=(CFG.rays.t_near, CFG.rays.t_far))
    rays_per_pose = CFG.rays.R
    O_all, D_all = cam.get_rays(frame='world')
    idx = torch.randperm(O_all.shape[0])[:rays_per_pose]

    O_batch, D_batch = torch.from_numpy(O_all[idx]), torch.from_numpy(D_all[idx])
    return O_batch, D_batch

