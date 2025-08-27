import torch
import torch.nn.functional as F
from typing import Tuple, Union
from .ops import compute_transmittance, compute_weights, render_opacity

def nerf_opacity(
    sigma: torch.Tensor,
    delta: torch.Tensor,
    max_sigma: float = 1e6,
    full_output: bool = True
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute NeRF-style volume-rendering opacity.

    This implements the discrete volume rendering equations:
      - Tᵢ = exp⁡(−τᵢ)                      transmittance before sample i
      - wᵢ = Tᵢ · (1 − exp⁡(−σᵢ·Δᵢ))        weight of sample i
      - C = ∑ᵢ wᵢ                          total opacity per ray

    Where
      τᵢ = ∑_{j<i} σⱼ·Δⱼ    is the optical thickness up to (but not including) i.

    Args:
        sigma (torch.Tensor):  Densities, shape (B, R, N), (R, N)
        delta (torch.Tensor):  Distances between samples, shape (B, R, N), (R, N)
        max_sigma (float):     Clip any +Inf densities to this value
        full_output (bool):    If True, returns (T, w, C); otherwise returns C only

    Returns:
        If full_output:
            T (torch.Tensor): Transmittance before each sample, shape (B, R, N), (R, N)
            w (torch.Tensor): Weights per sample,           shape (B, R, N), (R, N)
            C (torch.Tensor): Accumulated opacity per ray,  shape (B, R,), (R,)
        Else:
            C (torch.Tensor): Accumulated opacity per ray, shape (B, R,), (R,)
    """
    # 1) Sanitize NaNs and Infs
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=max_sigma, neginf=0.0)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0,     neginf=0.0)

    # 2) Clamp to valid ranges
    sigma = sigma.clamp(min=0.0, max=max_sigma)
    delta = delta.clamp(min=0.0)

    # 3) Compute transmittance T and weights w
    T = compute_transmittance(sigma, delta)   # (B, R, N), (R, N)
    w = compute_weights(sigma, delta, T)      # (B, R, N), (R, N)
    C = render_opacity(sigma, delta, w)       # (B, R,), (R,)

    return (T, w, C) if full_output else C


def compute_opacity_simple(
    sigma: torch.Tensor,
    delta: torch.Tensor
) -> torch.Tensor:
    """
    Fast scalar opacity:
      C = 1 − exp(−∫ σ(s) ds)
    approximated by Riemann sum:
      C ≈ 1 − exp(−∑ᵢ σᵢ·Δᵢ)

    Args:
        sigma (torch.Tensor): Densities, shape (B, R, N), (R, N)
        delta (torch.Tensor): Sample distances, shape (B, R, N), (R, N)

    Returns:
        C (torch.Tensor): Opacity per ray, shape (B, R,), (R,)
    """
    # ∑ σᵢ·Δᵢ across samples
    tau_total = torch.sum(sigma * delta, dim=-1)     # (R,) (B, R,)
    # analytic opacity
    return 1.0 - torch.exp(-tau_total)


