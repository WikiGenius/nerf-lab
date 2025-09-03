from typing import Tuple, Union
import torch
from torch import nn
from ..learning_utils.metrics import psnr_from_mse

# NOTE: these are *not* package deps; your script must provide them:
# from nerflab import Camera
# from nerflab.learning.forward_sigma import nerf_opacity

@torch.no_grad()
def render_full_opacity(
    model: nn.Module,
    H_row: torch.Tensor,
    rng,
    device: str,
    chunk_rays: int = 65536,
    return_sigma: bool = False,
    return_samples: bool = False,
):
    """
    Full-frame render (opacity) for a single pose.
    Returns either just C_hat_flat, or (C_hat_flat, sigma_full), or (C_hat_flat, (t,delta,pts)),
    or (C_hat_flat, sigma_full, (t,delta,pts)) depending on flags.
    Shapes:
      C_hat_flat: (H*W,)
      sigma_full: (H*W, N)
      t, delta:   (H*W, N)
      pts:        (H*W, N, 3)
    """
    from ...camera import Camera
    from ..ops.forward_sigma import nerf_opacity

    model.eval()
    cam_r = Camera(H_row)
    O, D = cam_r.get_rays()
    t, delta, pts = cam_r.sample_along_rays(O, D, rng=rng)

    Rays, N, _ = pts.shape
    out_opacity = []
    out_sigma = [] if return_sigma else None

    for i in range(0, Rays, chunk_rays):
        pts_sl = pts[i: i + chunk_rays].to(device)
        delta_sl = delta[i: i + chunk_rays].to(device)

        sigma_sl = model(pts_sl)  # (r', N)
        C_hat_sl = nerf_opacity(sigma_sl, delta_sl, full_output=False)  # (r',)

        out_opacity.append(C_hat_sl.detach().cpu())
        if return_sigma:
            out_sigma.append(sigma_sl.detach().cpu())

    C_hat_flat = torch.cat(out_opacity, dim=0)
    if return_sigma:
        sigma_full = torch.cat(out_sigma, dim=0)

    if return_sigma and return_samples:
        return C_hat_flat, sigma_full, (t, delta, pts)
    elif return_sigma:
        return C_hat_flat, sigma_full
    elif return_samples:
        return C_hat_flat, (t, delta, pts)
    else:
        return C_hat_flat

@torch.no_grad()
def eval_psnr_on_frame(
    model: nn.Module,
    H_row: torch.Tensor,
    img_row: torch.Tensor,   # (H, W) in [0,1]
    rng,
    device: str,
    chunk_rays_eval: int = 65536
) -> Tuple[float, float]:
    """
    True MSE/PSNR by rendering the full frame and comparing to img_row.
    """
    C_hat_flat = render_full_opacity(model, H_row, rng, device, chunk_rays_eval)
    gt_flat = img_row.reshape(-1).float().cpu()
    mse = torch.mean((C_hat_flat - gt_flat) ** 2).item()
    return mse, psnr_from_mse(mse)
