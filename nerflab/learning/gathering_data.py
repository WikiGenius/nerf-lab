# nerflab/learning/gathering_data.py
import numpy as np
import h5py
import torch

from ..camera.sampling import stratified_samples_batch
from .ops import compute_weights, get_batch_rays, query_density_field
from..config.config import CFG

def gather_and_save(
    poses,                 # Iterable[Tensor or ndarray] of shape [4,4] (cam-to-world)
    output_path="data.h5"
):
    all_H, all_o, all_d = [], [], []
    all_z, all_delta     = [], []
    all_sigma, all_C     = [], []
    all_depth            = []

    # Read once from config
    t_near     = float(CFG.rays.t_near)
    t_far      = float(CFG.rays.t_far)
    N_samples  = int(CFG.rays.N)
    deterministic = getattr(CFG.rays, "deterministic", True)

    for H in poses:
        # 0) Pose as torch
        H_t = torch.as_tensor(H, dtype=torch.float32)

        # 1) Rays (ensure torch)
        O_w, D_w = get_batch_rays(H_t) 

        # 2) Sample along rays
        t, delta, pts = stratified_samples_batch(O_w, D_w)

        # 3) Sigma field (ensure torch)
        sigma_gt = query_density_field(pts)        # expected [R,N]
        sigma_gt = torch.as_tensor(sigma_gt, dtype=pts.dtype, device=pts.device)

        # 4) Weights & integrals
        w     = compute_weights(sigma_gt, delta)   # [R,N]
        C     = w.sum(dim=-1)                      # [R]
        depth = (w * t).sum(dim=-1)                # [R]

        # 5) Collect on CPU as NumPy
        all_H.append(      H_t.cpu().numpy())
        all_o.append(      O_w.cpu().numpy())
        all_d.append(      D_w.cpu().numpy())
        all_z.append(         t.detach().cpu().numpy())
        all_delta.append(  delta.detach().cpu().numpy())
        all_sigma.append(  sigma_gt.detach().cpu().numpy())
        all_C.append(      C.detach().cpu().numpy())
        all_depth.append(  depth.detach().cpu().numpy())

    # 6) Stack & save (shapes must be consistent across poses)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("poses",    data=np.stack(all_H))
        f.create_dataset("rays_o",   data=np.stack(all_o))
        f.create_dataset("rays_d",   data=np.stack(all_d))
        f.create_dataset("z_vals",   data=np.stack(all_z))
        f.create_dataset("deltas",   data=np.stack(all_delta))
        f.create_dataset("sigma_gt", data=np.stack(all_sigma))
        f.create_dataset("C_gt",     data=np.stack(all_C))
        f.create_dataset("depth_gt", data=np.stack(all_depth))

    print(f"Saved dataset to {output_path}")
