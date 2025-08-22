# nerflab/transforms.py
from __future__ import annotations
from typing import Sequence, Union, Iterable, Optional
import torch


TensorLike = Union[torch.Tensor, Sequence[float], float]

# =============================================================================
# Small helpers (torch-native)
# =============================================================================

def homogenize(p: torch.Tensor) -> torch.Tensor:
    """
    Append a homogeneous 1 to points.

    Parameters
    ----------
    p : (..., 3) tensor

    Returns
    -------
    ph : (..., 4) tensor
    """
    assert p.shape[-1] == 3, "p must have last dim = 3"
    ones = torch.ones(*p.shape[:-1], 1, device=p.device, dtype=p.dtype)
    return torch.cat([p, ones], dim=-1)


def dehomogenize(ph: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert homogeneous coords to Euclidean.

    Parameters
    ----------
    ph : (..., 4) tensor

    Returns
    -------
    p : (..., 3) tensor
    """
    assert ph.shape[-1] == 4, "ph must have last dim = 4"
    w = ph[..., 3:4].clamp_min(eps)
    return ph[..., :3] / w



def invert_T(T: torch.Tensor) -> torch.Tensor:
    """
    Invert a rigid 4x4 (or batch of) homogeneous transforms.

    Parameters
    ----------
    T : (..., 4, 4) tensor
        [ R t ]
        [ 0 1 ]

    Returns
    -------
    Ti : (..., 4, 4) tensor
        Inverse transform:
        [ R^T  -R^T t ]
        [  0       1  ]
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must have shape (..., 4, 4)")

    R = T[..., :3, :3]                 # (..., 3, 3)
    t = T[..., :3, 3]                  # (..., 3)
    RT = R.transpose(-1, -2)           # (..., 3, 3)

    # t_inv = -R^T t, with proper broadcasting and shape
    t_inv = -torch.matmul(RT, t.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Build identity of matching batch/device/dtype, then fill
    eye4 = torch.eye(4, device=T.device, dtype=T.dtype)
    if T.ndim > 2:
        Ti = eye4.expand(T.shape[:-2] + (4, 4)).clone()
    else:
        Ti = eye4.clone()

    Ti[..., :3, :3] = RT
    Ti[..., :3, 3]  = t_inv
    return Ti



def distance(p1: TensorLike, p2: TensorLike) -> Union[float, torch.Tensor]:
    """
    Euclidean distance between points.

    Parameters
    ----------
    p1, p2 : (..., D) tensors/sequences

    Returns
    -------
    d : float if inputs are 1D sequences, else tensor (...,)
    """
    p1_t = torch.as_tensor(p1, dtype=torch.float32)
    p2_t = torch.as_tensor(p2, dtype=torch.float32, device=p1_t.device)
    d = torch.linalg.norm(p2_t - p1_t, dim=-1)
    # If both were plain 1D sequences, return a Python float for convenience.
    return float(d.item()) if d.ndim == 0 else d


def look_at(
    eye: TensorLike,
    target: TensorLike,
    up: TensorLike = (0.0, 1.0, 0.0),
    *,
    forward_is_neg_z: bool = True,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Construct a camera pose (4x4) looking from `eye` to `target`.

    Coordinates follow a right-handed convention:
      x: right, y: up, z: forward (by default pointing toward -Z like many graphics APIs).

    Parameters
    ----------
    eye : (3,)
    target : (3,)
    up : (3,), nominal up direction
    forward_is_neg_z : bool
        If True, forward vector is aligned with -Z.
    eps : float
    dtype : torch.dtype
    device : torch.device or None

    Returns
    -------
    T_c2w : (4,4) tensor
        Camera-to-world transform with rotation [x y z] as columns and translation = eye.
    """
    eye_t    = torch.as_tensor(eye,    dtype=dtype, device=device).reshape(3)
    target_t = torch.as_tensor(target, dtype=dtype, device=device).reshape(3)
    up_t     = torch.as_tensor(up,     dtype=dtype, device=device).reshape(3)

    f = target_t - eye_t
    if forward_is_neg_z:
        f = -f
    z = f / (torch.linalg.norm(f) + eps)

    # If up is (near-)colinear with z, pick a fallback up
    if torch.abs(torch.dot(up_t, z)) / (torch.linalg.norm(up_t) + eps) > 0.999:
        up_t = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    x = torch.linalg.cross(up_t, z, dim=-1); x = x / (torch.linalg.norm(x) + eps)
    y = torch.linalg.cross(z, x, dim=-1)
    
    R = torch.stack([x, y, z], dim=1)  # columns
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R
    T[:3, 3]  = eye_t
    return T

def validate_se3(
    H: torch.Tensor,
    *,
    name: str = "H",
    atol_last_row: float = 1e-6,
    tol_ortho: float = 1e-3,
    tol_det: float = 1e-3,
    repair: bool = False,   # if True, project R to nearest SO(3) and fix last row
) -> torch.Tensor:
    """
    Ensure H is a (B?,4,4) homogeneous transform with:
      - last row == [0,0,0,1] (within atol_last_row)
      - R^T R ≈ I (within tol_ortho)
      - det(R) ≈ +1 (within tol_det)
      - finite entries

    If repair=True, projects R to nearest SO(3) by SVD and sets last row exactly.
    """
    if H.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape (...,4,4), got {tuple(H.shape)}")

    batched = H.ndim == 3
    if not batched:
        H = H.unsqueeze(0)

    B = H.shape[0]
    device, dtype = H.device, H.dtype

    R = H[..., :3, :3]
    t = H[..., :3, 3]
    last = H[..., 3, :]

    target_last = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype).expand(B, 4)
    last_err = (last - target_last).abs().amax(dim=-1)
    row_ok = last_err <= atol_last_row

    I3 = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3)
    RtR = R.transpose(-1, -2) @ R
    ortho_err = torch.linalg.matrix_norm(RtR - I3, ord="fro", dim=(-2, -1))
    ortho_ok = ortho_err <= tol_ortho

    detR = torch.linalg.det(R)
    det_err = (detR - 1.0).abs()
    det_ok = det_err <= tol_det

    finite_ok = torch.isfinite(H.reshape(B, -1)).all(dim=-1)

    ok = row_ok & ortho_ok & det_ok & finite_ok

    if repair:
        # Project ALL to nearest SO(3) and fix last row; simpler & safe
        U, _, Vh = torch.linalg.svd(R)
        Rproj = U @ Vh
        # enforce det +1
        detRproj = torch.linalg.det(Rproj)
        flip = (detRproj < 0).to(dtype=dtype)
        # flip last column of U when needed
        U_fix = U.clone()
        U_fix[..., :, 2] = U_fix[..., :, 2] * (1.0 - 2.0 * flip)[..., None]
        Rproj = U_fix @ Vh

        Hfixed = H.clone()
        Hfixed[..., :3, :3] = Rproj
        Hfixed[..., 3, :] = target_last

        # Re-evaluate after repair
        R = Hfixed[..., :3, :3]
        RtR = R.transpose(-1, -2) @ R
        ortho_err = torch.linalg.matrix_norm(RtR - I3, ord="fro", dim=(-2, -1))
        det_err = (torch.linalg.det(R) - 1.0).abs()
        last_err = (Hfixed[..., 3, :] - target_last).abs().amax(dim=-1)
        ok = (ortho_err <= tol_ortho) & (det_err <= tol_det) & (last_err <= atol_last_row) \
             & torch.isfinite(Hfixed.reshape(B, -1)).all(dim=-1)

        H = Hfixed

    if not ok.all():
        bad_idx = (~ok).nonzero(as_tuple=False).squeeze(-1).tolist()
        details = {
            "row_err": last_err.detach().cpu().tolist(),
            "ortho_err": ortho_err.detach().cpu().tolist(),
            "det_err": det_err.detach().cpu().tolist(),
        }
        raise ValueError(
            f"{name} contains {len(bad_idx)} invalid SE(3) matrix/matrices at indices {bad_idx}. "
            f"Violations: non-orthonormal R / det!=+1 / bad last row / non-finite. "
            f"Details (per batch): {details}"
        )

    return H[0] if not batched else H

__all__ = [
    "homogenize",
    "dehomogenize",
    "invert_T",
    "distance",
    "look_at",
    "validate_se3"
]
