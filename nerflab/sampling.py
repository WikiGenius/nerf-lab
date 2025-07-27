#from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


#  ============================================================
# Stratified sampling (vectorized over rays)
# ============================================================

def stratified_samples_batch(origins: np.ndarray,
                             dirs: np.ndarray,
                             t_near: float | np.ndarray,
                             t_far:  float | np.ndarray,
                             N: int,
                             rng: Optional[np.random.Generator] = None,
                             deterministic: bool = False
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified (or deterministic mid-bin) samples along many rays.

    Parameters
    ----------
    origins : (R,3)
    dirs    : (R,3)
    t_near, t_far : float or (R,)
        Near/far per ray or shared scalars.
    N : int
        Samples per ray.
    rng : np.random.Generator, optional
    deterministic : bool
        If True, choose bin centers (no randomness).

    Returns
    -------
    t     : (R,N)  sample depths along each ray
    delta : (R,N)  distance to next sample (last uses t_far)
    pts   : (R,N,3) sampled 3D points
    """
    if N <= 0:
        raise ValueError("N must be > 0")

    if rng is None:
        rng = np.random.default_rng()

    R = origins.shape[0]
    # Broadcast to (R,)
    t_near = np.broadcast_to(np.float32(t_near), (R,))
    t_far  = np.broadcast_to(np.float32(t_far),  (R,))

    # Normalized bin edges
    edges01 = np.linspace(0.0, 1.0, N + 1, dtype=np.float32)  # (N+1,)

    if deterministic:
        # Midpoints of each bin
        u = 0.5 * (edges01[:-1] + edges01[1:])[None, :]        # (1,N)
        u = np.broadcast_to(u, (R, N)).astype(np.float32)
    else:
        # One random sample per bin
        u = rng.uniform(edges01[:-1], edges01[1:], size=(R, N)).astype(np.float32)
    # Map to each ray interval
    t = t_near[:, None] + (t_far - t_near)[:, None] * u        # (R,N)

    # Distances to next sample
    delta = np.diff(np.concatenate([t, t_far[:, None]], axis=1), axis=1)  # (R,N)

    # 3D points on rays
    pts = origins[:, None, :] + t[..., None] * dirs[:, None, :]           # (R,N,3)

    return t, delta, pts

# ============================================================
# Spherical angles (optional utility)
# ============================================================

def cartesian_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (x,y,z) → (r, theta, phi):
        r     : radius
        theta : polar angle from +Z (0..pi)
        phi   : azimuth from +X toward +Y (-pi..pi)
    points: (...,3)

    Returns
    -------
    r, theta, phi : arrays with shape points.shape[:-1]
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    r = np.linalg.norm(points, axis=-1)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z / (r + 1e-8), -1.0, 1.0))
    return r, theta, phi