import numpy as np

def nerf_opacity(sigma: np.ndarray, delta: np.ndarray,
                      max_sigma: float = 1e6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NeRF transmittance, weights, and accumulated opacity (no colors).

    Parameters
    ----------
    sigma : (R, N)
        Density at each sample.
    delta : (R, N)
        Distance between successive samples.

    Returns
    -------
    T : (R, N)        # transmittance before each sample
    w : (R, N)        # weights = T * (1 - exp(-sigma*delta))
    C : (R,)          # accumulated opacity per ray
    """
    # 1) Replace NaN/Inf
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=max_sigma, neginf=0.0).astype(np.float32)
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 2) Ensure non-negative
    sigma = np.clip(sigma, 0.0, max_sigma)
    delta = np.clip(delta, 0.0, None)

    # 3) Compute cumulative optical thickness
    # cumulative optical thickness up to (but NOT including) i
    tau_cum = np.cumsum(sigma * delta, axis=1)
    tau_shifted = np.pad(tau_cum, ((0, 0), (1, 0)), mode='constant')[:, :-1] # shift right

    T = np.exp(-tau_shifted)                    # (R,N)
    w = T * (1.0 - np.exp(-sigma * delta))      # (R,N)

    C = w.sum(axis=1)                           # (R,)

    return T.astype(np.float32), w.astype(np.float32), C.astype(np.float32)

def nerf_opacity_single(sigma: np.ndarray, delta: np.ndarray) -> float:
    _, w, _ = nerf_opacity(sigma[0].reshape(1, -1), delta[0].reshape(1, -1))
    return float(w.sum())

# print(nerf_opacity_single(sigma_g, delta))

def sigma_from_world(world, pts: np.ndarray, max_sigma = 10e6) -> np.ndarray:
    """
    Vectorized sigma query.

    Parameters
    ----------
    world : World
        Must expose world.density(x,y,z) -> float (your code already uses it).
    pts   : (R,N,3) array
        Sampled 3D points.

    Returns
    -------
    sigma : (R,N) float32
    """
    R, N, _ = pts.shape
    P = pts.reshape(-1, 3)                       # (R*N,3)
    # evaluate density per point
    sig = np.array([world.density(*p) for p in P], dtype=np.float32)
    # 1) Replace NaN/Inf
    sig = np.nan_to_num(sig, nan=0.0, posinf=max_sigma, neginf=0.0).astype(np.float32)
    # 2) Ensure non-negative
    sig = np.clip(sig, 0.0, max_sigma)

    return sig.reshape(R, N)
