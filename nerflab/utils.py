import numpy as np

# ---------- Small helpers ----------
def homogenize(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p, np.ones((*p.shape[:-1], 1), dtype=p.dtype)], axis=-1)

def dehomogenize(ph: np.ndarray) -> np.ndarray:
    return ph[..., :3] / ph[..., 3:4]

def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def distance(p1, p2) -> float:
    """
    Euclidean distance between two points (any dimension).

    Parameters
    ----------
    p1, p2 : array_like
        Coordinates of the two points (e.g., [x,y], [x,y,z], ...).

    Returns
    -------
    float
        ||p2 - p1||
    """
    return float(np.linalg.norm(np.asarray(p2, dtype=np.float32) - np.asarray(p1, dtype=np.float32)))

def look_at(eye, target, up=(0,1,0), forward_is_neg_z=True, eps=1e-8, dtype=np.float32):
    eye, target, up = [np.asarray(a, dtype=dtype) for a in (eye, target, up)]

    f = target - eye
    if forward_is_neg_z:
        f = -f
    z = f / (np.linalg.norm(f) + eps)

    if abs(np.dot(up, z)) / (np.linalg.norm(up) + eps) > 0.999:
        up = np.array([0,0,1], dtype=dtype)

    # NOTE: up × z to get +X (right) like world
    x = np.cross(up, z); x /= (np.linalg.norm(x) + eps)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)
    T = np.eye(4, dtype=dtype)
    T[:3,:3] = R
    T[:3, 3] = eye
    return T
