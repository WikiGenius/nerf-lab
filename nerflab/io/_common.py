from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional
import numpy as np
import torch


# ---------------------------- Filesystem helpers ---------------------------- #

def ensure_dir(path: str) -> None:
    """Create directory path if it does not exist (idempotent)."""
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file as a Python dict."""
    with open(path, "r") as f:
        return json.load(f)


def maybe_read_json(path: str) -> Optional[Dict[str, Any]]:
    """Read JSON if file exists, else return None."""
    if not os.path.exists(path):
        return None
    return read_json(path)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a JSON file with pretty indentation; ensure parent directory exists."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------ Tensor <-> NP ------------------------------- #

def to_numpy(x: torch.Tensor, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Detach and move to CPU, then convert to NumPy with optional dtype cast.
    """
    a = x.detach().cpu().numpy()
    return a.astype(dtype) if dtype is not None else a


def as_tensor_like(
    a: np.ndarray,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert numpy array to tensor on optional device/dtype.
    """
    t = torch.as_tensor(a)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def stack_arrays(
    arrays: list,
    *,
    as_torch: bool,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Stack a list of arrays/ tensors along a new leading axis.
    Arrays can be numpy or torch; output dtype/device controlled by flags.
    """
    if as_torch:
        return torch.stack(
            [
                a if torch.is_tensor(a)
                else torch.as_tensor(a, device=device, dtype=dtype)
                for a in arrays
            ],
            dim=0,
        )
    else:
        return np.stack([a if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
                         for a in arrays], axis=0)
