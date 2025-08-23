from __future__ import annotations
import os
import json

import numpy as np
from .geometry import World, Box, Sphere
from typing import Sequence, Union, Optional
import torch


TensorLike = Union[torch.Tensor, Sequence[float], float]





# =============================================================================
# World (de)serialization (JSON)
# =============================================================================

def _to_list(x: TensorLike) -> list:
    """Convert scalars/tensors/sequences to plain Python lists for JSON."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [(_to_list(v) if isinstance(v, (torch.Tensor, list, tuple)) else float(v)) for v in x]
    return [float(x)]


def save_world(world: World, filepath: str) -> None:
    """
    Serialize a World (with Box and Sphere shapes) to a JSON file.
    Ensures parent directory exists.

    Parameters
    ----------
    world : World
    filepath : str
    """
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    data = []
    for shape in world.shapes:
        if isinstance(shape, Box):
            data.append({
                "type": "Box",
                "center": _to_list(shape.center),
                "size":   _to_list(shape.size),
            })
        elif isinstance(shape, Sphere):
            data.append({
                "type": "Sphere",
                "center": _to_list(shape.center),
                "radius": _to_list(shape.radius)[0],
            })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_world(filepath: str, *, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> World:
    """
    Deserialize a JSON file into a World instance (torch tensors on `device`).

    Parameters
    ----------
    filepath : str
    dtype : torch.dtype
    device : torch.device or None

    Returns
    -------
    world : World
    """
    world = World()
    with open(filepath, "r") as f:
        data = json.load(f)

    for item in data:
        typ = item.get("type", "")
        if typ == "Box":
            c = torch.tensor(item["center"], dtype=dtype, device=device)
            s = torch.tensor(item["size"],   dtype=dtype, device=device)
            world.add_shape(Box(center=c, size=s))
        elif typ == "Sphere":
            c = torch.tensor(item["center"], dtype=dtype, device=device)
            r = torch.tensor(item["radius"], dtype=dtype, device=device)
            world.add_shape(Sphere(center=c, radius=r))
        else:
            raise ValueError(f"Unknown shape type in JSON: {typ}")
    return world


__all__ = [
    "save_world",
    "load_world",
]
