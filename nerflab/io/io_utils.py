# # nerflab/io/io_utils.py
from __future__ import annotations
from pathlib import Path
from typing import List
import torch
from nerflab import Intrinsics, Camera, CFG


def discover_cfg_hash(scene_dir: Path, split: str) -> str:
    cache_root = scene_dir / "rays" / "cache"
    if not cache_root.exists():
        raise FileNotFoundError(f"Missing cache root: {cache_root}")
    for cfg_dir in cache_root.iterdir():
        if not cfg_dir.is_dir():
            continue
        if (cfg_dir / split).exists():
            return cfg_dir.name
    raise FileNotFoundError(f"No cfg_hash folder found under: {cache_root}/*/{split}")

def list_frame_npz(scene_dir: Path, cfg_hash: str, split: str) -> list[Path]:
    d = scene_dir / "rays" / "cache" / cfg_hash / split
    if not d.exists():
        raise FileNotFoundError(f"Missing cache dir: {d}")
    return sorted(d.glob("*__seed*.npz"))

def frame_ids_from_paths(paths: list[Path]) -> List[str]:
    """Extract frame_id from '<id>__seedX.npz'."""
    return [p.stem.split("__seed")[0] for p in paths]

def camera_from_loaded_H(H_wc: torch.Tensor) -> Camera:
    """Build a (possibly batched) Camera using H_wc from the loaded batch."""
    intr = Intrinsics(**CFG.intrinsics.__dict__)
    cam = Camera(H_wc, intr, t_bounds=(CFG.rays.t_near, CFG.rays.t_far))
    return cam
