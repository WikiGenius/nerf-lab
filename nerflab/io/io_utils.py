# # nerflab/io/io_utils.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
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


def _parse_seed_from_npz_name(p: Path) -> Optional[int]:
    """
    Extract integer seed from '<frame_id>__seed<seed>.npz' name.
    Returns None if pattern is missing.
    """
    stem = p.stem  # '<frame_id>__seed<seed>'
    if "__seed" not in stem:
        return None
    try:
        return int(stem.split("__seed", 1)[1])
    except Exception:
        return None

def get_frame_ids_for_case(
    scene_dir: Path,
    split: str,
    *,
    seed: int,
    cfg_hash: Optional[str] = None,
    num_frames: int = 1,
    frame_offset: int = 0,
) -> Tuple[List[str], str]:
    """
    Resolve a slice of frame_ids for a given (scene, split, seed), optionally
    discovering cfg_hash.

    Selection policy
    ----------------
    1) If cfg_hash is None, pick the first cfg folder under 'rays/cache' that
       contains the requested split (same behavior as discover_cfg_hash).
    2) Within that cfg/split directory, filter NPZ files by '__seed{seed}.npz'.
    3) Sort paths lexicographically (so '000020' < '000101'), then slice
       [frame_offset : frame_offset + num_frames].
       Offsets can be negative (Python semantics).

    Raises
    ------
    FileNotFoundError : if cache paths are missing.
    ValueError        : if no files for the given seed, or slice is empty / OOB.

    Returns
    -------
    (frame_ids, cfg_hash_used)
    """
    # 1) cfg discovery if needed
    cfg_used = cfg_hash or discover_cfg_hash(scene_dir, split)

    # 2) list and filter by seed
    all_paths = list_frame_npz(scene_dir, cfg_used, split)  # sorted
    seed_paths = [p for p in all_paths if f"__seed{seed}.npz" in p.name]

    if not seed_paths:
        # Help the user by listing available seeds we saw
        avail = sorted({s for s in (_parse_seed_from_npz_name(p) for p in all_paths) if s is not None})
        raise ValueError(
            f"No NPZ files found for seed={seed} in '{scene_dir}/rays/cache/{cfg_used}/{split}'. "
            f"Available seeds: {avail}"
        )

    # 3) sort (defensive; list_frame_npz already sorts) and slice
    seed_paths = sorted(seed_paths)
    ids = frame_ids_from_paths(seed_paths)

    # Handle negative offsets like standard Python slicing
    n = len(ids)
    start = frame_offset if frame_offset >= 0 else max(0, n + frame_offset)
    end = start + num_frames

    # Clamp end to n, then validate count
    end = min(end, n)
    sel = ids[start:end]

    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0 (got {num_frames}).")
    if start >= n or len(sel) == 0:
        raise ValueError(
            f"Requested slice is empty: offset={frame_offset}, num_frames={num_frames}, total={n}."
        )
    if len(sel) < num_frames:
        # Be explicit so the caller knows they got fewer frames than requested
        raise ValueError(
            f"Only {len(sel)} frames available from offset {frame_offset} (requested {num_frames}, total {n})."
        )

    return sel, cfg_used
