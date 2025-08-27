# nerflab/io/dataset_load.py
from __future__ import annotations
import os, glob
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from ._common import read_json, as_tensor_like, stack_arrays


# ---------------------------- Basics / discovery ---------------------------- #

def list_cfg_hashes(scene_dir: str) -> List[str]:
    """
    List all cfg-hash JSON basenames under <scene>/rays/cfgs/*.json.

    Returns
    -------
    List[str] : cfg hashes (filenames without .json)
    """
    p = os.path.join(scene_dir, "rays", "cfgs", "*.json")
    return [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(p)]


def load_cfg(scene_dir: str, cfg_hash: str) -> Dict[str, Any]:
    """
    Load the config JSON corresponding to a cfg hash.
    """
    jpath = os.path.join(scene_dir, "rays", "cfgs", f"{cfg_hash}.json")
    return read_json(jpath)


def load_transforms(scene_dir: str) -> Dict[str, Any]:
    """
    Load transforms.json manifest (camera intrinsics + frames metadata).
    """
    tpath = os.path.join(scene_dir, "transforms.json")
    return read_json(tpath)


def frame_npz_path(scene_dir: str, cfg_hash: str, split: str, frame_id: str, seed: int) -> str:
    """
    Standard path for a rays cache NPZ for a given (cfg, split, frame, seed).
    """
    return os.path.join(scene_dir, "rays", "cache", cfg_hash, split, f"{frame_id}__seed{seed}.npz")


# ------------------------------- NPZ: one file ------------------------------ #

def _coerce_scalars_inplace(d: Dict[str, Any]) -> None:
    """
    Convert 0-d arrays to Python scalars/strings for convenience.
    Mutates the dict `d` in place.
    """
    for k in ("seed", "width", "height"):
        if k in d and isinstance(d[k], np.ndarray) and d[k].shape == ():
            d[k] = d[k].item()

    for k in ("frame_id", "cfg_hash"):
        if k in d and isinstance(d[k], np.ndarray) and d[k].shape == ():
            # keep str type even if it was a 0-d bytes array
            d[k] = d[k].item() if isinstance(d[k].item(), str) else str(d[k])


def load_frame_npz(
    path: str,
    *,
    as_torch: bool = False,
    device: Optional[torch.device] = None,
    float_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    Load a single rays NPZ written by our saver.

    Parameters
    ----------
    path : str
        NPZ file path.
    as_torch : bool
        If True, numpy arrays are converted to torch.Tensors on `device`
        with float arrays cast to `float_dtype`. Integer arrays keep integer dtype.

    Returns
    -------
    Dict[str, Any]
        Keys may include: O, D, t, delta, X, sigma, T, w, C, pix_idx, width, height,
        seed, frame_id, cfg_hash.
    """
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}

    _coerce_scalars_inplace(data)

    if not as_torch:
        return data

    dev = device or torch.device("cpu")
    out: Dict[str, Any] = {}

    float_keys = {"O", "D", "t", "delta", "X", "sigma", "T", "w", "C"}
    int_keys = {"pix_idx", "width", "height", "seed"}
    str_keys = {"frame_id", "cfg_hash"}

    for k, v in data.items():
        if k in float_keys and isinstance(v, np.ndarray):
            out[k] = as_tensor_like(v, device=dev, dtype=float_dtype)
        elif k in int_keys and isinstance(v, np.ndarray):
            # keep integer dtype
            out[k] = as_tensor_like(v, device=dev, dtype=torch.long if v.dtype.kind in "iu" else None)
        elif k in str_keys:
            out[k] = v  # keep as str
        else:
            # unknown key or already tensor/py object
            out[k] = v

    return out


# ---------------------- H_wc lookup for a list of frames -------------------- #

def get_H_wc_batch(scene_dir: str, frame_ids: List[str]) -> torch.Tensor:
    """
    Return H_wc as (B,4,4) for the provided `frame_ids` (order preserved).

    Raises
    ------
    KeyError  : if any frame id is missing
    ValueError: if any pose has wrong shape
    """
    tr = load_transforms(scene_dir)
    frames = {f["id"]: f for f in tr.get("frames", [])}
    H_list: List[torch.Tensor] = []

    for fid in frame_ids:
        f = frames.get(fid)
        if f is None:
            raise KeyError(f"frame_id '{fid}' not found in transforms.json")
        pose = f.get("pose", {})
        H = np.array(pose.get("H_wc"), dtype=np.float32)
        if H.shape != (4, 4):
            raise ValueError(f"H_wc for frame '{fid}' must be (4,4); got {H.shape}")
        H_list.append(torch.from_numpy(H))

    return torch.stack(H_list, dim=0)  # (B,4,4)


# ------------------------------ Mini-batch loader --------------------------- #

def _stack_optional_key(
    frames: List[Dict[str, Any]],
    key: str,
    *,
    as_torch: bool,
    device: Optional[torch.device],
    float_dtype: torch.dtype,
):
    """
    Stack a key across frames if present in the first frame. Returns None if absent.
    """
    if key not in frames[0]:
        return None

    vals = [f[key] for f in frames]
    # choose dtype
    if as_torch:
        dtype = float_dtype if key != "pix_idx" else torch.long
        return stack_arrays(vals, as_torch=True, device=device, dtype=dtype)
    else:
        return stack_arrays(vals, as_torch=False, device=None, dtype=None)


def load_batch(
    scene_dir: str,
    cfg_hash: str,
    split: str,
    frame_ids: List[str],
    seed: int,
    *,
    as_torch: bool = True,
    device: Optional[torch.device] = None,
    float_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    Load a set of frames (same cfg) and stack arrays along a new batch dim.

    Returns
    -------
    Dict[str, Any]
        Keys: H_wc (B,4,4), O,D,t,delta,X,sigma,T,w (B,R,⋯) when present,
              C (B,R) when present, width, height, pix_idx (R,), frame_ids, cfg_hash, seed.
    """
    paths = [frame_npz_path(scene_dir, cfg_hash, split, fid, seed) for fid in frame_ids]
    frames = [
        load_frame_npz(p, as_torch=as_torch, device=device, float_dtype=float_dtype)
        for p in paths
    ]
    if len(frames) == 0:
        raise ValueError("No frames loaded (empty frame_ids?).")

    # Shared metadata (assume identical for a cfg)
    width = int(frames[0]["width"])
    height = int(frames[0]["height"])
    out: Dict[str, Any] = {"width": width, "height": height, "frame_ids": frame_ids,
                           "cfg_hash": cfg_hash, "seed": int(seed)}

    # Stackables
    for k in ("O", "D", "t", "delta", "X", "sigma", "T", "w"):
        val = _stack_optional_key(frames, k, as_torch=as_torch, device=device, float_dtype=float_dtype)
        if val is not None:
            out[k] = val

    # C handled separately for clarity (shape: (B,R))
    if "C" in frames[0]:
        if as_torch:
            out["C"] = torch.stack([f["C"] for f in frames], dim=0)
        else:
            out["C"] = np.stack([f["C"] for f in frames], axis=0)

    # pix_idx shared (row-major 0..R-1)
    if "pix_idx" in frames[0]:
        out["pix_idx"] = frames[0]["pix_idx"]

    # poses (B,4,4)
    H_wc = get_H_wc_batch(scene_dir, frame_ids)
    out["H_wc"] = H_wc.to(device or (frames[0]["O"].device if as_torch and "O" in frames[0] else "cpu"))

    return out


# ------------------------------- Sanity checks ------------------------------ #

def validate_loaded_batch(batch: Dict[str, Any], atol: float = 1e-5) -> None:
    """
    Raise if shapes/invariants are off.

    Checks:
    - R == H*W when O present
    - D rows are unit-length when D present
    - t is non-decreasing along the last axis when t present
    """
    W, H = int(batch["width"]), int(batch["height"])
    R_expected = H * W

    if "O" in batch:
        B, R, C = batch["O"].shape
        assert C == 3, f"O last dim must be 3, got {C}"
        assert R == R_expected, f"R={R} but H*W={R_expected}"

    if "D" in batch:
        D = batch["D"].reshape(-1, 3)
        if torch.is_tensor(D):
            n = torch.linalg.norm(D, dim=-1)
            ok = torch.all(torch.abs(n - 1.0) < 1e-3)
        else:
            n = np.linalg.norm(D, axis=-1)
            ok = np.all(np.abs(n - 1.0) < 1e-3)
        assert bool(ok), "Non-unit directions in D"

    if "t" in batch:
        t = batch["t"]
        if torch.is_tensor(t):
            diffs = torch.diff(t, dim=-1)
            ok = torch.all(diffs >= -atol)
        else:
            diffs = np.diff(t, axis=-1)
            ok = np.all(diffs >= -atol)
        assert bool(ok), "t must be non-decreasing along last dim"


def _frame_id_index(scene_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict {frame_id: frame_dict} from transforms.json.
    """
    tr = load_transforms(scene_dir)
    frames = tr.get("frames", [])
    return {f["id"]: f for f in frames}

def _resolve_image_path(scene_dir: str, frame: Dict[str, Any]) -> str:
    """
    Resolve an image path from a frame entry in transforms.json.
    Supports either 'file_path' or nested 'image' objects commonly used.
    """
    # Common conventions: "file_path" or "image": {"file_path": ...}
    fp = frame.get("file_path")
    if fp is None and isinstance(frame.get("image"), dict):
        fp = frame["image"].get("file_path")
    if fp is None:
        raise KeyError(f"No image file path found for frame id '{frame.get('id','?')}'. "
                       "Expected 'file_path' or 'image.file_path' in transforms.json.")
    # Many manifests store a relative path; make absolute to scene_dir.
    abs_path = os.path.join(scene_dir, fp) if not os.path.isabs(fp) else fp
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image for frame '{frame.get('id','?')}' not found: {abs_path}")
    return abs_path

def _load_image(
    path: str,
    *,
    mode: str = "RGB",
    resize: Optional[Tuple[int, int]] = None,  # (W, H)
    as_torch: bool = True,
    chw: bool = True,
    normalize: bool = True,
    device: Optional[torch.device] = None,
    float_dtype: torch.dtype = torch.float32,
):
    """
    Load a single image from disk with optional resize and tensor conversion.
    Returns (img, (W, H)) where img is a tensor/ndarray and (W,H) are the final dims.
    """
    with Image.open(path) as im:
        if mode is not None:
            im = im.convert(mode)
        if resize is not None:
            # Pillow expects (W, H)
            im = im.resize(resize, Image.BILINEAR)
        W, H = im.size
        arr = np.array(im)  # (H,W,C) uint8

    if as_torch:
        t = torch.from_numpy(arr)  # uint8
        if normalize:
            t = t.to(device or torch.device("cpu"), dtype=float_dtype) / 255.0
        else:
            t = t.to(device or torch.device("cpu"))
        # (H,W,C) -> (C,H,W) if requested
        if chw:
            t = t.permute(2, 0, 1).contiguous()
        return t, (W, H)
    else:
        if normalize:
            arr = arr.astype(np.float32) / 255.0
        return arr, (W, H)

def load_batch_simple(
    scene_dir: str,
    frame_ids: List[str],
    *,
    resize: Optional[Tuple[int, int]] = None,   # (W, H) — set to ensure uniform shapes
    color_mode: str = None,                    # None keeps source mode
    chw: bool = False,                           # True: (B,C,H,W); False: (B,H,W,C)
    as_torch: bool = True,
    device: Optional[torch.device] = None,
    float_dtype: torch.dtype = torch.float32,
    normalize: bool = True,                     # True -> float in [0,1]; False -> uint8
) -> Dict[str, Any]:
    """
    Minimal loader that returns only camera poses and raw images.

    Args
    ----
    scene_dir : str
        Root directory containing transforms.json and images.
    frame_ids : List[str]
        Frame identifiers to load (must exist in transforms.json).
    resize : Optional[(W,H)]
        If provided, all images are resized to (W,H). Recommended for batching.
    color_mode : str
        Pillow convert() mode (e.g., 'RGB', 'L', None to keep original).
    chw : bool
        If True, images stacked as (B,C,H,W). Otherwise (B,H,W,C).
    as_torch : bool
        Return tensors on `device` (else numpy arrays).
    normalize : bool
        If True, outputs float in [0,1]. If False and as_torch, dtype=uint8.

    Returns
    -------
    Dict[str, Any]
        {
          "H_wc": (B,4,4) torch.Tensor,
          "images": (B,C,H,W) or (B,H,W,C) tensor/ndarray,
          "image_paths": List[str],
          "width": int,     # final width
          "height": int,    # final height
          "frame_ids": List[str],
        }

    Notes
    -----
    - If images have differing sizes and `resize` is None, this function raises.
      Provide `resize=(W,H)` to enforce uniform dimensions.
    """
    if len(frame_ids) == 0:
        raise ValueError("Empty frame_ids list.")

    # Resolve metadata for requested frames
    idx = _frame_id_index(scene_dir)
    missing = [fid for fid in frame_ids if fid not in idx]
    if missing:
        raise KeyError(f"frame_ids not found in transforms.json: {missing}")

    frames = [idx[fid] for fid in frame_ids]
    image_paths = [_resolve_image_path(scene_dir, f) for f in frames]

    # Load poses using existing helper (returns torch.Tensor)
    H_wc = get_H_wc_batch(scene_dir, frame_ids)  # (B,4,4)
    if device is not None:
        H_wc = H_wc.to(device)

    # Load images
    imgs = []
    sizes = []
    for p in image_paths:
        img, (W, H) = _load_image(
            p,
            mode=color_mode,
            resize=resize,
            as_torch=as_torch,
            chw=chw,
            normalize=normalize,
            device=device,
            float_dtype=float_dtype,
        )
        imgs.append(img)
        sizes.append((W, H))

    # Ensure uniform sizes (unless we resized)
    uniq_sizes = set(sizes)
    if len(uniq_sizes) != 1:
        hint = "Provide resize=(W,H) to enforce a common shape."
        raise ValueError(f"Images have differing sizes: {sorted(uniq_sizes)}. {hint}")

    W_final, H_final = sizes[0]

    # Stack
    if as_torch:
        images = torch.stack(imgs, dim=0)
    else:
        images = np.stack(imgs, axis=0)

    return {
        "H_wc": H_wc,                 # (B,4,4)
        "images": images,             # (B,C,H,W) or (B,H,W,C)
        "image_paths": image_paths,   # List[str]
        "width": int(W_final),
        "height": int(H_final),
        "frame_ids": frame_ids,
    }
