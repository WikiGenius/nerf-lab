# nerflab/io/dataset_save.py
from __future__ import annotations
import os, json, hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from ._common import ensure_dir, maybe_read_json, write_json, to_numpy
from ..learning.forward_sigma import nerf_opacity
from nerflab.world.world_json import query_density_field  # expects (B,R,N,3) or (R,N,3)


# ----------------------------- Config serialization ----------------------------- #

def _cfg_for_hash(CFG) -> Dict[str, Any]:
    """
    Canonicalize ray sampling settings for hashing.
    Note: Seed is intentionally excluded to keep folder stable across seeds.
    """
    r = CFG.rays
    return {
        "near": float(r.t_near),
        "far": float(r.t_far),
        "N": int(r.N),
        "deterministic": bool(r.deterministic),
        "num_fine": 0,
        "lindisp": False,
        "ray_sampling": "all_pixels",
    }


def _cfg_hash(CFG) -> str:
    """
    Stable short hash over the canonical ray settings.
    """
    canon = json.dumps(_cfg_for_hash(CFG), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


def write_cfg_json(scene_dir: str, CFG, *, seed: int) -> Tuple[str, str]:
    """
    Persist the canonicalized cfg (+seed field) under <scene>/rays/cfgs/<hash>.json.

    Returns
    -------
    (cfg_hash, json_path)
    """
    cfg_h = _cfg_hash(CFG)
    cfg_dir = os.path.join(scene_dir, "rays", "cfgs")
    ensure_dir(cfg_dir)
    jpath = os.path.join(cfg_dir, f"{cfg_h}.json")
    payload = _cfg_for_hash(CFG) | {"seed": int(seed)}
    write_json(jpath, payload)
    return cfg_h, jpath


# ------------------------------ Intrinsics & poses ------------------------------ #

def _intrinsics_dict(camera) -> Dict[str, Any]:
    """
    Serialize minimal pinhole intrinsics from a `camera.intr` object.
    """
    i = camera.intr
    cx = float(getattr(i, "cx", i.width * 0.5))
    cy = float(getattr(i, "cy", i.height * 0.5))
    return {
        "model": "PINHOLE",
        "width": int(i.width),
        "height": int(i.height),
        "fx": float(i.fx), "fy": float(i.fy),
        "cx": cx, "cy": cy,
        "distortion": None,
    }


def upsert_transforms_json(
    scene_dir: str,
    *,
    scene_id: str,
    camera,                    # nerflab.camera.Camera (per-frame pose passed)
    frame_id: str,
    split: str,
    image_rel_path: str,       # e.g. "images/train/000001.png"
    H_wc_frame: torch.Tensor,  # (4,4) for THIS frame
) -> str:
    """
    Create or update transforms.json for the given frame (idempotent per frame_id).
    Intrinsics consistency is enforced.
    """
    tpath = os.path.join(scene_dir, "transforms.json")
    manifest = maybe_read_json(tpath)

    if manifest is None:
        manifest = {
            "meta": {
                "scene_id": scene_id,
                "units": "meters",
                "coord_convention": "opengl",
                "image_axes": "u right, v down",
                "pose_matrix_order": "row-major-H_wc",
            },
            "camera": _intrinsics_dict(camera),
            "frames": [],
        }
    else:
        # verify intrinsics compatibility
        cam0 = manifest.get("camera", {})
        cam1 = _intrinsics_dict(camera)
        for k in ("width", "height", "fx", "fy", "cx", "cy"):
            if k in cam0 and abs(float(cam0[k]) - float(cam1[k])) > 1e-5:
                raise ValueError(f"intrinsics mismatch '{k}': stored={cam0[k]} new={cam1[k]}")

    H_list = to_numpy(H_wc_frame.to(dtype=torch.float32)).tolist()
    frames = manifest["frames"]
    idx = next((i for i, f in enumerate(frames) if f.get("id") == frame_id), None)
    frame_obj = {
        "id": frame_id,
        "split": split,
        "file_path": image_rel_path,
        "pose": {"type": "H", "from": "camera", "to": "world", "H_wc": H_list},
    }
    if idx is None:
        frames.append(frame_obj)
    else:
        frames[idx] = frame_obj

    write_json(tpath, manifest)
    return tpath


# ------------------------------ Batched frame save ----------------------------- #

@torch.no_grad()
def save_batch_frames(
    scene_dir: str,
    scene_id: str,
    split: str,
    *,
    camera,                       # nerflab.camera.Camera with H_wc: (B,4,4)
    CFG,
    frame_ids: List[str],         # len == B
    seed: int = 0,
    store_points: bool = True,
    points_dtype = np.float16,
    render_png: bool = True,
    binary_threshold: float = 0.5,
    path_world: str = "../data/world.json",
) -> List[str]:
    """
    End-to-end pipeline per batch:
      0) write cfg json (stable across seeds except the 'seed' field)
      1) generate all rays O,D for each frame
      2) sample along rays -> t, delta, X
      3) query density field -> sigma (B,R,N)
      4) opacity accumulation -> T, w, C
      5) save NPZ per frame + update transforms.json
      6) optionally render per-frame binary opacity PNG to images/<split>/<id>.png

    Returns
    -------
    List[str]: paths to the written NPZs (one per frame)
    """
    if not getattr(camera, "_is_batched", False):
        raise ValueError("Expected a batched Camera (H_wc with shape (B,4,4)).")

    B = int(camera.B)
    if len(frame_ids) != B:
        raise ValueError(f"len(frame_ids)={len(frame_ids)} must match batch size B={B}.")

    # 0) cfg
    cfg_hash, _ = write_cfg_json(scene_dir, CFG, seed=seed)

    # 1) rays
    O, D = camera.get_rays(frame="world", step=1, normalize=True)  # (B,R,3)
    _, R, _ = O.shape
    W, H = int(camera.intr.width), int(camera.intr.height)

    # 2) samples (coarse-only)
    gen = torch.Generator(device=O.device).manual_seed(int(seed))
    t, delta, X = camera.sample_along_rays(O, D, rng=gen, deterministic=None)  # (B,R,N),(B,R,N),(B,R,N,3)

    # 3) densities
    sigma = query_density_field(X, path_world=path_world)  # expect (B,R,N)
    if sigma is None:
        raise ValueError("`sigma` is required (shape (B,R,N) for batched camera).")
    if sigma.shape != t.shape:
        raise ValueError(f"`sigma` shape {tuple(sigma.shape)} must match t shape {tuple(t.shape)}")

    # 4) opacity (T,w: (B,R,N), C: (B,R))
    T, w, C = nerf_opacity(sigma, delta)

    # 5) write NPZ + transforms + (optional) render
    out_paths: List[str] = []
    pix_idx = torch.arange(R, device=O.device, dtype=torch.int64)  # row-major [0..R-1]

    if render_png:
        from nerflab.viz.render import Renderer, BinaryRenderCfg
        rnd = Renderer(BinaryRenderCfg(threshold=float(binary_threshold)))

    for b in range(B):
        frame_id = frame_ids[b]
        image_rel = f"images/{split}/{frame_id}.png"

        # transforms.json
        upsert_transforms_json(
            scene_dir, scene_id=scene_id, camera=camera,
            frame_id=frame_id, split=split, image_rel_path=image_rel, H_wc_frame=camera.H_wc[b]
        )

        arrays = {
            "O": to_numpy(O[b], np.float32),
            "D": to_numpy(D[b], np.float32),
            "t": to_numpy(t[b], np.float32),
            "delta": to_numpy(delta[b], np.float32),
            "X": to_numpy(X[b], points_dtype) if store_points else None,
            "sigma": to_numpy(sigma[b]),                     # keep input dtype
            "T": to_numpy(T[b], np.float32),
            "w": to_numpy(w[b], np.float32),
            "C": to_numpy(C[b], np.float32),                 # (R,)
            "pix_idx": to_numpy(pix_idx, np.int32),
            "seed": np.array(int(seed), dtype=np.int64),
            "width": np.array(W, dtype=np.int32),
            "height": np.array(H, dtype=np.int32),
            "frame_id": np.array(frame_id),
            "cfg_hash": np.array(cfg_hash),
        }
        arrays = {k: v for k, v in arrays.items() if v is not None}

        out_dir = os.path.join(scene_dir, "rays", "cache", cfg_hash, split)
        ensure_dir(out_dir)
        npz_path = os.path.join(out_dir, f"{frame_id}__seed{seed}.npz")
        np.savez_compressed(npz_path, **arrays)
        out_paths.append(npz_path)

        if render_png:
            # reshape C (R,) -> (H,W) row-major
            C_img = C[b].reshape(H, W)
            img_dir = os.path.join(scene_dir, "images", split)
            ensure_dir(img_dir)
            png_path = os.path.join(img_dir, f"{frame_id}.png")
            fig, ax, _ = rnd.binary(C_img, intr=camera.intr, title=f"Opacity {frame_id}", save_path=png_path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    return out_paths
