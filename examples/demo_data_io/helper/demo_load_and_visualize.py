#!/usr/bin/env python3
"""
Demo — Load cached frames & visualize.

What it does
------------
1) Loads a batch of frames from <scene>/rays/cache/<cfg_hash>/<split>/...
2) Validates invariants (R == H*W, unit-norm D, monotone t).
3) Visualizes:
   - World + selected camera using plot_world (optional)
   - sigma as a heatmap / scatter for a chosen frame
   - opacity mask using the binary renderer
4) Shows how to load a single NPZ as numpy or torch.

Assumes you already ran examples/demo_save_batch.py.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from nerflab import Intrinsics, Camera, CFG, load_world, plot_world
from nerflab.io import load_frame_npz, load_batch, validate_loaded_batch
from nerflab.viz.render import Renderer, BinaryRenderCfg
from nerflab.viz.plot_nonzero_sigma import plot_nonzero_sigma_row
from nerflab.viz.viz_sigma import viz_sigma_heatmap, viz_sigma_scatter  # adjust import if you keep separate
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load cached frames and visualize.")
    p.add_argument("--scene-dir", type=Path, default=Path("./demo_scene"),
                   help="Scene directory to read from.")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cfg-hash", type=str, default=None,
                   help="If not provided, derived from the first NPZ found under rays/cache/**/split.")
    p.add_argument("--num-frames", type=int, default=2, help="How many frames to load for the batch demo.")
    p.add_argument("--frame-offset", type=int, default=0, help="Starting frame index (when listing directory).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--idx-render", type=int, default=0, help="Index (within loaded batch) for per-frame visualizations.")
    p.add_argument("--show-world", action="store_true", help="Plot the world and selected camera.")
    return p.parse_args()


def _discover_cfg_hash(scene_dir: Path, split: str) -> str:
    cache_root = scene_dir / "rays" / "cache"
    for cfg_dir in cache_root.iterdir():
        if not cfg_dir.is_dir():
            continue
        if (cfg_dir / split).exists():
            return cfg_dir.name
    raise FileNotFoundError(f"No cfg_hash folder found under: {cache_root}/*/{split}")


def _list_frame_npz(scene_dir: Path, cfg_hash: str, split: str) -> list[Path]:
    d = scene_dir / "rays" / "cache" / cfg_hash / split
    if not d.exists():
        raise FileNotFoundError(f"Missing cache dir: {d}")
    return sorted(d.glob("*__seed*.npz"))


def _frame_ids_from_paths(paths: list[Path]) -> List[str]:
    """Extract frame_id from '<id>__seedX.npz'."""
    return [p.stem.split("__seed")[0] for p in paths]


def _camera_from_loaded_H(H_wc: torch.Tensor) -> Camera:
    """Build a (possibly batched) Camera using H_wc from the loaded batch."""
    intr = Intrinsics(**CFG.intrinsics.__dict__)
    cam = Camera(H_wc, intr, t_bounds=(CFG.rays.t_near, CFG.rays.t_far))
    return cam


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    scene_dir = args.scene_dir
    cfg_hash = args.cfg_hash or _discover_cfg_hash(scene_dir, args.split)

    # Choose frame IDs from existing NPZ files
    npz_paths_all = _list_frame_npz(scene_dir, cfg_hash, args.split)
    if len(npz_paths_all) == 0:
        raise RuntimeError("No NPZ files found. Run the save demo first.")

    npz_paths = npz_paths_all[args.frame_offset : args.frame_offset + args.num_frames]
    frame_ids = _frame_ids_from_paths(npz_paths)
    print(f"Loading cfg_hash={cfg_hash}, split={args.split}, frames={frame_ids}")

    # --- Single-frame load (numpy mode) ---------------------------------------
    npz0 = load_frame_npz(str(npz_paths[0]), as_torch=False)
    print(f"Single frame: O shape={npz0['O'].shape}, C shape={npz0['C'].shape}, width={npz0['width']}, height={npz0['height']}")

    # --- Same frame in torch mode (device-aware) ------------------------------
    t0 = load_frame_npz(str(npz_paths[0]), as_torch=True, device=device)
    assert torch.is_tensor(t0["O"]) and torch.is_tensor(t0["C"])

    # --- Batch load and validate ---------------------------------------------
    batch = load_batch(str(scene_dir), cfg_hash, args.split, frame_ids, seed=args.seed, as_torch=True, device=device)
    validate_loaded_batch(batch)
    print(f"Batched: H_wc={tuple(batch['H_wc'].shape)}, O={tuple(batch['O'].shape)}, C={tuple(batch['C'].shape)}")

    # --- Optional: plot the world and selected camera -------------------------
    if args.show_world:
        world = load_world("../data/world.json")  # adjust if elsewhere
        cam = _camera_from_loaded_H(batch["H_wc"])
        idx = max(0, min(args.idx_render, cam.B - 1))
        plot_world(
            world,
            cameras=cam,
            cam_indices=[idx],
            draw_rays=True,
            ray_mode="lines",
            rays_per_pose=100,
            samples_deterministic=False,
            ray_alpha=0.25,
            samples_alpha=0.1,
            shape_edge_alpha=1.0,
            shape_face_alpha=0.6,
            set_labels=False,
        )

    # --- Visualize sigma and opacity for a chosen frame -----------------------
    idx = max(0, min(args.idx_render, batch["C"].shape[0] - 1))
    H, W = int(batch["height"]), int(batch["width"])

    # Per-frame arrays (Torch tensors)
    t = batch["t"][idx]          # (R, N)
    sigma = batch["sigma"][idx]  # (R, N)
    C = batch["C"][idx]          # (R,)

    # ---- Optional scatter: needs X with shape (R, N, 3) ----
    # Convert to numpy for viz helpers
    t_np = t.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()

    # Downsample for scatter to keep it light
    # e.g., show at most ~5k rays and every-th sample along the ray
    R = H * W
    N = t_np.shape[-1]
    stride_r = max(1, R // 5000)    # adjust if you want fewer/more points
    stride_n = max(1, N // 16)

    if "X" in batch:
        X = batch["X"][idx]  # (R, N, 3)
        # Safety: ensure correct shape
        if X.ndim == 3 and X.shape[-1] == 3 and X.shape[:2] == sigma.shape:
            X_np = X.detach().cpu().numpy()
            X_sub = X_np[::stride_r, ::stride_n, :]     # (R', N', 3)
            sigma_sub = sigma_np[::stride_r, ::stride_n] # (R', N')

            # Your viz expects (R, N, 3) and (R, N)
            viz_sigma_scatter(X_sub, sigma_sub)  # no flattening
        else:
            print("Skipping scatter: X shape is not (R,N,3) matching sigma (R,N).")
    else:
        print("Skipping scatter: batch does not contain 'X' (store_points=False when saving).")

    # Heatmap over sigma (expects (R, N))
    viz_sigma_heatmap(sigma_np)

    # Opacity mask render
    rnd = Renderer(BinaryRenderCfg(threshold=0.5))
    fig, ax, _ = rnd.binary(
        C.reshape(H, W).detach().cpu().numpy(),
        intr=Intrinsics(**CFG.intrinsics.__dict__),
        title=f"Opacity frame {frame_ids[idx]}",
        save_path=str(scene_dir / f"opacity_{frame_ids[idx]}.png"),
    )
    try:
        plt.show()
    except Exception:
        pass
    # Example: pick a nonzero sigma row to inspect (torch or numpy both fine)
    r = plot_nonzero_sigma_row(t, sigma, strategy="max", mark_nonzero=True)
    print(f"Picked row: {r}")


if __name__ == "__main__":
    main()
