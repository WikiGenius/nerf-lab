#!/usr/bin/env python3
"""
Demo — Save batched ray/sigma data to a scene directory.

What it does
------------
1) Generates camera poses on a sphere (batched).
2) Builds a batched Camera with those poses.
3) Runs the end-to-end saver: writes NPZ per frame, updates transforms.json,
   and (optionally) renders binary opacity PNGs.

Requires
--------
- nerflab core: Intrinsics, Camera, CFG (provides intrinsics + rays cfg)
- nerflab.camera.pose_utils.iter_spherical_pose_batches
- nerflab.io.save_batch_frames (your refactored I/O)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import torch
from nerflab import Intrinsics, Camera, CFG  # assumed available in your package
from nerflab.camera.pose_utils import iter_spherical_pose_batches
from nerflab.io import save_batch_frames


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save batched rays/sigma to a scene folder.")
    p.add_argument("--scene-dir", type=Path, default=Path("../data/demo_scene"),
                   help="Output scene directory (will be created if missing).")
    p.add_argument("--scene-id", type=str, default="S_demo",
                   help="Identifier stored in transforms.json.")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Dataset split for images & cache folder structure.")
    p.add_argument("--total-poses", type=int, default=40,
                   help="Total number of poses to generate across all batches.")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Batch size for pose generation and saving.")
    p.add_argument("--radius", type=float, default=3.0,
                   help="Spherical camera radius.")
    p.add_argument("--pose-method", type=str, default="fibonacci",
                   help="Pose sampling method (e.g., 'fibonacci').")
    p.add_argument("--seed", type=int, default=7, help="Deterministic sampling seed.")
    p.add_argument("--render-png", action="store_true", help="Also render opacity PNGs.")
    p.add_argument("--device", type=str, default="cpu", help="torch device (cpu or cuda).")
    return p.parse_args()


def _make_camera_batched(H_wc: torch.Tensor) -> Camera:
    """
    Build a batched Camera using CFG intrinsics and the provided (B,4,4) poses.
    """
    intr = Intrinsics(**CFG.intrinsics.__dict__)
    cam = Camera(
        H_wc,
        intr,
        t_bounds=(CFG.rays.t_near, CFG.rays.t_far),
    )
    return cam


def _make_frame_ids(start_index: int, B: int) -> List[str]:
    """
    Simple zero-padded frame IDs for this batch.
    """
    return [f"{start_index + i:06d}" for i in range(B)]


def main() -> None:
    args = _parse_args()
    args.scene_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    total_written = 0
    start_index = 0

    # Stream poses in batches and save each batch.
    for H_wc in iter_spherical_pose_batches(
        c_total=args.total_poses,
        batch_size=args.batch_size,
        radius=args.radius,
        method=args.pose_method,
        device=device,
        dtype=torch.float32,
    ):
        cam = _make_camera_batched(H_wc)
        frame_ids = _make_frame_ids(start_index, H_wc.shape[0])

        npz_paths = save_batch_frames(
            scene_dir=str(args.scene_dir),
            scene_id=args.scene_id,
            split=args.split,
            camera=cam,
            CFG=CFG,
            frame_ids=frame_ids,
            seed=args.seed,
            store_points=True,
            render_png=args.render_png,
            binary_threshold=0.5,
            path_world="../data/world.json",  # adjust if your world lives elsewhere
        )

        total_written += len(npz_paths)
        start_index += len(npz_paths)
        print(f"[batch] wrote {len(npz_paths)} frames → total {total_written}")

    print(f"✅ Done. Scene written at: {args.scene_dir}")


if __name__ == "__main__":
    main()
