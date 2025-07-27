# viz/_config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class VizConfig:
    # ---------- figure & view -------------------------------------------------
    figsize: tuple[int, int] = (7, 6)
    dpi: int = 100
    axis_elev: float | None = 22.
    axis_azim: float | None = 35.
    axis_invert: tuple[str, ...] = ()           # e.g. ("x",) for left‑handed view

    # ---------- triad + grid --------------------------------------------------
    axis_triad_len: float = 0.8
    grid_lines: bool = False
    grid_bounds: tuple[tuple[float, float], ...] = ((-1, 7), (-1, 5), (-1, 4))
    grid_step: tuple[float, float, float] = (1.0, 1.0, 1.0)
    grid_res: tuple[int, int, int] = (80, 80, 60)
    grid_color: str = "#CCCCCC"
    grid_alpha: float = 0.6

    # ---------- shapes & alpha ------------------------------------------------
    default_alpha: float = 0.15

    # ---------- camera / ray defaults ----------------------------------------
    ray_step: int = 40
    ray_mode: str = "lines"          # "lines" | "quiver" | "points"
    camera_marker_size: int = 50

    # ---------- colour‑maps ---------------------------------------------------
    cmap: str = "viridis"

    # ───────── σ‑visualisation settings ──────
    scatter_size: int = 2            # px
    sigma_max_points: int = 20_000   # subsample cap
    heatmap_cmap: str = "magma"      # NEW
    heatmap_size: tuple[int, int] = (8, 4)  # NEW (figsize)

viz_cfg: VizConfig = VizConfig()
