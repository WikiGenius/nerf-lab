# viz/_config.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

# @dataclass(frozen=True)
@dataclass
class VizConfig:
    """
    Global visualization defaults for world/camera/ray plots.

    Tip:
      Keep function parameters in plotting code defaulting to None, then
      fill from this config so you get consistent global styling with
      opt-in per-call overrides.
    """

    # ---------- figure & view -------------------------------------------------
    figsize: Tuple[int, int] = (7, 6)
    dpi: int = 100
    axis_elev: float | None = 22.0
    axis_azim: float | None = 35.0
    axis_invert: Tuple[str, ...] = ()   # e.g. ("x",) for left-handed view

    # ---------- triad + grid --------------------------------------------------
    axis_triad_len: float = 0.8
    grid_lines: bool = False
    grid_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-1.0, 7.0), (-1.0, 5.0), (-1.0, 4.0)
    )
    grid_step: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    grid_res: Tuple[int, int, int] = (80, 80, 60)
    grid_color: str = "#CCCCCC"
    grid_alpha: float = 0.6  # 0..1

    # ---------- shapes (defaults) --------------------------------------------
    default_alpha: float = 0.15  # fallback for primitives, 0..1

    # ---------- opacity / style controls -------------------------------------
    shape_face_alpha: float = 0.15    # boxes/spheres fill  (0..1)
    shape_edge_alpha: float = 0.90    # box wireframe edges (0..1)
    ray_alpha: float = 0.01           # rays (lines/quivers) (0..1)
    ray_linewidth: float = 0.80       # line thickness (pt)
    samples_alpha: float = 0.01       # sampled/external points (0..1)

    # ---------- camera / ray defaults ----------------------------------------
    ray_step: int = 8                 # pixel stride when not using rays_per_pose
    ray_mode: str = "lines"           # "lines" | "quiver"
    camera_marker_size: int = 50

    # ---------- colour-maps ---------------------------------------------------
    cmap: str = "viridis"

    # ---------- σ-visualisation ----------------------------------------------
    scatter_size: int = 2
    sigma_max_points: int = 20_000
    heatmap_cmap: str = "magma"
    heatmap_size: Tuple[int, int] = (8, 4)

    # ---------- utilities -----------------------------------------------------
    def clamped(self) -> "VizConfig":
        """Return a copy with all alpha fields clamped into [0,1]."""
        return replace(
            self,
            default_alpha=_clamp01(self.default_alpha),
            grid_alpha=_clamp01(self.grid_alpha),
            shape_face_alpha=_clamp01(self.shape_face_alpha),
            shape_edge_alpha=_clamp01(self.shape_edge_alpha),
            ray_alpha=_clamp01(self.ray_alpha),
            samples_alpha=_clamp01(self.samples_alpha),
        )

    def with_overrides(self, **kwargs) -> "VizConfig":
        """Create a modified copy (immutably)."""
        return replace(self, **kwargs).clamped()

# Singleton-style global config (imported as VCFG elsewhere)
viz_cfg: VizConfig = VizConfig().clamped()
