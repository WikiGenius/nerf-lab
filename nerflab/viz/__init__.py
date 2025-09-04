# nerflab/viz/__init__.py
"""
Light-weight 3-D visualization helpers for NeRF / geometry work.

Typical use
-----------
    from nerflab.viz import plot_world, plot_box, viz_cfg
    plot_world(world, cameras=[cam], draw_rays=True)
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Iterable, List, Set, Tuple, TYPE_CHECKING
from functools import lru_cache
import difflib

# ──────────────────────────────────────────────────────────────────────────────
# Public surface specification
# - Keys are *submodule names* relative to this package.
# - Values are the list of *public names* re-exported from that submodule.
#   (Keep in sync with the actual defs in each submodule.)
# ──────────────────────────────────────────────────────────────────────────────
_PUBLIC: Dict[str, List[str]] = {
    # NOTE: The submodule is `axis` (singular), not `axes`.
    "axis": [
        "style_3d_axis",      # (ax, ...) -> None
        "axis_triad",         # (ax, length=...) -> None
        "grid3d",             # (ax, bounds, step, ...) -> None
        # Optional legacy aliases (export if your module defines them):
        # "set_3d_axes_equal", "draw_axis_triad", "draw_grid_3d",
    ],
    "primitives": [
        "plot_box",           # (ax, box, ...)
        "plot_sphere",        # (ax, sphere, ...)
    ],
    "viz_world": [
        "plot_world",
    ],
    "viz_world_interactive": [
        "plot_world_interactive",
    ],
    "viz_sigma": [
        "viz_sigma_scatter",
        "viz_sigma_heatmap",
        # add "plot_nonzero_sigma_row" here if you export it in viz_sigma
    ],
    "plot_cloud": [
        "plot_cloud_batch_with_camera",
    ],
    "pose": [
        "draw_pose_axes",
    ],
    "render": [
        "Renderer",
        # export configs if desired, e.g. "BinaryRenderCfg"
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Eager export of the config singleton (small, used often)
# ──────────────────────────────────────────────────────────────────────────────
from ..config.viz_config import viz_cfg  # noqa: E402

# Build __all__ (order is stable and user-friendly)
__all__: List[str] = ["viz_cfg"] + [name for names in _PUBLIC.values() for name in names]

# Build an attribute -> submodule map and validate collisions
_ATTR_TO_MODULE: Dict[str, str] = {}
for mod_name, names in _PUBLIC.items():
    for n in names:
        if n in _ATTR_TO_MODULE and _ATTR_TO_MODULE[n] != mod_name:
            raise RuntimeError(
                f"Public name '{n}' is mapped to multiple submodules: "
                f"'{_ATTR_TO_MODULE[n]}' and '{mod_name}'. Please disambiguate."
            )
        _ATTR_TO_MODULE[n] = mod_name

# ──────────────────────────────────────────────────────────────────────────────
# Lazy import machinery
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _load_submodule(mod_name: str) -> ModuleType:
    """Import and cache a submodule in this package."""
    # __name__ is "nerflab.viz"; build "nerflab.viz.<mod_name>"
    return import_module(f"{__name__}.{mod_name}")

def __getattr__(name: str) -> Any:
    """Lazy attribute loader for the public API."""
    mod_name = _ATTR_TO_MODULE.get(name)
    if mod_name is None:
        # Build a helpful error with close matches
        candidates = difflib.get_close_matches(name, __all__, n=5, cutoff=0.6)
        hint = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
        raise AttributeError(
            f"'{name}' is not part of nerflab.viz’s public API.{hint}"
        )
    module = _load_submodule(mod_name)
    val = getattr(module, name)
    globals()[name] = val  # cache the resolved object in module globals
    return val

def __dir__() -> List[str]:
    """Make tab-completion show lazy exports."""
    return sorted(set(globals().keys()) | set(__all__))

# ──────────────────────────────────────────────────────────────────────────────
# Static typing / IDE support
# - During type checking, eagerly import the symbols so tools know their types.
#   This has zero runtime cost thanks to TYPE_CHECKING.
# ──────────────────────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    from .axis import style_3d_axis, axis_triad, grid3d
    from .primitives import plot_box, plot_sphere
    from .viz_world import plot_world
    from .viz_world_interactive import plot_world_interactive
    from .viz_sigma import viz_sigma_scatter, viz_sigma_heatmap
    from .plot_cloud import plot_cloud_batch_with_camera
    from .pose import draw_pose_axes
    from .render import Renderer
