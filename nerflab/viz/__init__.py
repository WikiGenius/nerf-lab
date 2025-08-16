"""
Light‑weight 3‑D visualisation helpers for NeRF / geometry work.

Typical use
-----------
    from nerflab.viz import plot_world, plot_box, viz_cfg
    plot_world(world, cameras=[cam], draw_rays=True)
"""
from importlib import import_module
from types import ModuleType
from typing import Any

# re‑export all public names from sub‑modules
_PUBLIC = {
    "axes": ["set_3d_axes_equal", "draw_axis_triad", "draw_grid_3d"],
    "primitives": ["plot_box", "plot_sphere"],
    "viz_world": ["plot_world"],
    "viz_sigma": ["viz_sigma_scatter", "viz_sigma_heatmap"],
    "plot_cloud": ["plot_cloud_batch_with_camera"]
}

__all__ = ["viz_cfg"] + [name for names in _PUBLIC.values() for name in names]

from .viz_config import viz_cfg  # singleton

# --- lazy attribute loader --------------------------------------------------
def __getattr__(name: str) -> Any:                     # noqa: D401
    for mod_name, names in _PUBLIC.items():
        if name in names:
            module: ModuleType = import_module(f"{__name__}.{mod_name}")
            val = getattr(module, name)
            globals()[name] = val    # cache for future lookups
            return val
    raise AttributeError(f"{name} is not part of nerflab.viz’s public API")
