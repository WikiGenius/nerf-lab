from importlib import import_module
from types import ModuleType
from typing import Any

"""
Lazy attribute loader for nerflab.learning

Expose public learning-related functions (e.g., forward_sigma) at the nerflab.learning namespace.
"""

# Map submodule names to the list of public attributes they export
_PUBLIC = {
    # forward_sigma.py defines forward_sigma
    "forward_sigma": ["forward_sigma"],
    "gathering_data": ["gathering_data"],
    "ops": [
        "compute_tau",
        "compute_transmittance",
        "compute_weights",
        "render_opacity",
        "render_depth",
        "sample_and_render",
        "get_batch_rays",        
    ],
    # in future, add other learning helpers under their module name
    # e.g. "train_utils": ["train_model", "evaluate"],
}

# Automatically build __all__ from the values in _PUBLIC
__all__ = [name for names in _PUBLIC.values() for name in names]


def __getattr__(name: str) -> Any:
    """
    Lazily import and return the requested attribute from its submodule.
    Raises AttributeError if the name is not in the public API.
    """
    for mod_name, names in _PUBLIC.items():
        if name in names:
            module = import_module(f"{__name__}.{mod_name}")
            val = getattr(module, name)
            globals()[name] = val  # cache for future lookups
            return val
    raise AttributeError(f"{name!r} is not part of {__name__!r} package")
