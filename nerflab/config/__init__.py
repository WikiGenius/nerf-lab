"""
Config subpackage: dataclass configs and viz config. Keep runtime state in CFG.
"""

from .config import CFG, Cfg, IntrinsicsCfg, RaySampleCfg
from .viz_config import viz_cfg

__all__ = [
    "CFG", "Cfg", "IntrinsicsCfg", "RaySampleCfg",
    "viz_cfg",
]
