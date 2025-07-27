# --------------------- config.py ---------------------
from dataclasses import dataclass

@dataclass(frozen=True)
class IntrinsicsCfg:
    fx: float = 1200
    fy: float = 1200
    width: int = 640
    height: int = 480
    # cx: float | None = None   # fill in later if your Intrinsics class needs it
    # cy: float | None = None

@dataclass(frozen=True)
class RaySampleCfg:
    t_near: float = 10
    # t_far:  float = 5.0
    t_far:  float = 15
    # N: int = 5
    N: int = 20
    deterministic: bool = False


# One top-level immutable bundle if you like
@dataclass(frozen=True)
class Cfg:
    intrinsics: IntrinsicsCfg = IntrinsicsCfg()
    rays: RaySampleCfg        = RaySampleCfg()
CFG = Cfg()