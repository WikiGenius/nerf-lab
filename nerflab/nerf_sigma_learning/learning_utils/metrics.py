import math
from torch import nn

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    return 99.0 if mse <= 0 else 10.0 * math.log10((max_val ** 2) / mse)
