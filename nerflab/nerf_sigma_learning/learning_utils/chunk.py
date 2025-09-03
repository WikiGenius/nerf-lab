import torch
from torch import nn

def forward_chunked_sigma(model: nn.Module, pts: torch.Tensor, chunk_rays: int = 32768) -> torch.Tensor:
    """
    pts: (B, R, N, 3) -> (B, R, N) — processes rays in chunks for memory safety.
    """
    B, R, N, _ = pts.shape
    flat = pts.reshape(B * R, N, 3)
    out = []
    for i in range(0, B * R, chunk_rays):
        sl = flat[i : i + chunk_rays]
        out.append(model(sl))
    return torch.cat(out, dim=0).reshape(B, R, N)
