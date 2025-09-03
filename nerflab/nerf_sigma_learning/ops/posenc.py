import math
import torch
from torch import nn

class PosEnc(nn.Module):
    """
    NeRF-style Fourier features: [x, sin(2^k*pi*x), cos(2^k*pi*x)].
    """
    def __init__(self, in_ch=3, L=8, include_input=True, scale=math.pi):
        super().__init__()
        self.in_ch = in_ch
        self.L = L
        self.include_input = include_input
        self.scale = scale
        self.register_buffer("freqs", (2.0 ** torch.arange(L).float()) * self.scale, persistent=False)

    @property
    def out_dim(self):
        return (self.in_ch if self.include_input else 0) + self.in_ch * 2 * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.include_input:
            parts.append(x)
        xb = x.unsqueeze(-1) * self.freqs  # (..., in_ch, L)
        parts.append(torch.sin(xb).flatten(-2))
        parts.append(torch.cos(xb).flatten(-2))
        return torch.cat(parts, dim=-1)
