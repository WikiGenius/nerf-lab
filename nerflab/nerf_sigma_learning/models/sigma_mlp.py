import math
import torch
from torch import nn
from ..ops.posenc import PosEnc

class SigmaMLP(nn.Module):
    """
    Sigma-only NeRF MLP:
      - Input: 3D points (..., 3)
      - Output: sigma (...,) >= 0 via Softplus(density_bias + linear)
      - Positional encoding + optional skip connections.
    """
    def __init__(
        self,
        L_posenc: int = 8,
        hidden_dim: int = 64,
        depth: int = 4,
        skip_at=(2,),
        density_bias: float = -1.0,
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.pe = PosEnc(in_ch=3, L=L_posenc, include_input=True, scale=math.pi)
        in_dim = self.pe.out_dim
        self.skip_at = set(skip_at)
        self.act = nn.SiLU()

        layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (depth - 1)
        for i in range(depth - 1):
            inp = dims[i] + (in_dim if i in self.skip_at else 0)
            layers.append(nn.Linear(inp, dims[i + 1]))
        self.layers = layers

        sigma_inp = hidden_dim + (in_dim if (depth - 1) in self.skip_at else 0)
        self.sigma_linear = nn.Linear(sigma_inp, 1)
        self.softplus = nn.Softplus(beta=softplus_beta, threshold=20.0)
        self.register_buffer("density_bias", torch.tensor(float(density_bias)), persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.sigma_linear.weight)
        nn.init.zeros_(self.sigma_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3) -> sigma (...,)
        """
        orig = x.shape[:-1]
        x_flat = x.reshape(-1, 3)
        pe = self.pe(x_flat)
        h = pe
        for i, layer in enumerate(self.layers):
            if i in self.skip_at:
                h = torch.cat([h, pe], dim=-1)
            h = self.act(layer(h))
        if (len(self.layers)) in self.skip_at:
            h = torch.cat([h, pe], dim=-1)

        sigma_raw = self.sigma_linear(h).squeeze(-1)
        sigma = self.softplus(sigma_raw + self.density_bias)
        return sigma.reshape(*orig)
