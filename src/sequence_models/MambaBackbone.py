import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))
    
class MambaBackbone(nn.Module):
    """
    Causal token-wise predictor.

    Inputs:
        y: (B, L, p)
        u: (B, L, m) or None

    Returns:
        out: (B, L, H, p)

    add mask to loss func
    """
    def __init__(
        self,
        p: int,
        m: int = 0,
        H: int = 1,
        d_model: int = 128,
        n_layer: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = int(p)
        self.m = int(m)
        self.H = int(H)

        self.in_proj = nn.Linear(self.p + self.m, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model)

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.H * self.p),
        )

    def forward(
        self,
        y: torch.Tensor,
        u: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        y: (B, L, p)
        u: (B, L, m) or None
        attn_mask: (B, L) with 1=real, 0=pad  (optional; if provided, pads are zeroed)
        returns: (B, L, H, p)
        """
        B, L, p = y.shape

        if self.m > 0:
            if u is None or u.shape != (B, L, self.m):
                got = None if u is None else tuple(u.shape)
                raise ValueError(f"Expected u of shape {(B, L, self.m)}, got {got}")
            x = torch.cat([y, u], dim=-1)
        else:
            x = y

        h = self.in_proj(x)  # (B, L, d)

        # Zero out padded tokens so they carry no signal
        if attn_mask is not None:
            mask = attn_mask.to(h.dtype).unsqueeze(-1)  # (B, L, 1)
            h = h * mask

        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)  # (B, L, d)

        out = self.readout(h).view(B, L, self.H, self.p)  # (B, L, H, p)
        return out
