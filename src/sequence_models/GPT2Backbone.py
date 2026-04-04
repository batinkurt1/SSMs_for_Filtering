import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GPT2Backbone(nn.Module):
    """
    GPT-2 encoder with a token-wise multi-horizon head.

    forward(y, u=None, attn_mask=None) ->
        returns (B, L, H, p)
    """
    def __init__(
        self,
        p,
        m,
        H,
        d_model,
        n_layer,
        n_head,
        dropout,
        max_len,
    ):
        super().__init__()
        self.p, self.m, self.H = int(p), int(m), int(H)

        self.in_proj = nn.Linear(p + m, d_model)

        cfg = GPT2Config(
            n_positions=max_len,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.backbone = GPT2Model(cfg)

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, H * p),
        )

    def forward(
        self,
        y,
        u,
        attn_mask,
    ):
        """
        y: (B, L, p)
        u: (B, L, m) or None
        attn_mask: (B, L) with 1=real, 0=pad   (optional but recommended for padded windows)

        returns:
          (B, L, H, p) 
        """
        B, L, p = y.shape
        max_ctx = int(self.backbone.config.n_positions)

        if L > max_ctx:
            # Use the most recent context tokens when trajectory length exceeds GPT-2 context window.
            y = y[:, -max_ctx:, :]
            if u is not None:
                u = u[:, -max_ctx:, :]
            if attn_mask is not None:
                attn_mask = attn_mask[:, -max_ctx:]
            B, L, p = y.shape

        if self.m > 0:
            x = torch.cat([y, u], dim=-1)  # (B, L, p+m)
        else:
            x = y

        x = self.in_proj(x)  # (B, L, d_model)

        mask = None if attn_mask is None else attn_mask.to(device=x.device, dtype=torch.long)
        h = self.backbone(inputs_embeds=x, attention_mask=mask).last_hidden_state

        out = self.readout(h)                  # (B, L, H*p)
        out = out.view(B, L, self.H, self.p)   # (B, L, H, p)
        return out
    
