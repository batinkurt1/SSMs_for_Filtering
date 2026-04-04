import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


class SelectiveSSMResidualLayer(nn.Module):
    def __init__(self, d_model, n_x, s_A, use_delta, fix_sA, dropout, device):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMBlock(
            d=d_model,
            n_x=n_x,
            s_A=s_A,
            use_delta=use_delta,
            fix_sA=fix_sA,
            device=device,
        )
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

    def forward(self, h):
        upd = self.ssm(self.norm(h))
        return h + self.dropout(upd)

class SelectiveSSMBlock(nn.Module):
    """
    The Selective State-Space Model (SSM) block.
    """
    
    def __init__(self, d, n_x, s_A, use_delta, fix_sA, device):
        """
        Initialize the SSMBlock.
        Args:
            d (int): Number of input channels.
            n_x (int): Number of states per channel.
            s_A (float): Stability margin of matrix A.
            use_delta (bool): Whether to use the discretization parameter.
            fix_sA (bool): Whether to fix the first state of the first channel to -s_A.
            device (torch.device): Device to be used.
        Returns:
            None 
        """
        super(SelectiveSSMBlock, self).__init__()

        self.d = d
        self.n_x = n_x
        self.s_A = s_A
        self.use_delta = use_delta
        self.fix_sA = fix_sA
        self.device = device

        # Initialize A as a (d)x(n_x) matrix where each row corresponds to the diagonal of an (n_x)x(n_x) matrix.
        if self.fix_sA:
            self.register_buffer('A', s_A*torch.ones(d, n_x, device=self.device))
        else:
            self.A = nn.Parameter(-10*torch.rand(d, n_x, device=self.device)+s_A)
            with torch.no_grad(): self.A[:,0] = s_A

        # Initialize W_B and W_C as projection weights for the input and output.
        self.W_B = nn.Parameter(torch.randn(n_x, d, device=self.device))
        self.W_C = nn.Parameter(torch.randn(n_x, d, device=self.device))

        # Initialize delta for the discretization parameter.
        self.q_delta = nn.Parameter(torch.randn(d, device=self.device))
        self.p_delta = nn.Parameter(torch.randn(1, device=self.device))
        
    def selective_scan(self, u, delta, A, B, C):
        """
        Perform the selective scan operation with optimized per-timestep discretization.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len, d).
            delta (torch.Tensor): Discretization parameter of shape (batch_size, seq_len).
            A (torch.Tensor): Matrix A of shape (d, n_x) which is the diaginal of the large (n_x*d, n_x*d) matrix.
            B (torch.Tensor): Matrix B of shape (batch_size, seq_len, n_x).
            C (torch.Tensor): Matrix C of shape (batch_size, seq_len, n_x).
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, d). 
        """
        
        batch_size, seq_len, d = u.shape
        n_x = A.shape[1]

        # Preallocate output tensor for direct writes
        y_out = torch.zeros((batch_size, seq_len, d), device=u.device, dtype=u.dtype)

        # Perform sequential state-space computation with per-step discretization
        x = torch.zeros((batch_size, d, n_x), device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            # Compute discretized matrices for this timestep only
            deltaA_t = torch.exp(einsum(delta[:, t], A, 'b, d n -> b d n'))  # (batch_size, d, n_x)
            deltaB_u_t = einsum(delta[:, t], B[:, t], u[:, t], 'b, b n, b d -> b d n')  # (batch_size, d, n_x)
            
            # Update state
            x = deltaA_t * x + deltaB_u_t
            
            # Compute and store output
            y_out[:, t] = einsum(x, C[:, t, :], 'b d n, b n -> b d')

        return y_out
    
    def clip_state_matrix(self, threshold=-1e-6):
        with torch.no_grad():
            self.A.clamp_(max=threshold)

    def forward(self, u):
        """
        Forward pass for the SSMBlock.
        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, seq_len, d).
        Returns:
            y (torch.Tensor): Output of the SSMBlock.
        """
        
        batch_size, seq_len, d = u.shape

        # Compute the discretization parameter \Delta if use_delta is enabled
        if self.use_delta:
            delta = self.p_delta + einsum(u, self.q_delta, 'b l d, d -> b l')  # Shape: (batch_size, seq_len)
            delta = F.softplus(delta)
        else:
            delta = torch.ones((batch_size, seq_len), device=u.device, dtype=u.dtype)  # Default to ones

        # Compute B and C using W_B and W_C
        B = einsum(self.W_B, u, 'n d, b l d -> b l n')  # Shape: (batch_size, seq_len, n_x)
        C = einsum(self.W_C, u, 'n d, b l d -> b l n')  # Shape: (batch_size, seq_len, n_x)

        # Perform selective scan
        y = self.selective_scan(u, delta, self.A, B, C)  # Shape: (batch_size, seq_len, d)
        
        return y
    
class SelectiveSSMBackbone(nn.Module):
    def __init__(
        self,
        p: int,
        m: int = 0,
        H: int = 1,
        d_model: int = 256,
        n_layer: int = 1,
        n_x: int = 16,              # number of SSM states per feature channel
        s_A: float = -0.1,
        use_delta: bool = True,
        fix_sA: bool = False,
        dropout: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.p, self.m, self.H = p, m, H
        self.n_layer = int(n_layer)

        # (p + m) -> d_model
        self.in_proj = nn.Linear(p + m, d_model)

        # one encoder layer: LN -> SelectiveSSM -> Dropout -> Residual
        self.norm = nn.LayerNorm(d_model)

        # IMPORTANT: your SelectiveSSM expects a 'device' arg at init time.
        # Passing CPU is fine; .to(device) will move everything later.
        self.ssm = SelectiveSSMBlock(
            d=d_model, n_x=n_x, s_A=s_A,
            use_delta=use_delta, fix_sA=fix_sA,
            device=device,
        )

        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

        # additional residual SSM layers (kept separate for backward compatibility with old checkpoints)
        self.extra_blocks = nn.ModuleList([
            SelectiveSSMResidualLayer(
                d_model=d_model,
                n_x=n_x,
                s_A=s_A,
                use_delta=use_delta,
                fix_sA=fix_sA,
                dropout=dropout,
                device=device,
            )
            for _ in range(self.n_layer - 1)
        ])

        # final norm + token-wise multi-horizon head
        self.norm_f = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, H * p),
        )

    def forward(
        self,
        y: torch.Tensor,                          # (B, L, p)
        u: torch.Tensor | None = None,            # (B, L, m) or None
        attn_mask: torch.Tensor | None = None,    # (B, L) with 1=real, 0=pad (optional)
    ) -> torch.Tensor:
        B, L, p = y.shape

        # concat exogenous if provided
        if self.m > 0:
            x = torch.cat([y, u], dim=-1)  # (B, L, p+m)
        else:
            x = y

        h = self.in_proj(x)  # (B, L, d_model)

        # zero-out padded tokens so they carry no signal
        if attn_mask is not None:
            h = h * attn_mask.to(h.dtype).unsqueeze(-1)

        # single pre-norm residual layer
        upd = self.ssm(self.norm(h))  # (B, L, d_model)
        h = h + self.dropout(upd)

        # additional pre-norm residual layers
        for blk in self.extra_blocks:
            h = blk(h)

        # final norm + readout
        h = self.norm_f(h)  # (B, L, d_model)
        out = self.readout(h).view(B, L, self.H, self.p)  # (B, L, H, p)
        return out