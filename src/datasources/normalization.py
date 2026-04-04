import numpy as np
import torch

def normalize_y_u(y, u=None, stats=None, eps=1e-6):
    """
    If stats is None (TRAIN): fits per-feature mean/std on y (N,T+1,p) and u (N,T,m),
    returns normalized arrays + fitted stats.
    If stats is provided (VAL/TEST): applies those stats, returns normalized arrays + same stats.
    """
    if stats is None:
        mu_y  = y.mean(axis=(0,1), dtype=np.float64)
        std_y = y.std (axis=(0,1), ddof=0, dtype=np.float64)
        std_y = np.maximum(std_y, eps)

        if u is not None:
            mu_u  = u.mean(axis=(0,1), dtype=np.float64)
            std_u = u.std (axis=(0,1), ddof=0, dtype=np.float64)
            std_u = np.maximum(std_u, eps)
        else:
            mu_u = std_u = None

        stats = {
            "mu_y":  mu_y.astype(np.float32),
            "std_y": std_y.astype(np.float32),
            "mu_u":  None if mu_u  is None else mu_u.astype(np.float32),
            "std_u": None if std_u is None else std_u.astype(np.float32),
            "eps":   float(eps),
        }

    y_norm = ((y - stats["mu_y"]) / stats["std_y"]).astype(np.float32)
    u_norm = None if u is None else ((u - stats["mu_u"]) / stats["std_u"]).astype(np.float32)
    return y_norm, u_norm, stats

def denorm_y(y_std, stats):
    """
    Inverse standardization for (..., p) shaped data.
    Works with torch.Tensor or np.ndarray.
    """
    mu, sd = stats["mu_y"], stats["std_y"]

    if torch.is_tensor(y_std):
        mu_t = y_std.new_tensor(mu)   # shape (p,)
        sd_t = y_std.new_tensor(sd)   # shape (p,)

        return y_std * sd_t + mu_t
    else:
        # NumPy path (mu, sd are already arrays of shape (p,))
        return y_std * sd + mu
