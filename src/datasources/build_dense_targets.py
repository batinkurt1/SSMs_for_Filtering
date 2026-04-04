import numpy as np

def build_dense_targets(seq, H, start=0, L=None):
    """
    Build per-position next-H targets.

    seq   : (T+1, p) containing y_0..y_T
    H     : horizon (int)
    start : index of the history end (0 means history ends at y_0)
    L     : number of history positions to generate. If None, uses full context:
            L = (T - start) - H + 1

    Returns:
      targets: (L, H, p) where for i in [0..L-1],
               s = start + i + 1 and targets[i] = seq[s : s+H, :]
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0] - 1
    if L is None:
        L = T - start - H + 1
    if L <= 0:
        raise ValueError(f"Invalid (start={start}, H={H}) for sequence length T={T}. Got L={L} less than or equal to 0.")
    p = seq.shape[-1]
    out = np.empty((L, H, p), dtype=np.float32)
    for i in range(L):
        s = start + i + 1
        out[i] = seq[s:s+H, :]
    return out