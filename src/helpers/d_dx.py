import numpy as np

def d_dx(h, x, eps=1e-7):
    """
    Numerical Jacobian H 
      h : function R^n -> R^m, takes 1D array x and returns 1D array y
      x : (n,) vector
      eps : step size for finite differences
    Returns:
      H : (m, n) Jacobian
    """
    x = np.asarray(x, dtype=float).reshape(-1)      # (n,)
    f0 = np.asarray(h(x), dtype=float).reshape(-1)  # (m,)
    m, n = f0.size, x.size
    H = np.zeros((m, n), dtype=float)

    for k in range(n):
        xk = x.copy()
        xk[k] += eps
        fk = np.asarray(h(xk), dtype=float).reshape(-1)
        H[:, k] = (fk - f0) / eps
    return H
