import numpy as np

def generate_random_ABC(n: int, m: int, p: int,
                        dist: str = "uniform",
                        target_rho: float = 0.95,
                        seed: int | None = None,
                        max_tries_A: int = 64,
                        max_tries_BC: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random (A, B, C) with A stable, (A,B) controllable, (A,C) observable.

    Inputs:
        n, m, p: dimensions (A: nxn, B: nxm, C: pxn)
        dist: "gaussian" (N(0,1)) or "uniform" (U[-1,1])
        target_rho: spectral radius to scale A to if rho(A) > 1 (default 0.95)
        seed: RNG seed
        max_tries_A: max resamples of A
        max_tries_BC: max resamples of (B,C) per A

    Returns:
        (A, B, C) numpy arrays.

    Raises:
        RuntimeError if it cannot find valid (A,B,C) within the try limits.
    """
    rng = np.random.default_rng(seed)

    def sample(shape):
        if dist.lower() in {"gaussian", "normal"}:
            return rng.standard_normal(shape)
        elif dist.lower() in {"uniform", "uni"}:
            return rng.uniform(-1.0, 1.0, size=shape)
        else:
            raise ValueError("dist must be 'gaussian' or 'uniform'")

    def spectral_radius(A):
        return float(np.max(np.abs(np.linalg.eigvals(A))))

    def ctrb_rank(A, B):
        # Controllability matrix [B, AB, ..., A^{n-1}B]
        A_pows = [np.eye(n)]
        for _ in range(1, n):
            A_pows.append(A_pows[-1] @ A)
        M = np.hstack([Ap @ B for Ap in A_pows])  # n × (n*m)
        return np.linalg.matrix_rank(M)

    def obsv_rank(A, C):
        # Observability matrix [C; CA; ...; CA^{n-1}]
        A_pows = [np.eye(n)]
        for _ in range(1, n):
            A_pows.append(A_pows[-1] @ A)
        M = np.vstack([C @ Ap for Ap in A_pows])  # (p*n) × n
        return np.linalg.matrix_rank(M)

    for _ in range(max_tries_A):
        # 1) Sample and stabilize A
        A = sample((n, n)).astype(float, copy=False)
        rho = spectral_radius(A)
        if rho > 1.0:
            A = (target_rho / rho) * A   # scale so rho(A) ~ target_rho

        # 2) Try sampling B,C until controllable & observable
        for _ in range(max_tries_BC):
            B = sample((n, m)).astype(float, copy=False)
            C = sample((p, n)).astype(float, copy=False)

            if ctrb_rank(A, B) == n and obsv_rank(A, C) == n:
                return A, B, C

    raise RuntimeError("Failed to find (A,B,C) with desired properties; try increasing max_tries or adjusting dims.")
