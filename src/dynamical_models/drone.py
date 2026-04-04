import numpy as np

from helpers.generate_input_sequence import generate_input_sequence
from dynamical_models.drone_functions import drone_drift
from dynamical_models.drone_functions import drone_input_matrix
from dynamical_models.drone_functions import clip_drone_state
from dynamical_models.drone_functions import build_limits_from_cfg


def _build_noise_sequence(n_steps, dim, mean, cov, mode="iid", sum_k=4):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if mode == "iid":
        return np.random.multivariate_normal(mean=mean, cov=cov, size=n_steps)
    if mode == "sum_last_k":
        k = max(1, int(sum_k))
        etas = np.random.multivariate_normal(mean=mean, cov=cov, size=n_steps + k - 1)
        out = np.empty((n_steps, dim), dtype=float)
        scale = 1.0 / np.sqrt(k)
        for t in range(n_steps):
            out[t] = np.sum(etas[t:t + k], axis=0) * scale
        return out
    raise ValueError(f"Unknown noise mode '{mode}'. Use 'iid' or 'sum_last_k'.")

class Drone:
    def __init__(self, initial_state, m, l, J, g, limits, clip_state):
        super().__init__()

        # Constants
        self.m = float(m)
        self.l = float(l)
        self.J = float(J)
        self.g = float(g)
        
        # Limits
        self.limits = np.asarray(limits, dtype=float)

        # Indices for clipping
        self.STATE_ORDER = ("x", "z", "phi", "vx", "vz", "phidot")
        self.CLIP_IDX = np.array([i for i, k in enumerate(self.STATE_ORDER) if k in {"phi","vx","vz","phidot"}], dtype=int)

        self.clip_state = clip_state

        # Initial state
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        if initial_state.size != 6:
            raise ValueError(f"initial_state must have 6 elements, got {initial_state.shape}")
        self.state = np.clip(initial_state, self.limits[0], self.limits[1])

    def __repr__(self):
        x, z, phi, vx, vz, phidot = self.state
        return (f"x: {x:.3f}, z: {z:.3f}, phi: {phi:.3f}, "
                f"vx: {vx:.3f}, vz: {vz:.3f}, phidot: {phidot:.3f}")

    def state_transition(self, u, dt, process_noise_mean, process_noise_cov, process_noise=None):
        """
        u: control vector shape (2,) [thrusts u1, u2]
        dt: float (seconds)
        process_noise_mean: shape (6,)
        process_noise_cov: shape (6,6)
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != 2:
            raise ValueError("u must have 2 elements (u1, u2)")
        if process_noise is None:
            w = np.random.multivariate_normal(
                mean=np.asarray(process_noise_mean, dtype=float).reshape(-1),
                cov=np.asarray(process_noise_cov, dtype=float))
        else:
            w = np.asarray(process_noise, dtype=float).reshape(-1)

        x, z, phi, vx, vz, phidot = self.state

        # Nonlinear dynamics f and input matrix B
        f = drone_drift(self.state, self.g)
        B = drone_input_matrix(self.m, self.l, self.J)

        next_state = self.state + (f + B @ u) * float(dt) + w

        next_state[2] = _wrap_angle(next_state[2])  # wrap phi

        next_state = clip_drone_state(next_state, self.limits, self.clip_state)

        self.state = next_state

    def measurement_function(self, C, measurement_noise_mean, measurement_noise_cov, measurement_noise=None):
        """
        C: measurement matrix (p,6)
        returns y: shape (p,)
        """
        C = np.asarray(C, dtype=float)
        if measurement_noise is None:
            v = np.random.multivariate_normal(
                mean=np.asarray(measurement_noise_mean, dtype=float).reshape(-1),
                cov=np.asarray(measurement_noise_cov, dtype=float),
            )
        else:
            v = np.asarray(measurement_noise, dtype=float).reshape(-1)
        return C @ self.state + v  

def _wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def _sample_drone_params(cfg, use_fixed=False):
    if use_fixed:
        return {
            "m": float(cfg["m"]),
            "l": float(cfg["l"]),
            "J": float(cfg["J"]),
            "C": np.asarray(cfg["C"], dtype=float),
        }

    return {
        "m": float(np.random.uniform(cfg["mmin"], cfg["mmax"])),
        "l": float(np.random.uniform(cfg["lmin"], cfg["lmax"])),
        "J": float(np.random.uniform(cfg["Jmin"], cfg["Jmax"])),
        "C": np.random.uniform(cfg["Cmin"], cfg["Cmax"], size=(3, 6)),
    }


def simulate_drone(cfg):
    limits = build_limits_from_cfg(cfg)

    process_noise_mode = str(cfg["process_noise_mode"])
    measurement_noise_mode = str(cfg["measurement_noise_mode"])
    noise_sum_k = int(cfg["noise_sum_k"])
    switch_mode = str(cfg["system_switch_mode"])
    is_dynamics_switching = switch_mode == "dynamics_switching"
    switch_step = None

    base_params = _sample_drone_params(cfg, use_fixed=not bool(cfg["randomized_matrices"]))
    switch_params = None
    if is_dynamics_switching:
        switch_params = _sample_drone_params(cfg, use_fixed=False)
        switch_step = int(cfg["switch_step"])

    proc_noise_seq = _build_noise_sequence(
        n_steps=int(cfg["n_steps"]),
        dim=6,
        mean=cfg["process_noise_mean"],
        cov=cfg["process_noise_cov"],
        mode=process_noise_mode,
        sum_k=noise_sum_k,
    )
    meas_noise_seq = _build_noise_sequence(
        n_steps=int(cfg["n_steps"]) + 1,
        dim=3,
        mean=cfg["measurement_noise_mean"],
        cov=cfg["measurement_noise_cov"],
        mode=measurement_noise_mode,
        sum_k=noise_sum_k,
    )

    def params_at_step(k):
        if is_dynamics_switching and k >= switch_step:
            return switch_params
        return base_params

    initial_state = np.random.multivariate_normal(
        mean=np.asarray(cfg["initial_state_mean"], dtype=float).reshape(-1),
        cov=np.asarray(cfg["initial_state_cov"], dtype=float),
    )

    drone = Drone(
        initial_state=initial_state,
        m=base_params["m"],
        l=base_params["l"],
        J=base_params["J"],
        g=float(cfg["g"]),
        limits=limits,
        clip_state=bool(cfg["clip_state"]),
    )

    p = base_params["C"].shape[0]
    n_steps = int(cfg["n_steps"])
    states = np.empty((n_steps + 1, 6), dtype=float)
    measurements = np.empty((n_steps + 1, p), dtype=float)
    input_sequence = generate_input_sequence(n_steps, 2, float(cfg["umin"]), float(cfg["umax"]), seed=None)

    states[0] = drone.state
    measurements[0] = drone.measurement_function(
        base_params["C"],
        cfg["measurement_noise_mean"],
        cfg["measurement_noise_cov"],
        measurement_noise=meas_noise_seq[0],
    )

    for k in range(n_steps):
        params = params_at_step(k)
        drone.m = params["m"]
        drone.l = params["l"]
        drone.J = params["J"]
        drone.state_transition(
            input_sequence[k],
            float(cfg["dt"]),
            cfg["process_noise_mean"],
            cfg["process_noise_cov"],
            process_noise=proc_noise_seq[k],
        )
        states[k + 1] = drone.state.copy()
        measurements[k + 1] = drone.measurement_function(
            params_at_step(k + 1)["C"],
            cfg["measurement_noise_mean"],
            cfg["measurement_noise_cov"],
            measurement_noise=meas_noise_seq[k + 1],
        )

    return {
        "states": states,
        "measurements": measurements,
        "inputs": input_sequence,
        "system_params": {
            "C": np.asarray(base_params["C"], dtype=float).copy(),
            "m": float(base_params["m"]),
            "l": float(base_params["l"]),
            "J": float(base_params["J"]),
            "g": float(cfg["g"]),
            "dt": float(cfg["dt"]),
            "process_noise_mode": process_noise_mode,
            "measurement_noise_mode": measurement_noise_mode,
            "noise_sum_k": int(noise_sum_k),
            "switch_mode": ("dynamics_switching" if is_dynamics_switching else switch_mode),
            "switch_step": (None if switch_step is None else int(switch_step)),
            "C2": None if switch_params is None else np.asarray(switch_params["C"], dtype=float).copy(),
            "m2": None if switch_params is None else float(switch_params["m"]),
            "l2": None if switch_params is None else float(switch_params["l"]),
            "J2": None if switch_params is None else float(switch_params["J"]),
        },
    }