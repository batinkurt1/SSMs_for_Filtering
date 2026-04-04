import numpy as np
from helpers.cfg_loader import load_yaml

STATE_ORDER = ["x", "z", "phi", "vx", "vz", "phidot"]
CLIP_IDX = np.array([2, 3, 4, 5], dtype=int)


def _wrap_angle(phi: float) -> float:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def drone_drift(x, g):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 6:
        raise ValueError(f"Expected x in R^6, got shape {x.shape}")
    _, _, phi, vx, vz, phidot = x
    return np.array([
        vx * np.cos(phi) - vz * np.sin(phi),
        vx * np.sin(phi) + vz * np.cos(phi),
        phidot,
        vz * phidot - g * np.sin(phi),
        -vx * phidot - g * np.cos(phi),
        0.0,
    ], dtype=float)


def drone_input_matrix(m, l, J):
    return np.array([
        [0.0,   0.0],
        [0.0,   0.0],
        [0.0,   0.0],
        [0.0,   0.0],
        [1.0/m, 1.0/m],
        [ l/J, -l/J],
    ], dtype=float)


def clip_drone_state(x, limits, clip_state):
    x = np.asarray(x, dtype=float).reshape(-1)
    if not clip_state:
        return x
    x[CLIP_IDX] = np.clip(x[CLIP_IDX], limits[0, CLIP_IDX], limits[1, CLIP_IDX])
    return x


def build_limits_from_cfg(cfg):
    mins = np.array([cfg[f"{k}min"] for k in STATE_ORDER], dtype=float)
    maxs = np.array([cfg[f"{k}max"] for k in STATE_ORDER], dtype=float)
    return np.vstack([mins, maxs])

# ---------------- dynamics / measurement ----------------
def drone_state_transition_function(x, u):
    cfg = load_yaml("./configs/drone.yaml")  

    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    m  = float(cfg["m"])
    l  = float(cfg["l"])
    J  = float(cfg["J"])
    g  = float(cfg["g"])
    dt = float(cfg["dt"])

    limits = build_limits_from_cfg(cfg)
    f = drone_drift(x, g)
    B = drone_input_matrix(m, l, J)

    process_noise_mean = np.asarray(cfg["process_noise_mean"], dtype=float).reshape(-1)

    x_next = x + (f + B @ u) * dt + process_noise_mean
    x_next[2] = _wrap_angle(x_next[2])  # wrap phi
    return clip_drone_state(x_next, limits, bool(cfg["clip_state"]))

def drone_measurement_function(x):
    cfg = load_yaml("./configs/drone.yaml")
    x = np.asarray(x, dtype=float).reshape(-1)
    C = cfg["C"]
    measurement_noise_mean = np.asarray(cfg["measurement_noise_mean"], dtype=float).reshape(-1)
    if C is None:
        return x.copy()
    C = np.asarray(C, dtype=float)
    return C @ x + measurement_noise_mean


def make_drone_f_h(system_params: dict, sim_cfg: dict):
    """Build EKF-compatible drone dynamics/measurement functions for one trajectory."""
    C = np.asarray(system_params.get("C", sim_cfg["C"]), dtype=float)
    m = float(system_params.get("m", sim_cfg["m"]))
    l = float(system_params.get("l", sim_cfg["l"]))
    J = float(system_params.get("J", sim_cfg["J"]))
    g = float(system_params.get("g", sim_cfg["g"]))
    dt = float(system_params.get("dt", sim_cfg["dt"]))

    limits = build_limits_from_cfg(sim_cfg)
    clip_state = bool(sim_cfg["clip_state"])

    process_noise_mean = np.asarray(sim_cfg["process_noise_mean"], dtype=float).reshape(-1)
    measurement_noise_mean = np.asarray(sim_cfg["measurement_noise_mean"], dtype=float).reshape(-1)

    def f(x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if x.size != 6 or u.size != 2:
            raise ValueError("f expects x in R^6 and u in R^2")

        drift = drone_drift(x, g)
        B = drone_input_matrix(m, l, J)
        x_next = x + (drift + B @ u) * dt + process_noise_mean
        x_next[2] = _wrap_angle(x_next[2])
        return clip_drone_state(x_next, limits, clip_state)

    def h(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return C @ x + measurement_noise_mean

    return f, h
