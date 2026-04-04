import numpy as np
from pathlib import Path

from tqdm.auto import tqdm

from dynamical_models.drone import simulate_drone
from dynamical_models.lti_system import simulate_lti_system
from helpers.cfg_loader import load_yaml
from helpers.save_pickle import save_pickle


def simulate_system(system, n_traj, cfg):
    sim_fn = simulate_drone if system == "drone" else simulate_lti_system

    traj = []
    for _ in tqdm(range(n_traj), desc=f"Simulating {system}", unit="traj", dynamic_ncols=True):
        traj.append(sim_fn(cfg))

    states = np.stack([t["states"] for t in traj], axis=0)
    measurements = np.stack([t["measurements"] for t in traj], axis=0)
    inputs0 = traj[0].get("inputs", None)
    inputs = None if inputs0 is None else np.stack([t["inputs"] for t in traj], axis=0)

    return {
        "system": system,
        "n_traj": n_traj,
        "states": states,
        "measurements": measurements,
        "inputs": inputs,
        "meta": {"config": cfg},
    }


def save_results(results, outputs_dir):
    sys_name = results["system"]
    n_traj = results["n_traj"]
    n_steps = int(results["meta"]["config"]["n_steps"])

    input_enabled = results["inputs"] is not None
    cfg = results["meta"]["config"]
    randomized = bool(cfg["randomized_matrices"])
    case_name = str(cfg["case_name"])

    u_tag = "input" if input_enabled else "no_input"
    r_tag = "random_matrices" if randomized else "fixed_matrices"
    noise_tag = ""
    if case_name == "colored_noise":
        noise_tag = "-colored_noise"
    switch_tag = ""
    if case_name == "dynamics_switching":
        switch_tag = "-dynamics_switching"

    run_dir = Path(outputs_dir) / sys_name / f"N{n_traj}-S{n_steps}-{u_tag}-{r_tag}{noise_tag}{switch_tag}"
    out_path = run_dir / f"{sys_name}_dataset.pkl"
    saved = save_pickle(results, out_path)
    tqdm.write(f"Saved: {saved}")


def generate_training_data():
    cfg = load_yaml("configs/generate_data.yaml")
    system = cfg["system"]
    n_traj = cfg["n_traj"]
    outputs_dir = cfg["outputs_dir"]
    drone_cfg = load_yaml("configs/drone.yaml")
    base_lti_cfg = load_yaml("configs/lti_system.yaml")
    length_generalization_n_steps_lti = int(base_lti_cfg["length_generalization_n_steps"])
    colored_noise_sum_k_lti = int(base_lti_cfg["colored_noise_sum_k"])
    length_generalization_n_steps_drone = int(drone_cfg["length_generalization_n_steps"])
    colored_noise_sum_k_drone = int(drone_cfg["colored_noise_sum_k"])

    def build_case_cfgs(base_cfg, length_generalization_n_steps, colored_noise_sum_k):
        return [
            (
                "standard",
                {**base_cfg, "case_name": "standard", "n_steps": int(base_cfg["n_steps"]), "process_noise_mode": "iid", "measurement_noise_mode": "iid", "noise_sum_k": 1, "system_switch_mode": "none"},
                Path(outputs_dir),
            ),
            (
                "length_generalization",
                {**base_cfg, "case_name": "length_generalization", "n_steps": int(length_generalization_n_steps), "process_noise_mode": "iid", "measurement_noise_mode": "iid", "noise_sum_k": 1, "system_switch_mode": "none"},
                Path(outputs_dir),
            ),
            (
                "colored_noise",
                {**base_cfg, "case_name": "colored_noise", "n_steps": int(base_cfg["n_steps"]), "process_noise_mode": "sum_last_k", "measurement_noise_mode": "sum_last_k", "noise_sum_k": int(colored_noise_sum_k), "system_switch_mode": "none"},
                Path(outputs_dir),
            ),
            (
                "dynamics_switching",
                {**base_cfg, "case_name": "dynamics_switching", "n_steps": int(base_cfg["n_steps"]), "process_noise_mode": "iid", "measurement_noise_mode": "iid", "noise_sum_k": 1, "system_switch_mode": "dynamics_switching", "switch_step": int(base_cfg["n_steps"]) // 2},
                Path(outputs_dir),
            ),
        ]

    lti_case_cfgs = build_case_cfgs(base_lti_cfg, length_generalization_n_steps_lti, colored_noise_sum_k_lti)
    drone_case_cfgs = build_case_cfgs(drone_cfg, length_generalization_n_steps_drone, colored_noise_sum_k_drone)

    if system == "drone":
        for case_name, case_cfg, case_outputs_dir in drone_case_cfgs:
            tqdm.write(f"Generating drone data case: {case_name}")
            save_results(simulate_system("drone", n_traj, cfg=case_cfg), case_outputs_dir)
    elif system == "lti":
        for case_name, case_lti_cfg, case_outputs_dir in lti_case_cfgs:
            tqdm.write(f"Generating lti data case: {case_name}")
            save_results(simulate_system("lti", n_traj, cfg=case_lti_cfg), case_outputs_dir)
    elif system == "both":
        for case_name, case_cfg, case_outputs_dir in drone_case_cfgs:
            tqdm.write(f"Generating drone data case: {case_name}")
            save_results(simulate_system("drone", n_traj, cfg=case_cfg), case_outputs_dir)
        for case_name, case_lti_cfg, case_outputs_dir in lti_case_cfgs:
            tqdm.write(f"Generating lti data case: {case_name}")
            save_results(simulate_system("lti", n_traj, cfg=case_lti_cfg), case_outputs_dir)
