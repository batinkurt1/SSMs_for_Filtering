import pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm, trange

from helpers.cfg_loader import load_yaml
from helpers.load_model import load_model
from dynamical_models.lti_system import simulate_lti_system
from dynamical_models.drone import simulate_drone
from dynamical_models.drone_functions import make_drone_f_h
from datasources.normalization import denorm_y

from estimators.KF import KalmanFilter
from estimators.EKF import ExtendedKalmanFilter


SUPPORTED_TEST_CASES = (
    "standard",
    "length_generalization",
    "colored_noise",
    "dynamics_switching",
)


def _build_case_cfg(sim_cfg, case_name, base_n_steps, length_generalization_n_steps, colored_noise_sum_k):
    case_cfg = deepcopy(sim_cfg)

    if case_name == "standard":
        case_cfg["n_steps"] = base_n_steps
        case_cfg["process_noise_mode"] = "iid"
        case_cfg["measurement_noise_mode"] = "iid"
        case_cfg["noise_sum_k"] = 1
        case_cfg["system_switch_mode"] = "none"
        case_cfg.pop("switch_step", None)
        return case_cfg

    if case_name == "length_generalization":
        case_cfg["n_steps"] = length_generalization_n_steps
        case_cfg["process_noise_mode"] = "iid"
        case_cfg["measurement_noise_mode"] = "iid"
        case_cfg["noise_sum_k"] = 1
        case_cfg["system_switch_mode"] = "none"
        case_cfg.pop("switch_step", None)
        return case_cfg

    if case_name == "colored_noise":
        case_cfg["n_steps"] = base_n_steps
        case_cfg["process_noise_mode"] = "sum_last_k"
        case_cfg["measurement_noise_mode"] = "sum_last_k"
        case_cfg["noise_sum_k"] = colored_noise_sum_k
        case_cfg["system_switch_mode"] = "none"
        case_cfg.pop("switch_step", None)
        return case_cfg

    if case_name == "dynamics_switching":
        case_cfg["n_steps"] = base_n_steps
        case_cfg["process_noise_mode"] = "iid"
        case_cfg["measurement_noise_mode"] = "iid"
        case_cfg["noise_sum_k"] = 1
        case_cfg["system_switch_mode"] = "dynamics_switching"
        case_cfg["switch_step"] = int(case_cfg["n_steps"]) // 2
        return case_cfg

    raise ValueError(
        f"Unknown test case '{case_name}'. Supported: {', '.join(SUPPORTED_TEST_CASES)}"
    )


def _select_case_model_dirs(case_name, dirs_by_case):
    case_model_run_dirs = dirs_by_case[case_name]
    if len(case_model_run_dirs) > 0:
        return case_model_run_dirs

    field_hint = {
        "standard": "model_run_dirs_standard",
        "length_generalization": "model_run_dirs_length_generalization",
        "colored_noise": "model_run_dirs_colored_noise",
        "dynamics_switching": "model_run_dirs_dynamics_switching",
    }[case_name]
    raise ValueError(f"No model dirs configured for case '{case_name}'. Set '{field_hint}' in configs/test.yaml.")


def test():
    # Read test config
    cfg = load_yaml("configs/test.yaml")
    system          = cfg["system"]                  # "lti" | "drone"
    n_traj          = int(cfg["n_traj"])
    H               = int(cfg["H"])                  # eval horizon
    L_test          = cfg["L"]             # int or None (full-context)

    standard_model_dirs = [Path(p) for p in cfg["model_run_dirs_standard"]]
    length_gen_model_dirs = [Path(p) for p in cfg["model_run_dirs_length_generalization"]]
    colored_noise_model_dirs = [Path(p) for p in cfg["model_run_dirs_colored_noise"]]
    switch_model_dirs = [Path(p) for p in cfg["model_run_dirs_dynamics_switching"]]
    save_dir        = Path(cfg["save_dir"])

    # Load sim config
    if system == "lti":
        sim_cfg = load_yaml(f"configs/{system}_system.yaml")
    else:  # "drone"
        sim_cfg = load_yaml(f"configs/{system}.yaml")
    base_n_steps = int(sim_cfg["n_steps"])
    length_generalization_n_steps = int(sim_cfg["length_generalization_n_steps"])
    colored_noise_sum_k = int(sim_cfg["colored_noise_sum_k"])

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define requested cases
    requested_cases = cfg["test_cases"]
    dirs_by_case = {
        "standard": standard_model_dirs,
        "length_generalization": length_gen_model_dirs,
        "colored_noise": colored_noise_model_dirs,
        "dynamics_switching": switch_model_dirs,
    }

    # Evaluate each case and save
    save_dir.mkdir(parents=True, exist_ok=True)
    for case_name in requested_cases:
        case_cfg = _build_case_cfg(
            sim_cfg=sim_cfg,
            case_name=case_name,
            base_n_steps=base_n_steps,
            length_generalization_n_steps=length_generalization_n_steps,
            colored_noise_sum_k=colored_noise_sum_k,
        )
        case_model_run_dirs = _select_case_model_dirs(case_name, dirs_by_case)

        case_out = evaluate_case(
            case_name=case_name,
            system=system,
            n_traj=n_traj,
            H=H,
            L_test=L_test,
            model_run_dirs=case_model_run_dirs,
            sim_cfg=case_cfg,
            device=device,
        )

        out_case = {
            "system": system,
            "H": H,
            "L": (None if L_test is None else int(L_test)),
            "n_traj": n_traj,
            **case_out,
        }

        out_path = save_dir / f"{system}_test_results_{case_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out_case, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"test saved: {out_path}")


def evaluate_case(case_name, system, n_traj, H, L_test, model_run_dirs, sim_cfg, device):
    print(f"\n[case] {case_name} | process_noise_mode={sim_cfg['process_noise_mode']} "
          f"measurement_noise_mode={sim_cfg['measurement_noise_mode']} "
          f"noise_sum_k={sim_cfg['noise_sum_k']} "
          f"switch_mode={sim_cfg['system_switch_mode']}")

    # Monte Carlo trajectories (shared between baseline and learned models for this case)
    trajs = []
    for _ in trange(n_traj, desc=f"Simulating trajectories [{case_name}]"):
        traj = simulate_lti_system(sim_cfg) if system == "lti" else simulate_drone(sim_cfg)
        trajs.append(traj)

    # Baseline RMS over trajectories: (n_traj, A, H)
    baseline_rms = []
    y_trajs_all = []
    traj_system_params_all = []

    for traj in tqdm(trajs, desc=f"Baseline (KF/EKF) [{case_name}]"):
        base_rms_ah, y = baseline(traj, sim_cfg, system, H)
        baseline_rms.append(base_rms_ah)
        y_trajs_all.append(y)
        traj_system_params_all.append(traj.get("system_params", {}))

    baseline_rms_arr = np.stack(baseline_rms, axis=0)               # (n_traj, A, H)
    baseline_rms_time = np.sqrt(np.clip(np.mean(np.square(baseline_rms_arr), axis=0), 0.0, None))

    y_trajs_all_arr = np.stack(y_trajs_all, axis=0)                 # (n_traj, T+1, p)

    # Evaluate each model separately and collect results
    rms_time_by_model = []
    for rd in tqdm(model_run_dirs, desc=f"Evaluating models [{case_name}]"):
        model, meta = load_model(rd, device=device)  # loader should NOT call eval()
        model.eval()

        name_for_bar = meta.get("type", rd.name)

        rms_by_traj = []

        for traj in tqdm(trajs, desc=f"Trajs for {name_for_bar}", leave=False):
            y = traj["measurements"].astype(np.float32)     # (T+1, p)
            u = traj.get("inputs", None)
            if u is not None:
                u = u.astype(np.float32)                    # (T, m)

            rms_ah = eval_model(model, meta, y, u, H, L_test)  # (A, H)
            rms_by_traj.append(rms_ah)

        rms_by_traj_arr = np.stack(rms_by_traj, axis=0)     # (n_traj, A, H)
        rms_time = np.sqrt(np.clip(np.mean(np.square(rms_by_traj_arr), axis=0), 0.0, None))

        rms_time_by_model.append({
            "name": meta["type"],
            "type": meta["type"],
            "run_dir": str(rd),
            "rms_time_h": rms_time.tolist(),                # (A, H)
            "rms_ah_by_traj": rms_by_traj_arr.tolist(),     # (n_traj, A, H)
        })

    case_out = {
        "case": case_name,
        "baseline": {
            "kind": ("KF" if system == "lti" else "EKF"),
            "rms_time_h": baseline_rms_time.tolist(),          # (A, H)
            "rms_ah_by_traj": baseline_rms_arr.tolist(),       # (n_traj, A, H)
        },
        "y_trajs": y_trajs_all_arr.tolist(),                   # (n_traj, T+1, p)
        "traj_system_params": traj_system_params_all,
        "sim_settings": {
            "process_noise_mode": sim_cfg["process_noise_mode"],
            "measurement_noise_mode": sim_cfg["measurement_noise_mode"],
            "noise_sum_k": int(sim_cfg["noise_sum_k"]),
            "system_switch_mode": sim_cfg["system_switch_mode"],
        },
    }
    type_counts = {}
    for model_res in rms_time_by_model:
        model_type = str(model_res.get("type", model_res["name"]))
        count = type_counts.get(model_type, 0) + 1
        type_counts[model_type] = count

        key = f"model_{model_type}" if count == 1 else f"model_{model_type}_{count}"
        case_out[key] = model_res

    return case_out


def eval_model(model, meta, y_raw, u_raw, H_eval, L_test):
    T = y_raw.shape[0] - 1
    p = y_raw.shape[-1]
    anchors = range(1, T - H_eval + 2)
    A = T - H_eval + 1

    stats = model.norm_stats
    dev = next(model.parameters()).device  # ensure inputs go to same device as the model

    # store RMS per anchor and horizon
    rms_ah = np.empty((A, H_eval), dtype=np.float32)

    for idx, t0 in enumerate(anchors):
        if L_test is None:
            # Full context up to t0
            y_hist = y_raw[0:t0, :]

            y_hist_n = (y_hist - stats["mu_y"]) / stats["std_y"]


            u_hist_n = None
            if u_raw is not None:
                u_hist = u_raw[0:t0, :]
                u_hist_n = (u_hist - stats["mu_u"]) / stats["std_u"]

            mask = np.ones(y_hist_n.shape[0], dtype=np.int64)
        else:
            # Fixed L window with left padding
            L = int(L_test)
            start = max(0, t0 - L)
            k = t0 - start          # length of actual context

            # --- measurements ---
            y_ctx = y_raw[start:t0, :]                              # (k, p)
            y_ctx_n = (y_ctx - stats["mu_y"]) / stats["std_y"]      # normalize first

            if k < L:
                pad_n = np.zeros((L - k, p), dtype=np.float32)      # zeros in normalized space
                y_hist_n = np.concatenate([pad_n, y_ctx_n], axis=0) # (L, p)
                mask = np.concatenate([np.zeros(L - k, dtype=np.int64),
                                        np.ones(k, dtype=np.int64)], axis=0)
            else:
                y_hist_n = y_ctx_n
                mask = np.ones(L, dtype=np.int64)

            # inputs (if any)
            u_hist_n = None
            if u_raw is not None:
                m = u_raw.shape[-1]
                u_ctx = u_raw[start:t0, :]                          # (k, m)
                u_ctx_n = (u_ctx - stats["mu_u"]) / stats["std_u"]
                if k < L:
                    upad_n = np.zeros((L - k, m), dtype=np.float32) # zeros in normalized space
                    u_hist_n = np.concatenate([upad_n, u_ctx_n], axis=0)
                else:
                    u_hist_n = u_ctx_n

        # Tensors to model device
        y_t = torch.from_numpy(y_hist_n).float().unsqueeze(0).to(dev)  # (1, L*, p)
        u_t = None if u_hist_n is None else torch.from_numpy(u_hist_n).float().unsqueeze(0).to(dev)
        m_t = torch.from_numpy(mask).long().unsqueeze(0).to(dev)       # (1, L*)

        with torch.no_grad():
            pred = model.backbone(y_t, u_t, attn_mask=m_t)             # (1, L*, H, p)
            pred_last = pred[:, -1, :, :].squeeze(0).cpu().numpy()     # (H, p)
        pred_last = pred_last[:H_eval]

        pred_den = denorm_y(pred_last, stats)                          # (H, p)
        tgt = y_raw[t0:t0 + H_eval, :]                                 # (H, p)

        rms_ah[idx] = np.sqrt(np.mean((pred_den - tgt) ** 2, axis=-1))

    return rms_ah


def baseline(traj, sim_cfg, system, H_eval):
    y = traj["measurements"].astype(np.float32)  # (T+1, p)
    u = traj.get("inputs", None)
    if u is not None:
        u = u.astype(np.float32)

    T = y.shape[0] - 1
    p = y.shape[-1]
    anchors = range(1, T - H_eval + 2)
    A = T - H_eval + 1

    rms_ah = np.empty((A, H_eval), dtype=np.float32)

    if system == "lti":
        sys_params = traj.get("system_params", {})
        A_mat = np.asarray(sys_params.get("A", sim_cfg["A"]), dtype=float)
        B_raw = sys_params.get("B", sim_cfg["B"]) if bool(sim_cfg["input_enabled"]) else None
        B_mat = None if B_raw is None else np.asarray(B_raw, dtype=float)
        C_mat = np.asarray(sys_params.get("C", sim_cfg["C"]), dtype=float)

        kf = KalmanFilter(
            A=A_mat, C=C_mat,
            initial_state_mean=sim_cfg["initial_state_mean"],
            initial_state_cov=sim_cfg["initial_state_cov"],
            process_noise_mean=sim_cfg["process_noise_mean"],
            process_noise_cov=sim_cfg["process_noise_cov"],
            measurement_noise_mean=sim_cfg["measurement_noise_mean"],
            measurement_noise_cov=sim_cfg["measurement_noise_cov"],
            B=B_mat,
            input_enabled=bool(sim_cfg["input_enabled"]),
        )
        kf.filter(n_steps=T, measurement_sequence=y, input_sequence=u)

        for idx, t0 in enumerate(anchors):
            x_t = kf.estimated_states[t0 - 1].copy()
            fut_u = u[t0 - 1: t0 - 1 + H_eval] if (u is not None) else None
            y_fore, _ = kf.predict_next_H_outputs(H=H_eval, last_state=x_t, future_inputs=fut_u)  # (H, p)
            tgt = y[t0:t0 + H_eval, :]
            rms_ah[idx] = np.sqrt(np.mean((y_fore - tgt) ** 2, axis=-1))

    else:  # "drone"
        sys_params = traj.get("system_params", {})
        f, h = make_drone_f_h(sys_params, sim_cfg)
        ekf = ExtendedKalmanFilter(
            f=f,
            h=h,
            initial_state_mean=sim_cfg["initial_state_mean"],
            initial_state_cov=sim_cfg["initial_state_cov"],
            process_noise_mean=sim_cfg["process_noise_mean"],
            process_noise_cov=sim_cfg["process_noise_cov"],
            measurement_noise_mean=sim_cfg["measurement_noise_mean"],
            measurement_noise_cov=sim_cfg["measurement_noise_cov"],
            input_enabled=bool(sim_cfg["input_enabled"]),
        )

        ekf.filter(n_steps=T, measurement_sequence=y, input_sequence=u)

        for idx, t0 in enumerate(anchors):
            x_t = ekf.estimated_states[t0 - 1].copy()
            fut_u = u[t0 - 1: t0 - 1 + H_eval] if (u is not None) else None
            y_fore, _ = ekf.predict_next_H_outputs(H=H_eval, last_state=x_t, future_inputs=fut_u)
            tgt = y[t0:t0 + H_eval, :]
            rms_ah[idx] = np.sqrt(np.mean((y_fore - tgt) ** 2, axis=-1))

    return rms_ah, y


if __name__ == "__main__":
    test()
