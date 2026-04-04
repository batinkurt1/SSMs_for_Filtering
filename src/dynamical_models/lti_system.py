import numpy as np

from helpers.generate_input_sequence import generate_input_sequence
from helpers.create_random_matrices import generate_random_ABC

class LTI_System:
    def __init__(self, initial_state):
        super().__init__()

        self.state = np.asarray(initial_state, dtype=float).reshape(-1)

    def __repr__(self):
        return ", ".join([f"x{i}: {xi:.3f}" for i, xi in enumerate(self.state)])

    
    def state_transition(self, A, B, u, process_noise_mean, process_noise_cov, process_noise=None):
        """
        A: shape (n,n)
        B: shape (n,m) or None
        u: control vector shape (m,) or None
        process_noise_mean: shape (n,)
        process_noise_cov: shape (n,n)
        """
        if process_noise is None:
            process_noise = np.random.multivariate_normal(
                mean=np.asarray(process_noise_mean, dtype=float).reshape(-1),
                cov=np.asarray(process_noise_cov, dtype=float))
        else:
            process_noise = np.asarray(process_noise, dtype=float).reshape(-1)

        if B is not None and u is not None:
            self.state = A @ self.state + B @ u + process_noise
        else:
            self.state = A @ self.state + process_noise
        
    def measurement_function(self, C, measurement_noise_mean, measurement_noise_cov, measurement_noise=None):
        """
        C: measurement matrix (p,n)
        measurement_noise_mean: shape (p,)
        measurement_noise_cov: shape (p,p)
        returns y: shape (p,)
        """
        if measurement_noise is None:
            v = np.random.multivariate_normal(
                mean=np.asarray(measurement_noise_mean, dtype=float).reshape(-1),
                cov=np.asarray(measurement_noise_cov, dtype=float))
        else:
            v = np.asarray(measurement_noise, dtype=float).reshape(-1)
        
        return C @ self.state + v

def _build_noise_sequence(n_steps, dim, mean, cov, mode="iid", sum_k=4):
        mean = np.asarray(mean, dtype=float).reshape(-1)
        cov = np.asarray(cov, dtype=float)
        if mode == "iid":
            return np.random.multivariate_normal(mean=mean, cov=cov, size=n_steps)
        if mode == "sum_last_k":
            # Draw a warmup prefix so each time index t uses exactly k terms,
            # including t=0 (interpreted as a sum over pre-history and current noise).
            k = max(1, int(sum_k))
            etas = np.random.multivariate_normal(mean=mean, cov=cov, size=n_steps + k - 1)
            out = np.empty((n_steps, dim), dtype=float)
            scale = 1.0 / np.sqrt(k)
            for t in range(n_steps):
                # Variance-normalized rolling sum with fixed window length k.
                out[t] = np.sum(etas[t:t + k], axis=0) * scale
            return out
        raise ValueError(f"Unknown noise mode '{mode}'. Use 'iid' or 'sum_last_k'.")


def _simulate_lti_system(n_steps, A, B, C, initial_state_mean, initial_state_cov,
                            process_noise_mean, process_noise_cov,
                            measurement_noise_mean, measurement_noise_cov, umin, umax, input_enabled=True,
                            process_noise_mode="iid", measurement_noise_mode="iid", noise_sum_k=4,
                            switch_mode="none", switch_step=None, A2=None, C2=None):
        """
        Returns a dict:
        {
            "states": (n_steps+1, n),
            "measurements": (n_steps+1, p)
            "inputs": (n_steps, m) or None
        }
        """
        # Initialize system
        initial_state = np.random.multivariate_normal(
            mean=np.asarray(initial_state_mean, dtype=float).reshape(-1),
            cov=np.asarray(initial_state_cov, dtype=float))
                                        
        lti_sys = LTI_System(initial_state=initial_state)
    
        # Preallocate
        n = A.shape[0]
        p = C.shape[0]
        states = np.empty((n_steps + 1, n), dtype=float)
        measurements = np.empty((n_steps + 1, p), dtype=float)

        is_dynamics_switching = switch_mode == "dynamics_switching"
        if is_dynamics_switching:
            if switch_step is None:
                switch_step = n_steps // 2
            switch_step = int(switch_step)
            if A2 is None or C2 is None:
                raise ValueError("switch_mode='dynamics_switching' requires A2 and C2.")
            A2 = np.asarray(A2, dtype=float)
            C2 = np.asarray(C2, dtype=float)
        else:
            switch_step = None
            A2 = None
            C2 = None

        def A_at_transition(k):
            if is_dynamics_switching and (k >= switch_step):
                return A2
            return A

        def C_at_measurement(k):
            if is_dynamics_switching and (k >= switch_step):
                return C2
            return C

        if input_enabled and B is None:
            raise ValueError("input_enabled=True requires B.")

        if input_enabled:
            m = B.shape[1]
            input_sequence = generate_input_sequence(
                n_steps,
                m,
                umin=umin,
                umax=umax,
                seed=None)

        proc_noise_seq = _build_noise_sequence(
            n_steps=n_steps,
            dim=n,
            mean=process_noise_mean,
            cov=process_noise_cov,
            mode=process_noise_mode,
            sum_k=noise_sum_k,
        )
        meas_noise_seq = _build_noise_sequence(
            n_steps=n_steps + 1,
            dim=p,
            mean=measurement_noise_mean,
            cov=measurement_noise_cov,
            mode=measurement_noise_mode,
            sum_k=noise_sum_k,
        )

        # t = 0
        states[0] = lti_sys.state
        measurements[0] = lti_sys.measurement_function(
            C_at_measurement(0), measurement_noise_mean, measurement_noise_cov,
            measurement_noise=meas_noise_seq[0],
        )
    
        # Steps k = 0...n_steps - 1
        for k in range(n_steps):
            lti_sys.state_transition(A=A_at_transition(k), B=B if input_enabled else None,
                u=input_sequence[k] if input_enabled else None,
                process_noise_mean=process_noise_mean,
                process_noise_cov=process_noise_cov,
                process_noise=proc_noise_seq[k],
            )
            states[k + 1] = lti_sys.state.copy()
            measurements[k + 1] = lti_sys.measurement_function(
                C_at_measurement(k + 1), measurement_noise_mean, measurement_noise_cov,
                measurement_noise=meas_noise_seq[k + 1],
            )
    
        return {
            "states": states,
            "measurements": measurements,
            "inputs": input_sequence if input_enabled else None,
            "system_params": {
                "A": np.asarray(A, dtype=float).copy(),
                "B": None if B is None else np.asarray(B, dtype=float).copy(),
                "C": np.asarray(C, dtype=float).copy(),
                "process_noise_mode": process_noise_mode,
                "measurement_noise_mode": measurement_noise_mode,
                "noise_sum_k": int(noise_sum_k),
                "switch_mode": switch_mode,
                "switch_step": switch_step,
                "A2": None if A2 is None else np.asarray(A2, dtype=float).copy(),
                "C2": None if C2 is None else np.asarray(C2, dtype=float).copy(),
            },
        }

def simulate_lti_system(cfg):
    switch_mode = str(cfg["system_switch_mode"])

    if cfg["randomized_matrices"]:
        A, B, C = generate_random_ABC(
            n=int(cfg["state_dimension"]),
            m=int(cfg["input_dimension"]),
            p=int(cfg["output_dimension"]),
            dist=cfg["matrix_dist"],
            target_rho=float(cfg["target_rho"]),
        )
    else: A, B, C = (np.asarray(cfg["A"], dtype=float),
                    np.asarray(cfg["B"], dtype=float) if bool(cfg["input_enabled"]) else None,
                    np.asarray(cfg["C"], dtype=float))

    switch_step = None
    A2 = None
    C2 = None
    if switch_mode == "dynamics_switching":
        n = int(cfg["state_dimension"])
        m = int(cfg["input_dimension"])
        p = int(cfg["output_dimension"])
        while True:
            A_try, _, C_try = generate_random_ABC(
                n=n,
                m=max(m, 1),
                p=p,
                dist=cfg["matrix_dist"],
                target_rho=float(cfg["target_rho"]),
            )
            if (not np.allclose(A_try, A)) and (not np.allclose(C_try, C)):
                A2, C2 = A_try, C_try
                break
        switch_step = int(cfg["switch_step"])
        switch_mode = "dynamics_switching"

    return _simulate_lti_system(
        n_steps=int(cfg["n_steps"]),
        A=A,
        B=B,
        C=C,
        initial_state_mean=cfg["initial_state_mean"],
        initial_state_cov=cfg["initial_state_cov"],
        process_noise_mean=cfg["process_noise_mean"],
        process_noise_cov=cfg["process_noise_cov"],
        measurement_noise_mean=cfg["measurement_noise_mean"],
        measurement_noise_cov=cfg["measurement_noise_cov"],
        umin=float(cfg["umin"]),
        umax=float(cfg["umax"]),
        input_enabled=bool(cfg["input_enabled"]),
        process_noise_mode=str(cfg["process_noise_mode"]),
        measurement_noise_mode=str(cfg["measurement_noise_mode"]),
        noise_sum_k=int(cfg["noise_sum_k"]),
        switch_mode=switch_mode,
        switch_step=switch_step,
        A2=A2,
        C2=C2,
    )

