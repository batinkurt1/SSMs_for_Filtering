import numpy as np

class KalmanFilter:
    def __init__(self,
                 A, C,
                 initial_state_mean, initial_state_cov,
                 process_noise_mean, process_noise_cov,
                 measurement_noise_mean, measurement_noise_cov,
                 B=None, input_enabled=False):
        # system matrices
        self.A = np.asarray(A, dtype=float)
        self.B = None if B is None else np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)

        # noise parameters
        self.process_noise_mean = np.asarray(process_noise_mean, dtype=float)
        self.process_noise_cov  = np.asarray(process_noise_cov, dtype=float)
        self.measurement_noise_mean = np.asarray(measurement_noise_mean, dtype=float)
        self.measurement_noise_cov  = np.asarray(measurement_noise_cov, dtype=float)

        # initial belief
        self.initial_state_mean = np.asarray(initial_state_mean, dtype=float)
        self.initial_state_cov  = np.asarray(initial_state_cov, dtype=float)

        # flags
        self.input_enabled = bool(input_enabled)

        # cached results from last run
        self.estimated_states = None          # (n_steps+1, n)
        self.estimated_covariances = None     # (n_steps+1, n, n)

    
    def filter(self, n_steps,
               measurement_sequence,
               input_sequence=None):
        """
        Runs the same recursion as your function and stores results.
        Returns (estimated_states, estimated_covariances) with shapes:
          - states: (n_steps+1, n)
          - covs:   (n_steps+1, n, n)
        """
        A = self.A
        B = self.B
        C = self.C
        w_mean = self.process_noise_mean
        Q = self.process_noise_cov
        v_mean = self.measurement_noise_mean
        R = self.measurement_noise_cov

        y_seq = np.asarray(measurement_sequence, dtype=float)

        if self.input_enabled:
            if B is None or input_sequence is None:
                raise ValueError("input_enabled=True but B or input_sequence is None.")
            u_seq = np.asarray(input_sequence, dtype=float)
        else:
            u_seq = None  # ignored

        n = A.shape[0]
        estimated_states = np.empty((n_steps + 1, n), dtype=float)
        estimated_covariances = np.empty((n_steps + 1, n, n), dtype=float)

        for k in range(n_steps + 1):
            if k == 0:
                x_pred = self.initial_state_mean
                P_pred = self.initial_state_cov
            else:
                if self.input_enabled and B is not None and u_seq is not None:
                    u_k = u_seq[k - 1]
                    x_pred = A @ x_est + B @ u_k + w_mean
                else:
                    x_pred = A @ x_est + w_mean
                P_pred = A @ P_est @ A.T + Q

            y_k = y_seq[k]
            y_pred = C @ x_pred + v_mean
            S_k = C @ P_pred @ C.T + R
            PCt = P_pred @ C.T
            K_k = np.linalg.solve(S_k.T, PCt.T).T

            x_est = x_pred + K_k @ (y_k - y_pred)
            I_n = np.eye(n)
            I_KC = I_n - K_k @ C
            P_est = I_KC @ P_pred @ I_KC.T + K_k @ R @ K_k.T
            P_est = (P_est + P_est.T) / 2  # symmetrize

            estimated_states[k] = x_est
            estimated_covariances[k] = P_est

        self.estimated_states = estimated_states
        self.estimated_covariances = estimated_covariances
        return estimated_states, estimated_covariances

    def predict_next_H_outputs(self, H, last_state=None, future_inputs=None):
        if last_state is None:
            if self.estimated_states is None:
                raise RuntimeError("Run filter() first or pass last_state.")
            last_state = self.estimated_states[-1].copy()

        A = self.A
        B = self.B
        C = self.C
        w_mean = self.process_noise_mean
        v_mean = self.measurement_noise_mean

        x = np.asarray(last_state, dtype=float).reshape(-1)
        n = x.size
        p = C.shape[0]

        if self.input_enabled:
            if future_inputs is None:
                raise ValueError("input_enabled=True but future_inputs is None.")
            U = np.asarray(future_inputs, dtype=float)
            if U.shape[0] != H:
                raise ValueError(f"future_inputs must have length H; got {U.shape[0]} vs {H}")
        else:
            U = None

        x_future = np.empty((H, n), dtype=float)
        y_future = np.empty((H, p), dtype=float)

        for k in range(H):
            if self.input_enabled and B is not None and U is not None:
                x = A @ x + B @ U[k] + w_mean
            else:
                x = A @ x + w_mean
            y = C @ x + v_mean

            x_future[k] = x
            y_future[k] = y

        return y_future, x_future
