import numpy as np
from helpers.d_dx import d_dx

class ExtendedKalmanFilter:
    def __init__(self,
                 f, h,
                initial_state_mean, initial_state_cov,
                 process_noise_mean, process_noise_cov,
                 measurement_noise_mean, measurement_noise_cov,
                input_enabled):
        
        self.f = f
        self.h = h
        self.input_enabled = bool(input_enabled)

        self.initial_state_mean = np.asarray(initial_state_mean, dtype=float)
        self.initial_state_cov  = np.asarray(initial_state_cov,  dtype=float)

        self.w_mean = np.asarray(process_noise_mean, dtype=float)
        self.Q      = np.asarray(process_noise_cov,  dtype=float)
        self.v_mean = np.asarray(measurement_noise_mean, dtype=float)
        self.R      = np.asarray(measurement_noise_cov,  dtype=float)

        self.estimated_states = None          # (T+1, n)
        self.estimated_covariances = None     # (T+1, n, n)
 
    def f_call(self, x, u):
        if self.input_enabled and u is not None:
            return self.f(x, u)
        return self.f(x)

    def A_jacobian(self, x, u):
        # Jacobian wrt x of f(x,u) at (x,u). If f ignores u, this still works.
        if self.input_enabled:
            return d_dx(lambda z: self.f(z, u), x)
        return d_dx(self.f, x)

    def C_jacobian(self, x):
        return d_dx(self.h, x)

    def filter(self, n_steps, measurement_sequence, input_sequence=None):
        """
        EKF filtering pass.
        """
        y_seq = np.asarray(measurement_sequence, dtype=float)

        if self.input_enabled:
            if input_sequence is None:
                raise ValueError("input_enabled=True but input_sequence is None.")
            U = np.asarray(input_sequence, dtype=float)
        else:
            U = None

        x0 = self.initial_state_mean.reshape(-1)
        P0 = self.initial_state_cov
        n  = x0.size
        I  = np.eye(n)

        est_states = np.empty((n_steps + 1, n), dtype=float)
        est_covs   = np.empty((n_steps + 1, n, n), dtype=float)

        x_est = None
        P_est = None

        for k in range(n_steps + 1):
            if k == 0:
                x_pred = x0
                P_pred = P0
            else:
                u_km1 = U[k - 1] if self.input_enabled else None
                x_pred = self.f_call(x_est, u_km1) + self.w_mean
                A_k = self.A_jacobian(x_est, u_km1)
                P_pred = A_k @ P_est @ A_k.T + self.Q

            C_k   = self.C_jacobian(x_pred)
            y_pred= self.h(x_pred) + self.v_mean
            y_k   = y_seq[k]

            S_k = C_k @ P_pred @ C_k.T + self.R
            PCt = P_pred @ C_k.T
            K_k = np.linalg.solve(S_k.T, PCt.T).T

            x_est = x_pred + K_k @ (y_k - y_pred)
            I_KC = I - K_k @ C_k
            P_est = I_KC @ P_pred @ I_KC.T + K_k @ self.R @ K_k.T
            P_est = 0.5 * (P_est + P_est.T)

            est_states[k] = x_est
            est_covs[k]   = P_est

        self.estimated_states = est_states
        self.estimated_covariances = est_covs
        return est_states, est_covs

    def predict_next_H_outputs(self, H, last_state=None, future_inputs=None):
        if last_state is None:
            if self.estimated_states is None:
                raise RuntimeError("Run filter() first or pass last_state.")
            x = self.estimated_states[-1].copy()
        else:
            x = np.asarray(last_state, dtype=float).reshape(-1)

        if self.input_enabled:
            if future_inputs is None:
                raise ValueError("input_enabled=True but future_inputs is None.")
            U = np.asarray(future_inputs, dtype=float)
            if U.shape[0] != H:
                raise ValueError(f"future_inputs must have length H; got {U.shape[0]} vs {H}")
        else:
            U = [None] * H  # dummy list of Nones

        n = x.size
        p = self.v_mean.size
        x_future = np.empty((H, n), dtype=float)
        y_future = np.empty((H, p), dtype=float)

        for i in range(H):
            # propagate with nonlinear dynamics
            x = self.f_call(x, U[i]) + self.w_mean
            # output at the new state
            y = self.h(x) + self.v_mean

            x_future[i] = x
            y_future[i] = y

        return y_future, x_future
