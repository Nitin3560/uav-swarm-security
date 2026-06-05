from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_discrete_are
except ImportError:  # pragma: no cover - fallback for minimal environments
    solve_discrete_are = None


class DigitalTwin:
    """Per-agent constant-velocity Kalman filter used as a trusted estimator."""

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 1.0 / 48.0,
        sigma_p: float = 0.02,
        sigma_v: float = 0.05,
        sigma_meas: float = 0.02,
    ):
        self.n = int(n_agents)
        self.dt = float(dt)
        self.nz = 3
        self.nx = 6

        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.A = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.B = np.block([[0.5 * self.dt**2 * eye3], [self.dt * eye3]])
        self.H = np.block([eye3, zero3])
        self.Q = np.diag([sigma_p**2] * 3 + [sigma_v**2] * 3)
        self.R = np.eye(3) * sigma_meas**2

        self.x_hat = {i: np.zeros(6, dtype=float) for i in range(self.n)}
        self.P = {i: np.eye(6, dtype=float) * 0.1 for i in range(self.n)}
        self.x_prior = {i: np.zeros(6, dtype=float) for i in range(self.n)}
        self.P_prior = {i: np.eye(6, dtype=float) * 0.1 for i in range(self.n)}
        self.gamma = {i: np.zeros(3, dtype=float) for i in range(self.n)}
        self.S = {i: np.eye(3, dtype=float) * sigma_meas**2 for i in range(self.n)}

        if solve_discrete_are is not None:
            q_track = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
            r_ctrl = np.eye(3) * 0.1
            self.P_inf = solve_discrete_are(self.A, self.B, q_track, r_ctrl)
        else:
            self.P_inf = np.eye(6, dtype=float)

    def initialize(self, initial_positions: dict[int, np.ndarray]) -> None:
        for i, pos in initial_positions.items():
            idx = int(i)
            self.x_hat[idx] = np.zeros(6, dtype=float)
            self.x_hat[idx][:3] = np.asarray(pos, dtype=float)
            self.P[idx] = np.eye(6, dtype=float) * 0.1
            self.x_prior[idx] = self.x_hat[idx].copy()
            self.P_prior[idx] = self.P[idx].copy()

    def predict(self, control_inputs: dict[int, np.ndarray] | np.ndarray | None = None) -> None:
        for i in range(self.n):
            if control_inputs is None:
                u = np.zeros(3, dtype=float)
            elif isinstance(control_inputs, np.ndarray):
                u = np.asarray(control_inputs[i], dtype=float)
            else:
                u = np.asarray(control_inputs.get(i, np.zeros(3, dtype=float)), dtype=float)
            self.x_prior[i] = self.A @ self.x_hat[i] + self.B @ u
            self.P_prior[i] = self.A @ self.P[i] @ self.A.T + self.Q

    def update(self, sensor_positions: dict[int, np.ndarray] | np.ndarray) -> None:
        for i in range(self.n):
            z = np.asarray(sensor_positions[i], dtype=float)
            if np.any(~np.isfinite(z)):
                self.x_hat[i] = self.x_prior[i].copy()
                self.P[i] = self.P_prior[i].copy()
                self.gamma[i] = np.zeros(3, dtype=float)
                self.S[i] = self.H @ self.P_prior[i] @ self.H.T + self.R
                continue
            self.gamma[i] = z - self.H @ self.x_prior[i]
            self.S[i] = self.H @ self.P_prior[i] @ self.H.T + self.R
            try:
                gain = self.P_prior[i] @ self.H.T @ np.linalg.inv(self.S[i])
            except np.linalg.LinAlgError:
                gain = self.P_prior[i] @ self.H.T @ np.linalg.pinv(self.S[i])
            self.x_hat[i] = self.x_prior[i] + gain @ self.gamma[i]
            ident = np.eye(self.nx)
            ikh = ident - gain @ self.H
            self.P[i] = ikh @ self.P_prior[i] @ ikh.T + gain @ self.R @ gain.T

    def get_state(self, i: int) -> np.ndarray:
        return self.x_hat[int(i)].copy()

    def get_cov(self, i: int) -> np.ndarray:
        return self.P[int(i)].copy()

    def get_innovation(self, i: int) -> np.ndarray:
        return self.gamma[int(i)].copy()

    def get_innov_cov(self, i: int) -> np.ndarray:
        return self.S[int(i)].copy()

    def get_prior_state(self, i: int) -> np.ndarray:
        return self.x_prior[int(i)].copy()

    def get_prior_cov(self, i: int) -> np.ndarray:
        return self.P_prior[int(i)].copy()
