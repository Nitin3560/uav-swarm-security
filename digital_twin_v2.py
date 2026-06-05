"""digital_twin_v2.py — Extended DigitalTwin with sensor integrity gate.

Extension 1: Pre-update innovation gate
----------------------------------------
Before accepting any sensor measurement z_i(t) into the Kalman update step,
we compute the pre-update residual and test it against a threshold derived from
the sensor noise covariance R and a configurable scale factor (sensor_gate_sigma).

If  ||z_i - H x̂_i(t|t-1)||² > sensor_gate_sigma² * chi2_threshold(df=3, alpha)
    → reject measurement, propagate on dynamics only (coast mode)
    → set sensor_gated[i] = True for that timestep

This closes the sensor-layer vulnerability identified in the original study:
sensor corruption could "poison the twin update" by pulling x̂ toward the
corrupted measurement, collapsing the innovation γ and making the IDS blind.

With the gate active, a sufficiently large sensor bias is caught before it enters
the update.  The twin coasts on the dynamics model — tracking degrades slightly
but does not catastrophically worsen.

Gate threshold derivation
--------------------------
Under H0 (no sensor attack), z_i - H x̂_prior ~ N(0, S_i) where
    S_i = H P_prior H^T + R
The Mahalanobis distance squared is chi-squared distributed with df = nz = 3.
We choose a false-rejection rate alpha_gate (default 0.001, i.e. 0.1%) giving
    tau_gate = chi2.ppf(1 - alpha_gate, df=nz)
A measurement is rejected iff its Mahalanobis distance exceeds tau_gate.

This is identical in form to the communication-layer NIS test in ids.py —
the same statistical principle applied symmetrically to the sensor input.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_discrete_are
    from scipy.stats import chi2 as _chi2_dist
except ImportError:  # pragma: no cover
    solve_discrete_are = None
    _chi2_dist = None


class DigitalTwin:
    """Per-agent constant-velocity Kalman filter with optional sensor integrity gate.

    Parameters
    ----------
    n_agents : int
    dt : float
        Control timestep (seconds).
    sigma_p, sigma_v : float
        Process noise standard deviations for position and velocity states.
    sigma_meas : float
        Measurement noise standard deviation (isotropic, 3-axis).
    sensor_gate : bool
        Enable the pre-update sensor integrity gate (Extension 1).
        Default True.
    sensor_gate_alpha : float
        False-rejection rate for the gate chi-squared test.
        Default 0.01 (1% false rejection under H0), aligned with the IDS NIS
        threshold so the twin and IDS use the same nominal innovation boundary.
    """

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 1.0 / 48.0,
        sigma_p: float = 0.02,
        sigma_v: float = 0.05,
        sigma_meas: float = 0.02,
        sensor_gate: bool = True,
        sensor_gate_alpha: float = 0.01,
    ):
        self.n = int(n_agents)
        self.dt = float(dt)
        self.nz = 3
        self.nx = 6
        self.sensor_gate_enabled = bool(sensor_gate)

        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.A = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.B = np.block([[0.5 * self.dt**2 * eye3], [self.dt * eye3]])
        self.H = np.block([eye3, zero3])
        self.Q = np.diag([sigma_p**2] * 3 + [sigma_v**2] * 3)
        self.R = np.eye(3) * sigma_meas**2

        # Gate threshold: chi-squared with df=nz at significance level alpha
        if _chi2_dist is not None:
            self.tau_gate = float(_chi2_dist.ppf(1.0 - sensor_gate_alpha, df=self.nz))
        else:
            # Fallback: chi2(df=3, alpha=0.001) ≈ 16.27
            self.tau_gate = 16.27

        self.x_hat = {i: np.zeros(6, dtype=float) for i in range(self.n)}
        self.P = {i: np.eye(6, dtype=float) * 0.1 for i in range(self.n)}
        self.x_prior = {i: np.zeros(6, dtype=float) for i in range(self.n)}
        self.P_prior = {i: np.eye(6, dtype=float) * 0.1 for i in range(self.n)}
        self.gamma = {i: np.zeros(3, dtype=float) for i in range(self.n)}
        self.S = {i: np.eye(3, dtype=float) * sigma_meas**2 for i in range(self.n)}

        # Sensor gate state per agent
        self.sensor_gated: dict[int, bool] = {i: False for i in range(self.n)}
        # Running coast counter per agent (consecutive steps on dynamics only)
        self.coast_steps: dict[int, int] = {i: 0 for i in range(self.n)}
        # Pre-gate Mahalanobis distance squared (for logging / analysis)
        self.sensor_nis: dict[int, float] = {i: 0.0 for i in range(self.n)}

        if solve_discrete_are is not None:
            q_track = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
            r_ctrl = np.eye(3) * 0.1
            self.P_inf = solve_discrete_are(self.A, self.B, q_track, r_ctrl)
        else:
            self.P_inf = np.eye(6, dtype=float)

    # ------------------------------------------------------------------
    # Public interface (backward-compatible with original DigitalTwin)
    # ------------------------------------------------------------------

    def initialize(self, initial_positions: dict[int, np.ndarray]) -> None:
        for i, pos in initial_positions.items():
            idx = int(i)
            self.x_hat[idx] = np.zeros(6, dtype=float)
            self.x_hat[idx][:3] = np.asarray(pos, dtype=float)
            self.P[idx] = np.eye(6, dtype=float) * 0.1
            self.x_prior[idx] = self.x_hat[idx].copy()
            self.P_prior[idx] = self.P[idx].copy()
            self.sensor_gated[idx] = False
            self.coast_steps[idx] = 0

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
        """Update each agent's state estimate.

        With sensor_gate=True (default), each measurement is tested against
        the pre-update innovation Mahalanobis distance before entering the
        Kalman update.  Rejected measurements cause the agent to coast on its
        dynamics prediction.
        """
        for i in range(self.n):
            z = np.asarray(sensor_positions[i], dtype=float)

            # --- Hard NaN/Inf check (original behaviour) ---
            if np.any(~np.isfinite(z)):
                self._coast(i)
                continue

            # --- Pre-update innovation ---
            gamma_pre = z - self.H @ self.x_prior[i]
            S_i = self.H @ self.P_prior[i] @ self.H.T + self.R
            try:
                S_inv = np.linalg.inv(S_i)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S_i)

            # Mahalanobis distance squared (NIS for sensor)
            nis_sensor = float(gamma_pre @ S_inv @ gamma_pre)
            self.sensor_nis[i] = nis_sensor

            # --- Sensor integrity gate ---
            if self.sensor_gate_enabled and nis_sensor > self.tau_gate:
                # Measurement inconsistent with dynamics prediction →
                # reject and coast; flag for diagnostics
                self.sensor_gated[i] = True
                self.coast_steps[i] += 1
                self._coast(i)
                # Still expose the pre-gate innovation so the IDS
                # can observe the anomaly if it wants to
                self.gamma[i] = gamma_pre
                self.S[i] = S_i
                continue

            # --- Normal Kalman update ---
            self.sensor_gated[i] = False
            self.coast_steps[i] = 0
            self.gamma[i] = gamma_pre
            self.S[i] = S_i
            try:
                gain = self.P_prior[i] @ self.H.T @ S_inv
            except Exception:
                gain = self.P_prior[i] @ self.H.T @ np.linalg.pinv(S_i)
            self.x_hat[i] = self.x_prior[i] + gain @ self.gamma[i]
            ident = np.eye(self.nx)
            ikh = ident - gain @ self.H
            self.P[i] = ikh @ self.P_prior[i] @ ikh.T + gain @ self.R @ gain.T

    # ------------------------------------------------------------------
    # Getters (identical interface to original)
    # ------------------------------------------------------------------

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

    # Extension 1 — new getters for sensor gate diagnostics
    def get_sensor_gated(self, i: int) -> bool:
        """True if agent i's measurement was rejected by the sensor gate this step."""
        return self.sensor_gated[int(i)]

    def get_sensor_nis(self, i: int) -> float:
        """Pre-gate Mahalanobis distance squared for agent i's last measurement."""
        return self.sensor_nis[int(i)]

    def get_coast_steps(self, i: int) -> int:
        """Consecutive steps agent i has been coasting on dynamics only."""
        return self.coast_steps[int(i)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coast(self, i: int) -> None:
        """Propagate on dynamics only — do not update from measurement."""
        self.x_hat[i] = self.x_prior[i].copy()
        self.P[i] = self.P_prior[i].copy()
        self.gamma[i] = np.zeros(3, dtype=float)
        self.S[i] = self.H @ self.P_prior[i] @ self.H.T + self.R
