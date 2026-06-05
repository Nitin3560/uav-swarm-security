from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import block_diag

try:
    from paper_sim.ids import AttackClass
except ModuleNotFoundError:  # pragma: no cover - supports direct local execution
    from ids import AttackClass


try:
    import osqp
    import scipy.sparse as sp

    _USE_OSQP = True
except ImportError:  # pragma: no cover - deterministic fallback below
    _USE_OSQP = False


class TrustMPC:
    """Finite-horizon, attack-conditioned reference MPC for the supervisory layer."""

    DELTA_MAX = 0.05
    ETA = 2.0
    KAPPA = 0.5
    HORIZON = 5
    NX = 6
    NU = 3
    Q_TRACK = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_INTERV = np.eye(3) * 0.1
    F_FORM = np.diag([6.0, 6.0, 1.0])
    # Fixed response smoothing used in the reported study.
    SMOOTH_ALPHA = 0.20

    def __init__(self, twin: Any, n_agents: int = 4, dt: float = 1.0 / 48.0):
        self.twin = twin
        self.n = int(n_agents)
        self.dt = float(dt)
        self.T = self.HORIZON
        self.nx = self.NX
        self.nu = self.NU
        self.P_inf = twin.P_inf

        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.A = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.B = np.block([[0.5 * self.dt**2 * eye3], [self.dt * eye3]])
        self.H = np.block([eye3, zero3])

        self.Phi, self.Gamma = self._build_prediction_matrices()
        q_blocks = [self.Q_TRACK] * (self.T - 1) + [self.Q_TRACK + self.P_inf]
        self.Q_bar = block_diag(*q_blocks)
        self.R_bar = block_diag(*([self.R_INTERV] * self.T))

        self.H_qp_single = self.Gamma.T @ self.Q_bar @ self.Gamma + self.R_bar
        self.P_qp_single = 2.0 * self.H_qp_single
        self.G_single = np.vstack([np.eye(self.T * self.nu), -np.eye(self.T * self.nu)])
        self.b_single = np.ones(2 * self.T * self.nu, dtype=float) * self.DELTA_MAX

        self._osqp_solvers: dict[int, Any] = {}
        if _USE_OSQP:
            p_mat = sp.csc_matrix(self.P_qp_single)
            a_mat = sp.csc_matrix(self.G_single)
            for i in range(self.n):
                solver = osqp.OSQP()
                solver.setup(
                    p_mat,
                    np.zeros(self.T * self.nu, dtype=float),
                    a_mat,
                    -self.b_single,
                    self.b_single,
                    warm_starting=True,
                    verbose=False,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    max_iter=1000,
                )
                self._osqp_solvers[i] = solver

        self._prev_delta = {i: np.zeros(3, dtype=float) for i in range(self.n)}
        self._prev_U = {i: np.zeros(self.T * self.nu, dtype=float) for i in range(self.n)}
        self._last_omega = np.ones(self.n, dtype=float) / max(self.n, 1)
        self._last_mode = "nominal"

    def _build_prediction_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        phi = np.zeros((self.T * self.nx, self.nx), dtype=float)
        gamma = np.zeros((self.T * self.nx, self.T * self.nu), dtype=float)
        a_power = np.eye(self.nx)
        for t_idx in range(self.T):
            a_power = self.A @ a_power
            row = slice(t_idx * self.nx, (t_idx + 1) * self.nx)
            phi[row, :] = a_power
            for s_idx in range(t_idx + 1):
                col = slice(s_idx * self.nu, (s_idx + 1) * self.nu)
                gamma[row, col] = np.linalg.matrix_power(self.A, t_idx - s_idx) @ self.B
        return phi, gamma

    def _trust_weights(self, ids_out: dict[str, Any]) -> np.ndarray:
        scores = np.array(
            [float(ids_out.get("spoof_agent_scores", {}).get(i, 0.0)) for i in range(self.n)],
            dtype=float,
        )
        if float(ids_out["lambdas"].get(AttackClass.H3_SPOOF, 0.0)) <= 0.0:
            scores[:] = 0.0
        weights = np.exp(-self.ETA * scores / (np.max(scores) + 1e-9))
        return weights / (weights.sum() + 1e-9)

    def _age_weights(self, neighbor_states: dict[int, dict[int, dict[str, Any] | None]], t: float) -> np.ndarray:
        weights = np.ones(self.n, dtype=float)
        for j in range(self.n):
            ages = []
            for neighbors in neighbor_states.values():
                state = neighbors.get(j)
                if state is not None:
                    ages.append(max(0.0, float(t) - float(state.get("timestamp", t))))
            if ages:
                weights[j] = float(np.exp(-self.KAPPA * np.mean(ages)))
        return weights / (weights.sum() + 1e-9)

    def _attack_targets(
        self,
        refs: np.ndarray,
        offsets: np.ndarray,
        ids_out: dict[str, Any],
        neighbor_states: dict[int, dict[int, dict[str, Any] | None]],
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        k_hat = ids_out["k_hat"]
        twin_pos = np.array([self.twin.get_state(i)[:3] for i in range(self.n)])
        target = refs.copy()
        trust = np.ones(self.n, dtype=float) / max(self.n, 1)
        mode = "nominal"

        if k_hat == AttackClass.H2_JAMMING:
            target = refs.copy()
            mode = "jamming_silent"
        elif k_hat == AttackClass.H3_SPOOF:
            trust = self._trust_weights(ids_out)
            center = np.sum(twin_pos * trust[:, None], axis=0)
            target = center[None, :] + offsets
            mode = "spoof_trust_consensus"
        elif k_hat == AttackClass.H4_REPLAY:
            trust = self._age_weights(neighbor_states, t)
            center = np.sum(twin_pos * trust[:, None], axis=0)
            target = center[None, :] + offsets
            mode = "replay_age_weighted"
        elif k_hat == AttackClass.H1_WIND:
            target = refs + 0.5 * (twin_pos - refs)
            mode = "wind_twin_guided"
        self._last_omega = trust.copy()
        self._last_mode = mode
        return target, trust, mode

    def _reference_horizon(self, target_pos: np.ndarray, x0: np.ndarray) -> np.ndarray:
        horizon = np.zeros(self.T * self.nx, dtype=float)
        ref_vel = np.asarray(x0[3:], dtype=float)
        for tau in range(self.T):
            row = tau * self.nx
            horizon[row : row + 3] = target_pos + ref_vel * tau * self.dt
            horizon[row + 3 : row + 6] = ref_vel
        return horizon

    def _formation_linear_term(self, i: int, refs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Gradient of the full-horizon formation cost with respect to U_i."""
        x0_i = self.twin.get_state(i)
        f_vec = np.zeros(self.T * self.nu, dtype=float)
        for j in range(self.n):
            if j == i:
                continue
            x0_j = self.twin.get_state(j)
            desired_rel = offsets[i] - offsets[j]
            for tau in range(self.T):
                row = slice(tau * self.nx, (tau + 1) * self.nx)
                gamma_tau = self.Gamma[row, :]
                x_diff_free = self.Phi[row, :] @ (x0_i - x0_j)
                e_tau = self.H @ x_diff_free - desired_rel
                f_vec += gamma_tau.T @ self.H.T @ self.F_FORM @ e_tau
        return (2.0 / max(self.n - 1, 1)) * f_vec

    def _project_horizon(self, u_vec: np.ndarray) -> np.ndarray:
        out = np.asarray(u_vec, dtype=float).copy()
        for tau in range(self.T):
            seg = slice(tau * self.nu, (tau + 1) * self.nu)
            norm = float(np.linalg.norm(out[seg]))
            if norm > self.DELTA_MAX:
                out[seg] *= self.DELTA_MAX / (norm + 1e-9)
        return np.clip(out, -self.DELTA_MAX, self.DELTA_MAX)

    def _solve_horizon_qp(self, refs: np.ndarray, target: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        delta_out = np.zeros((self.n, 3), dtype=float)
        for i in range(self.n):
            x0 = self.twin.get_state(i)
            x_ref_horizon = self._reference_horizon(target[i], x0)
            x_pred_free = self.Phi @ x0
            error = x_pred_free - x_ref_horizon
            linear = 2.0 * self.Gamma.T @ self.Q_bar @ error
            linear += self._formation_linear_term(i, refs, offsets)

            u_opt: np.ndarray
            if _USE_OSQP and i in self._osqp_solvers:
                solver = self._osqp_solvers[i]
                solver.update(q=linear, l=-self.b_single, u=self.b_single)
                result = solver.solve()
                if result.info.status_val in {1, 2} and result.x is not None and np.all(np.isfinite(result.x)):
                    u_opt = np.asarray(result.x, dtype=float)
                else:
                    u_opt = self._prev_U[i].copy()
            else:
                try:
                    u_opt = -0.5 * np.linalg.solve(self.H_qp_single, linear)
                except np.linalg.LinAlgError:
                    u_opt = np.zeros(self.T * self.nu, dtype=float)

            u_opt = self._project_horizon(u_opt)
            self._prev_U[i] = u_opt.copy()
            delta_out[i] = u_opt[: self.nu]
        return delta_out

    def step(
        self,
        ref_positions: np.ndarray,
        desired_offsets: np.ndarray,
        ids_out: dict[str, Any],
        neighbor_states: dict[int, dict[int, dict[str, Any] | None]],
        t: float,
    ) -> dict[int, np.ndarray]:
        refs = np.asarray(ref_positions, dtype=float)
        offsets = np.asarray(desired_offsets, dtype=float)
        if ids_out["k_hat"] == AttackClass.H0_NONE:
            delta = np.zeros((self.n, 3), dtype=float)
            self._last_omega = np.ones(self.n, dtype=float) / max(self.n, 1)
            self._last_mode = "nominal"
        else:
            target, _trust, _mode = self._attack_targets(refs, offsets, ids_out, neighbor_states, t)
            delta = self._solve_horizon_qp(refs, target, offsets)

        out: dict[int, np.ndarray] = {}
        for i in range(self.n):
            smoothed = self.SMOOTH_ALPHA * delta[i] + (1.0 - self.SMOOTH_ALPHA) * self._prev_delta[i]
            norm = float(np.linalg.norm(smoothed))
            if norm > self.DELTA_MAX:
                smoothed *= self.DELTA_MAX / (norm + 1e-9)
            self._prev_delta[i] = smoothed.copy()
            out[i] = smoothed
        return out
