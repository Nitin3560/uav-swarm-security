from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

try:
    from scipy.stats import chi2
except ImportError:  # pragma: no cover - 99th percentile for df=3
    chi2 = None


class AttackClass(IntEnum):
    H0_NONE = 0
    H1_WIND = 1
    H2_JAMMING = 2
    H3_SPOOF = 3
    H4_REPLAY = 4


class IDS:
    """Chi-squared anomaly detector, GLRT-style attribution, and CUSUM confirmation."""

    ALPHA = 0.01
    DF = 3
    B_CUSUM = {
        AttackClass.H1_WIND: 0.8,
        AttackClass.H2_JAMMING: 1.5,
        AttackClass.H3_SPOOF: 1.2,
        AttackClass.H4_REPLAY: 0.6,
    }
    RHO = 1e-3
    TAU_STALE = 3.0

    def __init__(self, n_agents: int = 4, dt: float = 1.0 / 48.0):
        self.n = int(n_agents)
        self.dt = float(dt)
        self.tau = float(chi2.ppf(1 - self.ALPHA, df=self.DF)) if chi2 is not None else 11.34486673
        self._h = {k: -np.log(self.RHO) / v for k, v in self.B_CUSUM.items()}
        self.reset()

    def reset(self) -> None:
        self._g = {k: 0.0 for k in AttackClass if k != AttackClass.H0_NONE}
        self._ttd_started = False
        self._ttd_start_t: float | None = None
        self._ttd_recorded: float | None = None
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _quad(g: np.ndarray, s: np.ndarray) -> float:
        try:
            return float(g @ np.linalg.inv(s) @ g)
        except np.linalg.LinAlgError:
            return float(g @ np.linalg.pinv(s) @ g)

    def _chi2_test(self, twin: Any) -> tuple[dict[int, float], float, dict[int, bool]]:
        chi2_agents: dict[int, float] = {}
        flags: dict[int, bool] = {}
        for i in range(self.n):
            val = self._quad(twin.get_innovation(i), twin.get_innov_cov(i))
            chi2_agents[i] = val
            flags[i] = val > self.tau
        chi2_swarm = float(np.mean(list(chi2_agents.values()))) if chi2_agents else 0.0
        return chi2_agents, chi2_swarm, flags

    def _glrt(
        self,
        twin: Any,
        neighbor_states: dict[int, dict[int, dict[str, Any] | None]],
        qcomm: dict[int, float],
        t: float,
    ) -> tuple[dict[AttackClass, float], np.ndarray]:
        chi2_vals = np.array([self._quad(twin.get_innovation(i), twin.get_innov_cov(i)) for i in range(self.n)])
        innov_norms = np.array([np.linalg.norm(twin.get_innovation(i)) for i in range(self.n)])

        mean_drift = float(np.mean(innov_norms))
        std_drift = float(np.std(innov_norms))
        wind_score = mean_drift / (std_drift + 1e-6) if mean_drift > 0.03 else 0.0

        q_values = np.array([float(qcomm.get(i, 1.0)) for i in range(self.n)])
        jam_scores = -np.log(np.clip(q_values, 1e-9, 1.0))
        mean_jam = float(np.mean(jam_scores))
        std_jam = float(np.std(jam_scores))

        sender_errors = np.zeros(self.n, dtype=float)
        sender_counts = np.zeros(self.n, dtype=float)
        for neighbors in neighbor_states.values():
            for sender_id, state in neighbors.items():
                if state is None:
                    continue
                sender = int(sender_id)
                msg_pos = np.asarray(state["pos"], dtype=float)
                sender_errors[sender] += float(np.linalg.norm((msg_pos - twin.get_state(sender)[:3])[:2]))
                sender_counts[sender] += 1.0
        sender_errors = np.divide(sender_errors, np.maximum(sender_counts, 1.0))
        max_idx = int(np.argmax(sender_errors)) if sender_errors.size else 0
        max_spoof_err = float(sender_errors[max_idx]) if sender_errors.size else 0.0
        rest = np.delete(sender_errors, max_idx) if sender_errors.size > 1 else np.array([0.0])
        rest_spoof_err = float(np.mean(rest))
        spoof_isolation = max_spoof_err / (rest_spoof_err + 1e-6)
        comm_healthy = float(np.mean(q_values) > 0.8)
        spoof_score = max(0.0, max_spoof_err - 0.5) * 4.0 + max(0.0, np.log(spoof_isolation + 1e-6) - 2.0)

        stale_scores: list[float] = []
        for neighbors in neighbor_states.values():
            for state in neighbors.values():
                if state is None:
                    continue
                msg_t = float(state.get("timestamp", t))
                stale_scores.append(max(0.0, float(t) - msg_t - self.TAU_STALE))
        mean_stale = float(np.mean(stale_scores)) if stale_scores else 0.0
        if mean_stale > 0.5:
            spoof_score = 0.0

        return {
            AttackClass.H1_WIND: float(wind_score - 2.0),
            AttackClass.H2_JAMMING: float(3.0 * mean_jam - 0.5 * std_jam),
            AttackClass.H3_SPOOF: float(spoof_score * comm_healthy),
            AttackClass.H4_REPLAY: mean_stale,
        }, sender_errors

    def _cusum_update(self, lambdas: dict[AttackClass, float], anomaly_flag_swarm: bool) -> AttackClass:
        for k, lam in lambdas.items():
            if anomaly_flag_swarm or k in {AttackClass.H2_JAMMING, AttackClass.H3_SPOOF, AttackClass.H4_REPLAY}:
                self._g[k] = max(0.0, self._g[k] + float(lam) - self.B_CUSUM.get(k, 1.0))
            else:
                self._g[k] = max(0.0, self._g[k] - 0.1)
        confirmed = [k for k in lambdas if self._g[k] > self._h.get(k, 1e9)]
        if not confirmed:
            return AttackClass.H0_NONE
        return max(confirmed, key=lambda k: self._g[k])

    def step(
        self,
        twin: Any,
        neighbor_states: dict[int, dict[int, dict[str, Any] | None]],
        qcomm: dict[int, float],
        t: float,
        attack_start_t: float = 20.0,
    ) -> dict[str, Any]:
        chi2_agents, chi2_swarm, anomaly_flags = self._chi2_test(twin)
        lambdas, spoof_agent_scores = self._glrt(twin, neighbor_states, qcomm, t)
        k_hat_raw = max(lambdas, key=lambdas.get)
        attack_evidence = (
            lambdas.get(AttackClass.H2_JAMMING, 0.0) > 1.0
            or lambdas.get(AttackClass.H3_SPOOF, 0.0) > 1.0
            or lambdas.get(AttackClass.H4_REPLAY, 0.0) > 0.5
        )
        anomaly_flag_swarm = any(anomaly_flags.values()) or attack_evidence
        k_hat = self._cusum_update(lambdas, anomaly_flag_swarm)

        target_k3 = int(np.argmax(spoof_agent_scores)) if k_hat == AttackClass.H3_SPOOF else None
        if t >= attack_start_t and not self._ttd_started:
            self._ttd_started = True
            self._ttd_start_t = float(t)
        if self._ttd_started and self._ttd_recorded is None and k_hat != AttackClass.H0_NONE:
            self._ttd_recorded = float(t) - float(self._ttd_start_t if self._ttd_start_t is not None else t)

        out = {
            "chi2_agents": chi2_agents,
            "chi2_swarm": chi2_swarm,
            "anomaly_flags": anomaly_flags,
            "anomaly_flag_swarm": anomaly_flag_swarm,
            "lambdas": lambdas,
            "cusum_g": dict(self._g),
            "k_hat_raw": k_hat_raw,
            "k_hat": k_hat,
            "target_k3": target_k3,
            "spoof_agent_scores": {i: float(spoof_agent_scores[i]) for i in range(self.n)},
            "ttd": self._ttd_recorded,
        }
        self.history.append(out)
        return out
