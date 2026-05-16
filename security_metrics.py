from __future__ import annotations

from typing import Any

import numpy as np

from paper_sim.ids import AttackClass


class SecurityMetrics:
    """Accumulate detection, attribution, and formation metrics for one run."""

    def __init__(
        self,
        attack_start: float = 20.0,
        attack_end: float = 30.0,
        nominal_start: float = 10.0,
        nominal_end: float = 20.0,
        true_attack_class: AttackClass = AttackClass.H0_NONE,
    ):
        self.atk_start = float(attack_start)
        self.atk_end = float(attack_end)
        self.nom_start = float(nominal_start)
        self.nom_end = float(nominal_end)
        self.true_k = true_attack_class
        self._steps: list[dict[str, Any]] = []

    def log(
        self,
        t: float,
        ids_out: dict[str, Any],
        positions: np.ndarray,
        ref_positions: np.ndarray,
        desired_offsets: np.ndarray,
    ) -> None:
        positions = np.asarray(positions, dtype=float)
        offsets = np.asarray(desired_offsets, dtype=float)
        n_agents = positions.shape[0]
        form_err = 0.0
        count = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                desired_delta = offsets[i] - offsets[j]
                actual_delta = positions[i] - positions[j]
                form_err += float(np.linalg.norm((actual_delta - desired_delta)[:2]))
                count += 1
        form_err /= max(count, 1)
        self._steps.append(
            {
                "t": float(t),
                "in_attack": self.atk_start <= t < self.atk_end,
                "in_nominal": self.nom_start <= t < self.nom_end,
                "flag": bool(ids_out["anomaly_flag_swarm"]),
                "k_hat": ids_out["k_hat"],
                "k_hat_raw": ids_out["k_hat_raw"],
                "chi2_swarm": float(ids_out["chi2_swarm"]),
                "ttd": ids_out.get("ttd"),
                "form_err": form_err,
            }
        )

    def summary(self) -> dict[str, float]:
        atk_steps = [s for s in self._steps if s["in_attack"]]
        nom_steps = [s for s in self._steps if s["in_nominal"]]
        dr = np.mean([s["flag"] for s in atk_steps]) if atk_steps else float("nan")
        far = np.mean([s["flag"] for s in nom_steps]) if nom_steps else float("nan")

        if atk_steps and self.true_k != AttackClass.H0_NONE:
            confirmed = [s for s in atk_steps if s["k_hat"] != AttackClass.H0_NONE]
            aa = np.mean([s["k_hat"] == self.true_k for s in confirmed]) if confirmed else 0.0
            raw_aa = np.mean([s["k_hat_raw"] == self.true_k for s in atk_steps])
        else:
            aa = float("nan")
            raw_aa = float("nan")

        ttd_vals = [s["ttd"] for s in self._steps if s["ttd"] is not None]
        ttd = float(ttd_vals[0]) if ttd_vals else float("nan")
        if self.true_k == AttackClass.H1_WIND and atk_steps:
            disr = np.mean([s["k_hat"] in (AttackClass.H0_NONE, AttackClass.H1_WIND) for s in atk_steps])
        else:
            disr = float("nan")
        fca = np.mean([s["form_err"] for s in atk_steps]) if atk_steps else float("nan")
        return {
            "security_DR": float(dr),
            "security_FAR": float(far),
            "security_AA": float(aa),
            "security_raw_AA": float(raw_aa),
            "security_TTD_s": float(ttd),
            "security_DisR": float(disr),
            "security_FCA_m": float(fca),
        }
