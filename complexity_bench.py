"""complexity_bench.py — Per-stage timing instrumentation (Extension 3).

This module wraps each pipeline stage (Kalman predict/update, GLRT,
CUSUM, MPC solve) with a lightweight timer and accumulates statistics
across the simulation run.

Usage
-----
In run_study_v2.py, instantiate a PipelineTimer at the start of the run
and call its context-manager wrappers around each stage.  At the end of
the run, call timer.summary() to get a per-stage timing DataFrame and
timer.complexity_table(n_agents) to get the theoretical complexity table.

Example
-------
    timer = PipelineTimer(ctrl_hz=48.0)

    with timer.time("kalman_predict"):
        twin.predict(accel_cmd)

    with timer.time("kalman_update"):
        twin.update(measured_pos)

    with timer.time("ids_step"):
        ids_out = ids_mod.step(...)

    with timer.time("mpc_step"):
        trust_delta = trust_mpc.step(...)

    df_timing = timer.summary()
    df_complexity = timer.complexity_table(n_agents)
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd


@dataclass
class StageStats:
    name: str
    times_us: list[float] = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self.times_us.append(elapsed_s * 1e6)

    def mean_us(self) -> float:
        return float(np.mean(self.times_us)) if self.times_us else 0.0

    def std_us(self) -> float:
        return float(np.std(self.times_us)) if self.times_us else 0.0

    def max_us(self) -> float:
        return float(np.max(self.times_us)) if self.times_us else 0.0

    def p99_us(self) -> float:
        return float(np.percentile(self.times_us, 99)) if self.times_us else 0.0

    def budget_fraction(self, ctrl_hz: float) -> float:
        """Fraction of the control period budget consumed."""
        period_us = 1e6 / ctrl_hz
        return self.mean_us() / period_us if period_us > 0 else 0.0


class PipelineTimer:
    """Lightweight per-stage timing accumulator.

    Parameters
    ----------
    ctrl_hz : float
        Control loop frequency (Hz).  Used to compute budget fractions.
    """

    STAGES = [
        "kalman_predict",
        "kalman_update",
        "ids_step",
        "mpc_step",
        "total_ctrl_step",
    ]

    def __init__(self, ctrl_hz: float = 48.0):
        self.ctrl_hz = float(ctrl_hz)
        self._stats: dict[str, StageStats] = defaultdict(lambda: StageStats(""))
        for s in self.STAGES:
            self._stats[s] = StageStats(name=s)

    @contextmanager
    def time(self, stage: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            if stage not in self._stats:
                self._stats[stage] = StageStats(name=stage)
            self._stats[stage].record(elapsed)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with per-stage timing statistics."""
        rows = []
        period_us = 1e6 / self.ctrl_hz
        for stage, stats in self._stats.items():
            if not stats.times_us:
                continue
            rows.append({
                "stage": stage,
                "mean_us": round(stats.mean_us(), 2),
                "std_us": round(stats.std_us(), 2),
                "p99_us": round(stats.p99_us(), 2),
                "max_us": round(stats.max_us(), 2),
                "budget_pct": round(100.0 * stats.budget_fraction(self.ctrl_hz), 2),
                "n_samples": len(stats.times_us),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def complexity_table(n_agents: int, nx: int = 6, nz: int = 3,
                         n_hyp: int = 4, horizon: int = 5, nu: int = 3) -> pd.DataFrame:
        """Return a theoretical complexity table for the given swarm size.

        Parameters
        ----------
        n_agents : int   Number of UAVs.
        nx : int         State dimension per agent (default 6: pos+vel).
        nz : int         Measurement dimension per agent (default 3: pos).
        n_hyp : int      Number of GLRT hypotheses (default 4: H1-H4).
        horizon : int    MPC prediction horizon (default 5).
        nu : int         Control input dimension per agent (default 3).
        """
        # Kalman predict: A @ x (nx×nx mat-vec) + P update (nx×nx mat-mat) per agent
        kalman_predict_flops = n_agents * (nx * nx + 2 * nx * nx * nx)

        # Kalman update: S = H P H^T + R (nz×nz), K = P H^T S^{-1} (nx×nz),
        #                x_hat update (nx), P update (nx×nx) per agent
        kalman_update_flops = n_agents * (
            nz * nx * nz          # H P H^T
            + nz ** 3             # S^{-1}
            + nx * nz * nz        # P H^T S^{-1}
            + nx * nz             # K γ
            + nx * nx             # (I - KH) P
        )

        # GLRT: chi2 per agent (nz^2 per agent) + hypothesis scoring (n_hyp passes)
        glrt_flops = n_agents * nz ** 2 + n_hyp * n_agents

        # CUSUM: n_hyp scalar additions
        cusum_flops = n_hyp * 3  # add, compare, clip

        # MPC: QP cost O(n_agents * (T*nu)^2) — dominant term is Hessian build
        #      Using OSQP warm-start, each solve is roughly O(T^2 * nu^2) per agent
        mpc_flops = n_agents * (horizon * nu) ** 2

        rows = [
            {
                "Stage": "Kalman Predict",
                "Complexity": f"O(N · nx³)",
                "N=4": _fmt(kalman_predict_flops),
                "N=8": _fmt(kalman_predict_flops * 2),
                "N=16": _fmt(kalman_predict_flops * 4),
                "Notes": "Scales linearly in N; nx=6 is small",
            },
            {
                "Stage": "Kalman Update",
                "Complexity": f"O(N · (nx² + nz³))",
                "N=4": _fmt(kalman_update_flops),
                "N=8": _fmt(kalman_update_flops * 2),
                "N=16": _fmt(kalman_update_flops * 4),
                "Notes": "nz=3 inversion is O(27) — negligible",
            },
            {
                "Stage": "Sensor Gate",
                "Complexity": "O(N · nz²)",
                "N=4": _fmt(n_agents * nz ** 2),
                "N=8": _fmt(8 * nz ** 2),
                "N=16": _fmt(16 * nz ** 2),
                "Notes": "Extension 1 addition — trivial cost",
            },
            {
                "Stage": "GLRT Attribution",
                "Complexity": "O(K · N)",
                "N=4": _fmt(glrt_flops),
                "N=8": _fmt(n_hyp * 8 + 8 * nz ** 2),
                "N=16": _fmt(n_hyp * 16 + 16 * nz ** 2),
                "Notes": "K=4 hypotheses; chi2 per agent",
            },
            {
                "Stage": "CUSUM Update",
                "Complexity": "O(K)",
                "N=4": _fmt(cusum_flops),
                "N=8": _fmt(cusum_flops),
                "N=16": _fmt(cusum_flops),
                "Notes": "Independent of N — K scalar ops",
            },
            {
                "Stage": "Trust MPC",
                "Complexity": "O(N · (T·nu)²)",
                "N=4": _fmt(mpc_flops),
                "N=8": _fmt(8 * (horizon * nu) ** 2),
                "N=16": _fmt(16 * (horizon * nu) ** 2),
                "Notes": "Bottleneck at large N; OSQP warm-start helps",
            },
            {
                "Stage": "TOTAL",
                "Complexity": "O(N · (nx³ + T²·nu²))",
                "N=4": _fmt(kalman_predict_flops + kalman_update_flops + glrt_flops + mpc_flops),
                "N=8": _fmt(2 * (kalman_predict_flops + kalman_update_flops + glrt_flops + mpc_flops)),
                "N=16": _fmt(4 * (kalman_predict_flops + kalman_update_flops + glrt_flops + mpc_flops)),
                "Notes": "MPC dominates at N≥8",
            },
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def hardware_feasibility_table(timing_df: pd.DataFrame, ctrl_hz: float = 48.0) -> pd.DataFrame:
        """Given measured timing, estimate feasibility on target hardware classes."""
        period_ms = 1000.0 / ctrl_hz
        # Hardware slowdown factors relative to a modern workstation
        hardware = {
            "Workstation (baseline)": 1.0,
            "Jetson Orin NX": 3.0,
            "Jetson Nano": 8.0,
            "Raspberry Pi 4": 15.0,
            "Raspberry Pi Zero 2W": 60.0,
        }
        if timing_df.empty or "stage" not in timing_df.columns:
            return pd.DataFrame()

        total_row = timing_df[timing_df["stage"] == "total_ctrl_step"]
        if total_row.empty:
            mean_total_us = float(timing_df["mean_us"].sum())
        else:
            mean_total_us = float(total_row["mean_us"].iloc[0])

        rows = []
        for hw, factor in hardware.items():
            est_us = mean_total_us * factor
            est_ms = est_us / 1000.0
            feasible = est_ms < period_ms * 0.8  # 80% budget headroom
            rows.append({
                "Hardware": hw,
                "Est. cycle time (ms)": round(est_ms, 3),
                "Budget (ms)": round(period_ms, 2),
                "Budget used (%)": round(100.0 * est_ms / period_ms, 1),
                "Real-time feasible": "✓" if feasible else "✗",
            })
        return pd.DataFrame(rows)


def _fmt(flops: float) -> str:
    """Format a FLOP count as a short human-readable string."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GF"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MF"
    if flops >= 1e3:
        return f"{flops / 1e3:.1f} KF"
    return f"{flops:.0f} F"
