"""Prototype: ISAC sensing-quality-aware sensor error minimization.

Compares:
  1. Fixed-R Kalman filter (baseline)
  2. Sensing-quality adaptive-R Kalman filter
  3. Adaptive-R + NIS soft gate + correntropy reweighting (proposed)

FIXES vs original
-----------------
BUG-1  accepted flag was set False for soft reweights AND hard rejects with no
       distinction. Now FilterOutput carries separate soft_reweighted and
       hard_rejected booleans so metrics are not conflated.
BUG-2  NIS threshold nis_alpha=0.001 gives tau≈16.3, which is very permissive
       (only 0.1 % of good measurements are flagged). Default changed to
       nis_alpha=0.05 (tau≈7.8), the standard RAIM / integrity-monitoring value.
       Original value is kept as a named constant for easy comparison.
BUG-3  correntropy_scale=6.0 was arbitrary and undocumented. It is now set as
       a fraction of the NIS threshold (default 0.5*tau) so the soft-gate
       roll-off is principled relative to the rejection boundary.
BUG-4  NEES (Normalized Estimation Error Squared) was never computed. It is now
       computed per step and included in the row data and summary statistics.
       NEES is the mandatory state-space consistency check for IEEE papers.
BUG-5  Sensing-quality fade was an abrupt step (q multiplied at hard time
       boundaries). Replaced with a smooth sigmoid ramp over 0.5 s so the
       transition is physically realistic (matches Rician/Rayleigh channel
       behaviour).
BUG-6  No downstream control-decision cost was computed, leaving the ISAC claim
       unsupported. A discrete LQR controller is now added. At each step the
       controller computes the action from the estimate and compares it with
       the action that would have been selected from the true state. The
       accumulated cost is the estimation-induced LQG surrogate:
       e_x^T Q_lqg e_x + e_u^T R_lqg e_u, where e_u = u_hat - u_true.
       This makes the metric sensitive to measurement integrity without
       drowning it in absolute trajectory/control effort.
BUG-7  Digital-twin (DT) state error was simulated only in the dashboard widget,
       not in the code. Added: DT state is a delayed copy of the filter estimate
       (one-step lag), and DT error is norm(x_true - x_dt).
MINOR  Trailing comma inside float() in the outlier/nominal RMSE lines is
       harmless in CPython but confusing; removed.
MINOR  Sign convention in improvement printout was inverted in the test helper;
       fixed in the summary print.
NOTE   The sensing quality signal q_k is still synthetic (not derived from an
       explicit CRB/FIM formula). For the camera-ready paper, q_k should be
       q_k = SINR_k * G_r(theta_k) / SINR_ref so that R_k = R0/q_k is the
       true CRB-derived measurement covariance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
    from scipy.linalg import solve_discrete_are
except Exception:
    chi2 = None
    solve_discrete_are = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NIS_ALPHA_STANDARD = 0.05    # standard RAIM / integrity-monitoring level
NIS_ALPHA_LOOSE    = 0.001   # original value — kept for reference


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FilterOutput:
    state: np.ndarray
    covariance: np.ndarray
    nis: float
    nees: float                  # FIX BUG-4: was missing
    pos_nees: float              # 3D position-only consistency check
    soft_reweighted: bool        # FIX BUG-1: was conflated with hard_rejected
    hard_rejected: bool          # FIX BUG-1: new separate flag
    measurement_weight: float
    quality: float


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
class ISACQualityKalmanFilter:
    """Constant-velocity EKF with optional ISAC quality adaptation and NIS gate.

    Parameters
    ----------
    mode : str
        'fixed_kf'        — fixed R = R0 (baseline)
        'adaptive_r'      — R_k = R0 / q_k  (quality-adaptive, no gate)
        'adaptive_r_gate' — R_k = R0 / q_k  + NIS soft gate + correntropy
    nis_alpha : float
        False-alarm rate for the NIS chi-square gate.  Default 0.05 (standard).
    correntropy_scale : float | None
        Bandwidth of the NIS-space soft-gate kernel.  When None it is set to
        0.5 * tau automatically (principled relative to the rejection boundary).
    hard_reject_alpha : float
        Tail probability for a true hard reject. Unlike the soft gate, this is
        a severe-integrity threshold, not a tuning floor on the correntropy
        weight. Default 1e-6 means hard rejection happens only for innovations
        that are statistically implausible under the nominal measurement model.
    """

    def __init__(
        self,
        dt: float,
        base_sigma_meas: float,
        sigma_accel: float,
        mode: str,
        nis_alpha: float = NIS_ALPHA_STANDARD,
        min_quality: float = 0.04,
        correntropy_scale: float | None = None,
        gate_warmup_steps: int = 10,
        hard_reject_alpha: float = 1e-6,
    ):
        self.dt   = float(dt)
        self.mode = mode
        self.base_sigma_meas = float(base_sigma_meas)
        self.min_quality     = float(min_quality)

        eye3  = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.F = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.H = np.block([eye3, zero3])

        # Continuous white-noise-acceleration process noise (exact discretisation)
        q_pos   = 0.25 * self.dt**4 * sigma_accel**2
        q_cross = 0.50 * self.dt**3 * sigma_accel**2
        q_vel   =        self.dt**2 * sigma_accel**2
        self.Q  = np.block([
            [q_pos * eye3,   q_cross * eye3],
            [q_cross * eye3, q_vel * eye3  ],
        ])

        self.R0 = np.eye(3) * self.base_sigma_meas**2
        self.x  = np.zeros(6)
        self.P  = np.eye(6) * 1.0
        self.initialized = False
        self.step_count = 0
        self.gate_warmup_steps = int(gate_warmup_steps)

        # NIS gate threshold — chi2 inverse CDF at (1 - alpha), df = 3
        self.tau = float(chi2.ppf(1.0 - nis_alpha, df=3)) if chi2 is not None else 7.815
        self.hard_tau = (
            float(chi2.ppf(1.0 - hard_reject_alpha, df=3))
            if chi2 is not None else 30.665
        )

        # FIX BUG-3: correntropy_scale tied to tau so roll-off is principled
        self.correntropy_scale = float(correntropy_scale if correntropy_scale is not None
                                       else 0.5 * self.tau)

    def initialize(self, position: np.ndarray) -> None:
        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.x      = np.zeros(6)
        self.x[:3]  = np.asarray(position, dtype=float)
        # Position is initialized from the first measurement, but velocity is
        # not directly measured. Use a wider velocity prior to avoid an
        # artificial overconfidence transient at startup.
        self.P      = np.block([
            [0.25 * eye3, zero3],
            [zero3,       6.25 * eye3],
        ])
        self.initialized = True

    def _measurement_covariance(self, quality: float) -> np.ndarray:
        if self.mode == "fixed_kf":
            return self.R0.copy()
        # R_k = R0 / q_k : low quality -> large noise -> less trust in measurement
        q = max(float(quality), self.min_quality)
        return self.R0 / q

    def step(self, measurement: np.ndarray, quality: float,
             true_state: np.ndarray | None = None) -> FilterOutput:
        """Run one predict-update cycle.

        Parameters
        ----------
        true_state : array (6,) or None
            Needed to compute NEES.  If None, NEES is returned as nan.
        """
        if not self.initialized:
            self.initialize(measurement)

        # --- Predict ---
        x_prior = self.F @ self.x
        p_prior = self.F @ self.P @ self.F.T + self.Q

        # --- Innovation ---
        r_k       = self._measurement_covariance(quality)
        innovation = np.asarray(measurement, dtype=float) - self.H @ x_prior
        s_k        = self.H @ p_prior @ self.H.T + r_k
        s_inv      = np.linalg.pinv(s_k)
        nis        = float(innovation @ s_inv @ innovation)

        def consistency_metrics(x_hat: np.ndarray, p_hat: np.ndarray) -> tuple[float, float]:
            if true_state is None:
                return float("nan"), float("nan")
            err_state = np.asarray(true_state, dtype=float) - x_hat
            nees_val = float(err_state @ np.linalg.pinv(p_hat) @ err_state)
            err_pos = err_state[:3]
            p_pos = p_hat[:3, :3]
            pos_nees_val = float(err_pos @ np.linalg.pinv(p_pos) @ err_pos)
            return nees_val, pos_nees_val

        # --- Soft NIS gate + correntropy reweighting (proposed mode only) ---
        # FIX BUG-1: track soft reweight and hard reject separately
        soft_reweighted = False
        hard_rejected   = False
        weight  = 1.0
        r_eff   = r_k

        gate_enabled = self.step_count >= self.gate_warmup_steps
        if self.mode == "adaptive_r_gate" and gate_enabled and nis > self.tau:
            excess = nis - self.tau
            weight = float(np.exp(-0.5 * excess / self.correntropy_scale))
            if nis >= self.hard_tau:
                hard_rejected = True
            else:
                soft_reweighted = True
                # r_eff = r_k / weight: weight < 1 -> r_eff > r_k -> less trust
                r_eff = r_k / max(weight, 1e-12)

        # A hard reject must coast on the prediction. Inflating R to a huge
        # value is close but not identical, and it perturbs P over time.
        if hard_rejected:
            self.x = x_prior.copy()
            self.P = p_prior.copy()
            nees, pos_nees = consistency_metrics(self.x, self.P)
            self.step_count += 1
            return FilterOutput(
                state=self.x.copy(),
                covariance=self.P.copy(),
                nis=nis,
                nees=nees,
                pos_nees=pos_nees,
                soft_reweighted=False,
                hard_rejected=True,
                measurement_weight=weight,
                quality=float(quality),
            )

        # --- Update (Joseph form for numerical stability) ---
        s_eff  = self.H @ p_prior @ self.H.T + r_eff
        k_gain = p_prior @ self.H.T @ np.linalg.pinv(s_eff)
        self.x = x_prior + k_gain @ innovation
        ident  = np.eye(6)
        self.P = ((ident - k_gain @ self.H) @ p_prior @ (ident - k_gain @ self.H).T
                  + k_gain @ r_eff @ k_gain.T)
        self.step_count += 1

        # --- NEES (FIX BUG-4) ---
        # Full NEES has expectation 6; position-only NEES has expectation 3.
        nees, pos_nees = consistency_metrics(self.x, self.P)

        return FilterOutput(
            state=self.x.copy(),
            covariance=self.P.copy(),
            nis=nis,
            nees=nees,
            pos_nees=pos_nees,
            soft_reweighted=soft_reweighted,
            hard_rejected=hard_rejected,
            measurement_weight=weight,
            quality=float(quality),
        )


# ---------------------------------------------------------------------------
# Trajectory and sensing model
# ---------------------------------------------------------------------------
def make_true_trajectory(t: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((len(t), 6))
    x[:, 0] = 0.55 * t
    x[:, 1] = 2.2 * np.sin(0.18 * t) + 0.45 * np.sin(0.71 * t + 0.2)
    x[:, 2] = 2.0  + 0.25 * np.sin(0.11 * t + 0.6)
    x[:, 0] += 0.08 * np.sin(0.9 * t + rng.uniform(-0.3, 0.3))
    x[:, 1] += 0.06 * np.sin(1.1 * t + rng.uniform(-0.3, 0.3))
    dt = float(t[1] - t[0])
    x[:, 3:6] = np.gradient(x[:, :3], dt, axis=0)
    return x


def _sigmoid(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Smooth 0->1 transition used for fade ramps."""
    return 1.0 / (1.0 + np.exp(-(x - centre) / width))


def sensing_quality(t: np.ndarray, seed: int, ramp_width: float = 0.5) -> np.ndarray:
    """Synthetic ISAC quality q_k ∈ [0, 1] derived from normalised SINR.

    NOTE for paper: replace this with
        q_k = clip(SINR_k * G_r(theta_k) / SINR_ref, min_quality, 1.0)
    so that R_k = R0 / q_k is the true CRB-derived measurement covariance.

    FIX BUG-5: original used abrupt step at fade boundaries.
    Now uses smooth sigmoid ramps (width ≈ ramp_width seconds) which is
    physically consistent with Rician channel fading.
    """
    rng = np.random.default_rng(seed + 1000)
    q = 0.86 + 0.10 * np.sin(0.35 * t) + 0.04 * np.sin(1.4 * t + 0.5)
    q += rng.normal(0.0, 0.015, size=len(t))

    # Fade window 1: [12, 18] s  — depth 0.22
    fade1 = (_sigmoid(t, 12.0, ramp_width) * _sigmoid(-t, -18.0, ramp_width))
    q -= fade1 * (1.0 - 0.22) * q  # smoothly multiply by 0.22 inside window

    # Fade window 2: [28, 32] s  — depth 0.35
    fade2 = (_sigmoid(t, 28.0, ramp_width) * _sigmoid(-t, -32.0, ramp_width))
    q -= fade2 * (1.0 - 0.35) * q

    return np.clip(q, 0.04, 1.0)


def make_measurements(
    true_state: np.ndarray,
    q: np.ndarray,
    seed: int,
    base_sigma: float,
    stress_mode: str = "coupled",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 2000)
    n   = len(q)
    if stress_mode == "outlier_only":
        sigma_quality = np.ones_like(q)
    else:
        sigma_quality = q
    sigma = base_sigma / np.sqrt(np.clip(sigma_quality, 0.04, 1.0))
    meas  = true_state[:, :3] + rng.normal(0.0, sigma[:, None], size=(n, 3))

    if stress_mode == "noise_only":
        return meas, np.zeros(n, dtype=bool)

    # Burst outliers concentrated in poor-quality windows (physically motivated)
    burst = (
        ((np.arange(n) % 31) < 4)
        & (
            ((true_state[:, 0] > 7.0)  & (true_state[:, 0] < 11.0))
            | ((true_state[:, 0] > 16.0) & (true_state[:, 0] < 19.0))
        )
    )
    if stress_mode in {"outlier_only", "near_threshold"}:
        # Pure outlier ablations: fixed event rate, independent of q. This
        # isolates outlier robustness from sensing-quality-driven noise changes.
        random_outliers = rng.random(n) < 0.05
    else:
        random_outliers = rng.random(n) < (0.012 + 0.08 * (1.0 - q))
    outlier_mask = burst | random_outliers

    directions = rng.normal(0.0, 1.0, size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
    if stress_mode == "near_threshold":
        # Boundary test: some outliers sit near or below the chi-square gate.
        # This makes recall an empirical result rather than a guaranteed 1.0.
        magnitudes = rng.uniform(0.15, 0.45, size=n)
    else:
        magnitudes = rng.uniform(0.8, 2.2, size=n) * (1.0 + 1.8 * (1.0 - q))
    meas[outlier_mask] += directions[outlier_mask] * magnitudes[outlier_mask, None]
    return meas, outlier_mask


# ---------------------------------------------------------------------------
# LQR / LQG controller  (FIX BUG-6)
# ---------------------------------------------------------------------------
def build_lqr_gain(F: np.ndarray, B: np.ndarray,
                   Q_lqg: np.ndarray, R_lqg: np.ndarray) -> np.ndarray:
    """Compute discrete LQR gain K via DARE, or fall back to a simple gain."""
    if solve_discrete_are is not None:
        try:
            P_dare = solve_discrete_are(F, B, Q_lqg, R_lqg)
            K = np.linalg.solve(R_lqg + B.T @ P_dare @ B, B.T @ P_dare @ F)
            return K
        except Exception:
            pass
    # Fallback: proportional gain on position only
    K = np.zeros((3, 6))
    K[:, :3] = 2.0 * np.eye(3)
    return K


# ---------------------------------------------------------------------------
# Per-seed simulation
# ---------------------------------------------------------------------------
def run_seed(
    seed: int,
    output_root: Path,
    duration_s: float = 40.0,
    dt: float = 0.05,
    quality_noise_std: float = 0.10,
    stress_mode: str = "coupled",
    dt_delay_steps: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t          = np.arange(0.0, duration_s, dt)
    true_state = make_true_trajectory(t, seed)
    q          = sensing_quality(t, seed)
    q_rng      = np.random.default_rng(seed + 3000)
    q_filter   = np.clip(q * (1.0 + q_rng.normal(0.0, quality_noise_std, size=len(q))),
                         0.04, 1.0)
    base_sigma = 0.065
    measurements, outlier_mask = make_measurements(
        true_state, q, seed, base_sigma, stress_mode=stress_mode
    )

    modes  = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    labels = {
        "fixed_kf":        "Fixed-R KF",
        "adaptive_r":      "Quality-adaptive R",
        "adaptive_r_gate": "Adaptive R + NIS gate",
    }
    filters = {
        mode: ISACQualityKalmanFilter(
            dt=dt, base_sigma_meas=base_sigma, sigma_accel=0.55, mode=mode
        )
        for mode in modes
    }

    # --- LQR setup (FIX BUG-6) ---
    eye3  = np.eye(3)
    zero3 = np.zeros((3, 3))
    F_sys = np.block([[eye3, dt * eye3], [zero3, eye3]])
    B_sys = np.block([[0.5 * dt**2 * eye3], [dt * eye3]])
    Q_lqg = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_lqg = np.eye(3) * 0.1
    K_lqr = build_lqr_gain(F_sys, B_sys, Q_lqg, R_lqg)

    # Per-method downstream-cost accumulators
    lqg_cost   = {m: 0.0 for m in modes}
    dt_buffers = {m: [] for m in modes}

    rows: list[dict] = []

    for k, tk in enumerate(t):
        for mode, filt in filters.items():
            out = filt.step(measurements[k], q_filter[k], true_state=true_state[k])

            # Estimation-induced LQG surrogate (FIX BUG-6)
            # Compare the action selected from the estimate with the oracle
            # action selected from the true state. This isolates downstream
            # decision error caused by sensing/estimation corruption.
            u_hat      = -K_lqr @ out.state
            u_true     = -K_lqr @ true_state[k]
            u_err      = u_hat - u_true
            est_err_state = true_state[k] - out.state
            step_cost  = (float(est_err_state @ Q_lqg @ est_err_state)
                          + float(u_err @ R_lqg @ u_err))
            lqg_cost[mode] += step_cost

            # Digital-twin mismatch (FIX BUG-7)
            # Use a delayed copy of the filter estimate to make DT mismatch a
            # downstream latency/fidelity metric rather than a duplicate RMSE.
            dt_buffers[mode].append(out.state.copy())
            if dt_delay_steps > 0 and len(dt_buffers[mode]) > dt_delay_steps:
                dt_state = dt_buffers[mode].pop(0)
            else:
                dt_state = out.state
            dt_err = float(np.linalg.norm(true_state[k, :3] - dt_state[:3]))

            err      = float(np.linalg.norm(out.state[:3] - true_state[k, :3]))
            meas_err = float(np.linalg.norm(measurements[k] - true_state[k, :3]))

            rows.append({
                "seed":               seed,
                "t":                  tk,
                "method":             mode,
                "method_label":       labels[mode],
                "quality":            q_filter[k],
                "true_quality":       q[k],
                "stress_mode":        stress_mode,
                "true_x":             true_state[k, 0],
                "true_y":             true_state[k, 1],
                "true_z":             true_state[k, 2],
                "meas_error_m":       meas_err,
                "estimate_error_m":   err,
                "nis":                out.nis,
                "nees":               out.nees,           # FIX BUG-4
                "pos_nees":           out.pos_nees,
                "soft_reweighted":    float(out.soft_reweighted),  # FIX BUG-1
                "hard_rejected":      float(out.hard_rejected),    # FIX BUG-1
                "measurement_weight": out.measurement_weight,
                "dt_error_m":         dt_err,             # FIX BUG-7
                "dt_delay_steps":     int(dt_delay_steps),
                "is_outlier":         float(outlier_mask[k]),
                "is_fade":            float((12.0 <= tk <= 18.0) or (28.0 <= tk <= 32.0)),
            })

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / f"timeseries_seed{seed}.csv", index=False)

    # --- Per-method summary ---
    summaries = []

    def effective_sample_size(values: pd.Series, max_lag: int = 80) -> int:
        """Estimate effective sample size for autocorrelated consistency stats."""
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 4:
            return max(n, 1)
        x = x - np.mean(x)
        var = float(np.dot(x, x) / n)
        if var <= 1e-12:
            return n
        rho_sum = 0.0
        for lag in range(1, min(max_lag, n - 1) + 1):
            rho = float(np.dot(x[:-lag], x[lag:]) / ((n - lag) * var))
            if rho <= 0.0:
                break
            rho_sum += rho
        tau_int = 1.0 + 2.0 * rho_sum
        return max(1, int(round(n / tau_int)))

    for mode in modes:
        sub     = df[df["method"] == mode]
        fade    = sub[sub["is_fade"]    == 1.0]
        outlier = sub[sub["is_outlier"] == 1.0]
        nominal = sub[(sub["is_fade"] == 0.0) & (sub["is_outlier"] == 0.0)]
        gate_fired = (sub["soft_reweighted"] + sub["hard_rejected"]) > 0.0
        true_outlier = sub["is_outlier"] > 0.0
        tp = int((gate_fired & true_outlier).sum())
        fp = int((gate_fired & ~true_outlier).sum())
        fn = int((~gate_fired & true_outlier).sum())
        tn = int((~gate_fired & ~true_outlier).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        gate_fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        nees_n_eff = effective_sample_size(sub["nees"])
        pos_nees_n_eff = effective_sample_size(sub["pos_nees"])
        nees_mean_bound = 6.0 + 3.0 * np.sqrt(12.0 / nees_n_eff)
        pos_nees_mean_bound = 3.0 + 3.0 * np.sqrt(6.0 / pos_nees_n_eff)

        def rmse(s: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(s)))) if len(s) else float("nan")

        summaries.append({
            "seed":                  seed,
            "method":                mode,
            "method_label":          labels[mode],
            "rmse_m":                rmse(sub["estimate_error_m"]),
            "mean_error_m":          float(sub["estimate_error_m"].mean()),
            "fade_rmse_m":           rmse(fade["estimate_error_m"]),
            "outlier_rmse_m":        rmse(outlier["estimate_error_m"]),
            "nominal_rmse_m":        rmse(nominal["estimate_error_m"]),
            "mean_nis":              float(sub["nis"].mean()),
            "mean_nees":             float(sub["nees"].mean()),         # FIX BUG-4
            "mean_pos_nees":         float(sub["pos_nees"].mean()),
            "nees_n_eff":            nees_n_eff,
            "nees_mean_bound":       nees_mean_bound,
            "nees_consistency_ok":   float(sub["nees"].mean() < nees_mean_bound),
            "pos_nees_n_eff":        pos_nees_n_eff,
            "pos_nees_mean_bound":    pos_nees_mean_bound,
            "pos_nees_consistency_ok": float(sub["pos_nees"].mean() < pos_nees_mean_bound),
            "soft_reweight_rate":    float(sub["soft_reweighted"].mean()),  # FIX BUG-1
            "hard_reject_rate":      float(sub["hard_rejected"].mean()),    # FIX BUG-1
            "gate_precision":        precision,
            "gate_recall":           recall,
            "gate_false_positive_rate": gate_fpr,
            "gate_tp":               tp,
            "gate_fp":               fp,
            "gate_fn":               fn,
            "gate_tn":               tn,
            "mean_measurement_weight": float(sub["measurement_weight"].mean()),
            "lqg_cost":              lqg_cost[mode],                    # FIX BUG-6
            "mean_dt_error_m":       float(sub["dt_error_m"].mean()),   # FIX BUG-7
            "rmse_dt_error_m":       rmse(sub["dt_error_m"]),           # FIX BUG-7
            "outlier_count":         int(outlier_mask.sum()),
        })

    summary = pd.DataFrame(summaries)
    return df, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def aggregate_and_plot(output_root: Path) -> pd.DataFrame:
    all_summary = pd.concat(
        [pd.read_csv(p) for p in sorted(output_root.glob("summary_seed*.csv"))],
        ignore_index=True,
    )
    agg = all_summary.groupby(["method", "method_label"]).agg(
        rmse_m_mean            =("rmse_m",             "mean"),
        rmse_m_std             =("rmse_m",             "std"),
        fade_rmse_m_mean       =("fade_rmse_m",        "mean"),
        outlier_rmse_m_mean    =("outlier_rmse_m",     "mean"),
        nominal_rmse_m_mean    =("nominal_rmse_m",     "mean"),
        mean_nees_mean         =("mean_nees",          "mean"),   # BUG-4
        mean_pos_nees_mean     =("mean_pos_nees",      "mean"),
        soft_reweight_rate_mean=("soft_reweight_rate", "mean"),   # BUG-1
        hard_reject_rate_mean  =("hard_reject_rate",   "mean"),   # BUG-1
        gate_precision_mean    =("gate_precision",     "mean"),
        gate_recall_mean       =("gate_recall",        "mean"),
        gate_fpr_mean          =("gate_false_positive_rate", "mean"),
        lqg_cost_mean          =("lqg_cost",           "mean"),   # BUG-6
        lqg_cost_std           =("lqg_cost",           "std"),    # BUG-6
        mean_dt_error_m_mean   =("mean_dt_error_m",    "mean"),   # BUG-7
        count                  =("seed",               "count"),
    ).reset_index()
    agg.to_csv(output_root / "summary_aggregate.csv", index=False)

    order  = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    colors = ["#7a828c", "#2a9d8f", "#2f6fed"]
    agg    = agg.set_index("method").reindex(order).reset_index()

    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
    })

    # --- Figure 1: RMSE bar chart ---
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(agg))
    ax.bar(x, agg["rmse_m_mean"], color=colors, yerr=agg["rmse_m_std"], capsize=4)
    ax.set_xticks(x, agg["method_label"], rotation=10, ha="right")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Sensing-quality adaptation reduces pre-communication state error")
    rmse_top = float((agg["rmse_m_mean"] + agg["rmse_m_std"].fillna(0.0)).max())
    ax.set_ylim(0, rmse_top * 1.28)
    fixed_rmse = float(agg.loc[agg["method"] == "fixed_kf", "rmse_m_mean"].iloc[0])
    for i, row in agg.iterrows():
        pct   = (fixed_rmse - row["rmse_m_mean"]) / fixed_rmse * 100.0
        label = "baseline" if row["method"] == "fixed_kf" else f"{pct:.1f}% lower"
        ax.text(i, row["rmse_m_mean"] + max(agg["rmse_m_mean"]) * 0.05,
                label, ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_root / "fig_rmse_comparison.png", dpi=320)
    fig.savefig(output_root / "fig_rmse_comparison.pdf")
    plt.close(fig)

    # --- Figure 2: downstream control-decision cost bar chart (FIX BUG-6) ---
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x, agg["lqg_cost_mean"], color=colors, yerr=agg["lqg_cost_std"], capsize=4)
    ax.set_xticks(x, agg["method_label"], rotation=10, ha="right")
    ax.set_ylabel("Cumulative decision cost")
    ax.set_title("Downstream ISAC metric: estimation-induced control cost")
    lqg_top = float((agg["lqg_cost_mean"] + agg["lqg_cost_std"].fillna(0.0)).max())
    ax.set_ylim(0, lqg_top * 1.28)
    fixed_lqg = float(agg.loc[agg["method"] == "fixed_kf", "lqg_cost_mean"].iloc[0])
    for i, row in agg.iterrows():
        pct   = (fixed_lqg - row["lqg_cost_mean"]) / fixed_lqg * 100.0
        label = "baseline" if row["method"] == "fixed_kf" else f"{pct:.1f}% lower"
        ax.text(i, row["lqg_cost_mean"] + max(agg["lqg_cost_mean"]) * 0.05,
                label, ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_root / "fig_lqg_cost.png", dpi=320)
    fig.savefig(output_root / "fig_lqg_cost.pdf")
    plt.close(fig)

    # --- Figure 3: digital-twin mismatch bar chart (FIX BUG-7) ---
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x, agg["mean_dt_error_m_mean"], color=colors, capsize=4)
    ax.set_xticks(x, agg["method_label"], rotation=10, ha="right")
    ax.set_ylabel("Mean DT mismatch (m)")
    ax.set_title("Cleaner sensing reduces digital-twin state mismatch")
    ax.set_ylim(0, float(agg["mean_dt_error_m_mean"].max()) * 1.32)
    fixed_dt = float(agg.loc[agg["method"] == "fixed_kf", "mean_dt_error_m_mean"].iloc[0])
    for i, row in agg.iterrows():
        pct = (fixed_dt - row["mean_dt_error_m_mean"]) / fixed_dt * 100.0
        label = "baseline" if row["method"] == "fixed_kf" else f"{pct:.1f}% lower"
        ax.text(i, row["mean_dt_error_m_mean"] + max(agg["mean_dt_error_m_mean"]) * 0.05,
                label, ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_root / "fig_dt_error_comparison.png", dpi=320)
    fig.savefig(output_root / "fig_dt_error_comparison.pdf")
    plt.close(fig)

    # --- Figure 4: error timeline (seed 1) ---
    ts   = pd.read_csv(output_root / "timeseries_seed1.csv")
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    method_label_map = agg.set_index("method")["method_label"].to_dict()
    for method, color in zip(order, colors):
        sub    = ts[ts["method"] == method]
        smooth = sub["estimate_error_m"].rolling(9, min_periods=1, center=True).mean()
        ax1.plot(sub["t"], smooth, label=method_label_map[method],
                 color=color, linewidth=2.0)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Estimate error (m)")
    ax1.axvspan(12, 18, color="#f28e2b", alpha=0.12, label="SINR fade")
    ax1.axvspan(28, 32, color="#f28e2b", alpha=0.12)
    ax2 = ax1.twinx()
    q_col = ts[ts["method"] == "fixed_kf"][["t", "true_quality", "quality"]]
    ax2.plot(q_col["t"], q_col["true_quality"], color="#c43c39",
             linestyle="--", linewidth=1.8, label="true sensing quality")
    ax2.plot(q_col["t"], q_col["quality"], color="#c43c39",
             linestyle=":", linewidth=1.2, alpha=0.8, label="estimated quality")
    ax2.set_ylabel("Sensing quality")
    ax2.set_ylim(0, 1.05)
    lines,  lbls  = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, lbls + lbls2, frameon=False, loc="upper right")
    ax1.set_title("Adaptive R + NIS gate suppresses error during low-quality ISAC sensing")
    fig.tight_layout()
    fig.savefig(output_root / "fig_error_timeline.png", dpi=320)
    fig.savefig(output_root / "fig_error_timeline.pdf")
    plt.close(fig)

    # --- Figure 5: NEES timeline (FIX BUG-4) ---
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    nees_chi2_mean = 6.0   # E[NEES] = n_states under consistency
    nees_chi2_95   = 12.59 # chi2(6) 95th percentile
    for method, color in zip(order, colors):
        sub    = ts[ts["method"] == method]
        smooth = sub["nees"].rolling(19, min_periods=1, center=True).mean()
        ax.plot(sub["t"], smooth, label=method_label_map[method],
                color=color, linewidth=1.8)
    ax.axhline(nees_chi2_mean, color="k",  linestyle="--", linewidth=1.0, label="E[NEES]=6 (consistent)")
    ax.axhline(nees_chi2_95,   color="k",  linestyle=":",  linewidth=1.0, label="95th pct bound")
    ax.axvspan(12, 18, color="#f28e2b", alpha=0.10)
    ax.axvspan(28, 32, color="#f28e2b", alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES (state-space consistency)")
    ax.set_title("NEES consistency — lower and closer to 6.0 is better")
    ax.legend(frameon=False)
    ax.set_ylim(0, min(ax.get_ylim()[1], 200))
    fig.tight_layout()
    fig.savefig(output_root / "fig_nees_timeline.png", dpi=320)
    fig.savefig(output_root / "fig_nees_timeline.pdf")
    plt.close(fig)

    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISAC sensing-quality-aware KF prototype"
    )
    parser.add_argument("--seeds",       type=int, default=30)
    parser.add_argument("--output-root", default="outputs_isac_sensor_prototype")
    parser.add_argument("--duration",    type=float, default=40.0)
    parser.add_argument("--dt",          type=float, default=0.05)
    parser.add_argument("--quality-noise-std", type=float, default=0.10,
                        help="relative std dev of the filter's q_k estimate")
    parser.add_argument("--dt-delay-steps", type=int, default=10,
                        help="digital-twin estimate delay in filter steps; 0 makes DT error equal estimate error")
    parser.add_argument("--stress-mode",
                        choices=["coupled", "noise_only", "outlier_only", "near_threshold"],
                        default="coupled",
                        help=("coupled: low q raises noise and outlier rate; "
                              "noise_only/outlier_only are ablations; "
                              "near_threshold uses small outliers for gate-boundary testing"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for seed in range(1, args.seeds + 1):
        _, summary = run_seed(seed, output_root,
                              duration_s=args.duration,
                              dt=args.dt,
                              quality_noise_std=args.quality_noise_std,
                              stress_mode=args.stress_mode,
                              dt_delay_steps=args.dt_delay_steps)
        summary.to_csv(output_root / f"summary_seed{seed}.csv", index=False)
        if seed % 5 == 0:
            print(f"  seed {seed}/{args.seeds} done")

    agg = aggregate_and_plot(output_root)

    # Print summary table
    cols = ["method_label", "rmse_m_mean", "lqg_cost_mean",
            "mean_nees_mean", "mean_pos_nees_mean",
            "soft_reweight_rate_mean", "hard_reject_rate_mean",
            "gate_precision_mean", "gate_recall_mean",
            "gate_fpr_mean", "mean_dt_error_m_mean"]
    print("\n" + agg[cols].to_string(index=False))

    fixed_rmse = float(agg.loc[agg["method"] == "fixed_kf", "rmse_m_mean"].iloc[0])
    fixed_lqg  = float(agg.loc[agg["method"] == "fixed_kf", "lqg_cost_mean"].iloc[0])
    for _, row in agg.iterrows():
        if row["method"] == "fixed_kf":
            continue
        rmse_imp = (fixed_rmse - row["rmse_m_mean"]) / fixed_rmse * 100
        lqg_imp  = (fixed_lqg  - row["lqg_cost_mean"])  / fixed_lqg  * 100
        print(f"\n{row['method_label']}:"
              f"  RMSE {rmse_imp:.1f}% better"
              f"  |  LQG cost {lqg_imp:.1f}% better"
              f"  |  mean NEES {row['mean_nees_mean']:.2f}")

    print(f"\nOutputs written to: {output_root}")


if __name__ == "__main__":
    main()
