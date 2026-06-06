"""Replay EuRoC MAV ground truth through the ISAC prototype filter.

The EuRoC dataset provides accurate 3D position ground truth. This script uses
that ground truth as the true trajectory and simulates an ISAC sensing channel:

- SINR/quality-driven Gaussian measurement noise, R_k = R0 / q_k
- synthetic sensing-quality fades tied to dynamic flight segments
- random and burst outlier injections

It then runs the same three filter variants used by the standalone ISAC
prototype:

- Fixed-R KF
- quality-adaptive R KF
- adaptive R + NIS/correntropy gate

Input format is TUM/open_vins EuRoC export:

    # timestamp(s) tx ty tz qx qy qz qw
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.linalg import solve_discrete_are
    from scipy.stats import chi2
except ImportError:  # pragma: no cover
    chi2 = None
    solve_discrete_are = None


NIS_ALPHA_SOFT = 0.05
NIS_ALPHA_HARD = 1e-6
BASE_SIGMA_MEAS = 0.065
SIGMA_ACCEL = 2.5
QUALITY_NOISE = 0.10
MIN_QUALITY = 0.04
DT_DELAY_STEPS = 10


@dataclass
class FilterOutput:
    state: np.ndarray
    covariance: np.ndarray
    nis: float
    nees: float
    pos_nees: float
    soft_reweighted: bool
    hard_rejected: bool
    measurement_weight: float
    quality: float


class ISACKalmanFilter:
    """Constant-velocity KF with ISAC quality adaptation and integrity gating."""

    def __init__(self, dt: float, mode: str):
        self.dt = float(dt)
        self.mode = mode
        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.F = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.H = np.block([eye3, zero3])

        q_pos = 0.25 * self.dt**4 * SIGMA_ACCEL**2
        q_cross = 0.50 * self.dt**3 * SIGMA_ACCEL**2
        q_vel = self.dt**2 * SIGMA_ACCEL**2
        self.Q = np.block(
            [[q_pos * eye3, q_cross * eye3], [q_cross * eye3, q_vel * eye3]]
        )
        self.R0 = np.eye(3) * BASE_SIGMA_MEAS**2
        self.x = np.zeros(6)
        self.P = np.block([[0.25 * eye3, zero3], [zero3, 6.25 * eye3]])
        self.initialized = False
        self.step_count = 0
        self.coast_count = 0
        self.max_coast = 5

        self.tau_soft = (
            float(chi2.ppf(1 - NIS_ALPHA_SOFT, df=3)) if chi2 is not None else 7.815
        )
        self.tau_hard = (
            float(chi2.ppf(1 - NIS_ALPHA_HARD, df=3)) if chi2 is not None else 30.665
        )
        self.cs_scale = 0.5 * self.tau_soft

    def _measurement_covariance(self, q: float) -> np.ndarray:
        if self.mode == "fixed_kf":
            return self.R0.copy()
        return self.R0 / max(float(q), MIN_QUALITY)

    def step(
        self,
        measurement: np.ndarray,
        quality: float,
        true_state: np.ndarray | None = None,
    ) -> FilterOutput:
        if not self.initialized:
            self.x[:3] = np.asarray(measurement, dtype=float)
            self.initialized = True

        x_prior = self.F @ self.x
        p_prior = self.F @ self.P @ self.F.T + self.Q

        r_k = self._measurement_covariance(quality)
        innovation = np.asarray(measurement, dtype=float) - self.H @ x_prior
        s_k = self.H @ p_prior @ self.H.T + r_k
        s_inv = np.linalg.pinv(s_k)
        nis = float(innovation @ s_inv @ innovation)

        soft_reweighted = False
        hard_rejected = False
        weight = 1.0
        r_eff = r_k

        if self.mode == "adaptive_r_gate" and self.step_count >= 50:
            if nis >= self.tau_hard:
                hard_rejected = True
            elif nis > self.tau_soft:
                excess = nis - self.tau_soft
                weight = max(float(np.exp(-0.5 * excess / self.cs_scale)), 1e-12)
                r_eff = r_k / weight
                soft_reweighted = True

        if hard_rejected and self.coast_count < self.max_coast:
            self.x = x_prior.copy()
            self.P = p_prior.copy()
            self.coast_count += 1
        else:
            if hard_rejected:
                hard_rejected = False
                r_eff = r_k
            self.coast_count = 0
            s_eff = self.H @ p_prior @ self.H.T + r_eff
            gain = p_prior @ self.H.T @ np.linalg.pinv(s_eff)
            self.x = x_prior + gain @ innovation
            ident = np.eye(6)
            ikh = ident - gain @ self.H
            self.P = ikh @ p_prior @ ikh.T + gain @ r_eff @ gain.T

        self.step_count += 1

        nees = float("nan")
        pos_nees = float("nan")
        if true_state is not None:
            err = np.asarray(true_state, dtype=float) - self.x
            nees = float(err @ np.linalg.pinv(self.P) @ err)
            err_pos = err[:3]
            pos_nees = float(err_pos @ np.linalg.pinv(self.P[:3, :3]) @ err_pos)

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


def load_euroc_gt(path: Path) -> np.ndarray:
    """Load TUM-format EuRoC ground truth."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(p) for p in parts[:8]])
    if not rows:
        raise ValueError(f"No TUM-format rows found in {path}")
    return np.array(rows, dtype=float)


def resample_to_dt(gt: np.ndarray, dt: float) -> np.ndarray:
    """Linearly interpolate ground truth to a fixed control timestep."""
    t0, t1 = gt[0, 0], gt[-1, 0]
    t_uniform = np.arange(t0, t1, dt)
    out = np.zeros((len(t_uniform), gt.shape[1]))
    out[:, 0] = t_uniform
    for col in range(1, gt.shape[1]):
        out[:, col] = np.interp(t_uniform, gt[:, 0], gt[:, col])
    return out


def make_quality(n: int, speed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a proxy ISAC sensing-quality signal from MAV dynamics."""
    speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)
    q = 0.90 - 0.30 * speed_norm
    q += rng.normal(0.0, 0.015, size=n)

    def sigmoid(x: np.ndarray, centre: float, width: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x - centre) / width))

    t_frac = np.linspace(0.0, 1.0, n)
    fade1 = sigmoid(t_frac, 0.20, 0.01) * sigmoid(-t_frac, -0.35, 0.01)
    q -= fade1 * (1.0 - 0.22) * q
    fade2 = sigmoid(t_frac, 0.65, 0.01) * sigmoid(-t_frac, -0.75, 0.01)
    q -= fade2 * (1.0 - 0.35) * q
    return np.clip(q, MIN_QUALITY, 1.0)


def make_measurements(
    true_pos: np.ndarray, q: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(q)
    sigma = BASE_SIGMA_MEAS / np.sqrt(np.clip(q, MIN_QUALITY, 1.0))
    measurements = true_pos + rng.normal(0.0, sigma[:, None], size=(n, 3))

    random_outliers = rng.random(n) < (0.012 + 0.08 * (1.0 - q))
    burst = (
        ((np.arange(n) % 31) < 4)
        & (np.arange(n) > n // 4)
        & (np.arange(n) < 3 * n // 4)
    )
    outlier_mask = random_outliers | burst

    directions = rng.normal(0.0, 1.0, (n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
    magnitudes = rng.uniform(0.8, 2.2, n) * (1.0 + 1.8 * (1.0 - q))
    measurements[outlier_mask] += directions[outlier_mask] * magnitudes[outlier_mask, None]
    return measurements, outlier_mask


def build_lqr(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye3 = np.eye(3)
    zero3 = np.zeros((3, 3))
    state_mat = np.block([[eye3, dt * eye3], [zero3, eye3]])
    input_mat = np.block([[0.5 * dt**2 * eye3], [dt * eye3]])
    q_lqg = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    r_lqg = np.eye(3) * 0.1
    if solve_discrete_are is not None:
        try:
            p_lqr = solve_discrete_are(state_mat, input_mat, q_lqg, r_lqg)
            gain = np.linalg.solve(
                r_lqg + input_mat.T @ p_lqr @ input_mat, input_mat.T @ p_lqr @ state_mat
            )
            return gain, q_lqg, r_lqg
        except Exception:
            pass
    gain = np.zeros((3, 6))
    gain[:, :3] = 2.0 * eye3
    return gain, q_lqg, r_lqg


def replay(
    gt_path: Path,
    dt: float = 0.05,
    quality_noise: float = QUALITY_NOISE,
    seed: int = 0,
) -> pd.DataFrame:
    gt = resample_to_dt(load_euroc_gt(gt_path), dt)
    n = len(gt)
    time_s = gt[:, 0] - gt[0, 0]
    pos = gt[:, 1:4]
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    rng = np.random.default_rng(seed)
    q_true = make_quality(n, speed, rng)
    q_filter_rng = np.random.default_rng(seed + 3000)
    q_filter = np.clip(
        q_true * (1.0 + q_filter_rng.normal(0.0, quality_noise, n)),
        MIN_QUALITY,
        1.0,
    )
    measurements, outlier_mask = make_measurements(
        pos, q_true, np.random.default_rng(seed + 2000)
    )
    true_state = np.hstack([pos, vel])

    modes = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    labels = {
        "fixed_kf": "Fixed-R KF",
        "adaptive_r": "Quality-adaptive R",
        "adaptive_r_gate": "Adaptive R + NIS gate",
    }
    filters = {mode: ISACKalmanFilter(dt=dt, mode=mode) for mode in modes}
    k_lqr, q_lqg, r_lqg = build_lqr(dt)
    lqg_cost = {mode: 0.0 for mode in modes}
    dt_buffer = {mode: [] for mode in modes}

    rows = []
    for k in range(n):
        for mode, filt in filters.items():
            out = filt.step(measurements[k], q_filter[k], true_state=true_state[k])

            u_hat = -k_lqr @ out.state
            u_true = -k_lqr @ true_state[k]
            err_state = true_state[k] - out.state
            err_u = u_hat - u_true
            lqg_cost[mode] += float(err_state @ q_lqg @ err_state + err_u @ r_lqg @ err_u)

            dt_buffer[mode].append(out.state.copy())
            if len(dt_buffer[mode]) > DT_DELAY_STEPS:
                dt_state = dt_buffer[mode].pop(0)
            else:
                dt_state = out.state
            dt_err = float(np.linalg.norm(true_state[k, :3] - dt_state[:3]))

            rows.append(
                {
                    "t": time_s[k],
                    "mode": mode,
                    "label": labels[mode],
                    "est_err_m": float(np.linalg.norm(out.state[:3] - true_state[k, :3])),
                    "nis": out.nis,
                    "nees": out.nees,
                    "pos_nees": out.pos_nees,
                    "soft_rew": float(out.soft_reweighted),
                    "hard_rej": float(out.hard_rejected),
                    "dt_err_m": dt_err,
                    "is_outlier": float(outlier_mask[k]),
                    "quality": q_filter[k],
                    "true_quality": q_true[k],
                    "speed_ms": speed[k],
                }
            )

    df = pd.DataFrame(rows)
    for mode in modes:
        df.loc[df["mode"] == mode, "lqg_cost_total"] = lqg_cost[mode]
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mode, sub in df.groupby("mode"):
        label = sub["label"].iloc[0]
        records.append(
            {
                "method": mode,
                "label": label,
                "rmse_m": float(np.sqrt(np.mean(sub["est_err_m"] ** 2))),
                "mean_nees": float(sub["nees"].mean()),
                "mean_pos_nees": float(sub["pos_nees"].mean()),
                "lqg_cost": float(sub["lqg_cost_total"].iloc[0]),
                "mean_dt_err_m": float(sub["dt_err_m"].mean()),
                "soft_rew_rate": float(sub["soft_rew"].mean()),
                "hard_rej_rate": float(sub["hard_rej"].mean()),
                "outlier_count": int(df[df["mode"] == "fixed_kf"]["is_outlier"].sum()),
            }
        )
    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, seq_name: str, out_dir: Path) -> None:
    colors = {
        "fixed_kf": "#7a828c",
        "adaptive_r": "#2a9d8f",
        "adaptive_r_gate": "#2f6fed",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"EuRoC {seq_name} ISAC Filter Replay", fontweight="bold")

    ax = axes[0, 0]
    for mode, sub in df.groupby("mode"):
        smooth = sub["est_err_m"].rolling(15, center=True, min_periods=1).mean()
        ax.plot(sub["t"], smooth, color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Estimation error over time")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for mode, sub in df.groupby("mode"):
        smooth = sub["nees"].rolling(25, center=True, min_periods=1).mean()
        ax.plot(sub["t"], smooth, color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.axhline(6.0, color="k", ls="--", lw=1, label="E[NEES]=6")
    ax.set_ylim(0, min(ax.get_ylim()[1], 150))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_title("Filter consistency")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    q_col = df[df["mode"] == "fixed_kf"][["t", "true_quality", "speed_ms"]]
    ax2 = ax.twinx()
    ax.plot(q_col["t"], q_col["true_quality"], color="#c43c39", lw=1.5, label="q_k")
    ax2.plot(
        q_col["t"],
        q_col["speed_ms"],
        color="#888",
        lw=1.0,
        alpha=0.6,
        ls="--",
        label="speed",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sensing quality q_k")
    ax2.set_ylabel("Speed (m/s)")
    ax.set_title("Quality proxy from MAV speed")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)

    ax = axes[1, 1]
    for mode, sub in df.groupby("mode"):
        smooth = sub["dt_err_m"].rolling(15, center=True, min_periods=1).mean()
        ax.plot(sub["t"], smooth, color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DT delay error (m)")
    ax.set_title(f"Digital-twin mismatch ({DT_DELAY_STEPS}-step delay)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"euroc_{seq_name}_replay.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"euroc_{seq_name}_replay.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay EuRoC MAV GT through ISAC filter")
    parser.add_argument("--gt", required=True, help="Path to TUM-format ground truth .txt")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="euroc_out")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    out_dir = Path(args.out)
    seq_name = gt_path.stem

    print(f"Loading {gt_path.name} ...")
    df = replay(gt_path, dt=args.dt, seed=args.seed)
    summary = summarise(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"euroc_{seq_name}_timeseries.csv", index=False)
    summary.to_csv(out_dir / f"euroc_{seq_name}_summary.csv", index=False)

    print(f"\n{'Label':25s} {'RMSE':>8} {'NEES':>8} {'LQG cost':>12} {'DT err':>8}")
    print("-" * 65)
    fixed_rmse = float(summary.loc[summary["method"] == "fixed_kf", "rmse_m"].iloc[0])
    fixed_lqg = float(summary.loc[summary["method"] == "fixed_kf", "lqg_cost"].iloc[0])
    for _, row in summary.iterrows():
        rmse_improvement = (
            (fixed_rmse - row.rmse_m) / fixed_rmse * 100 if row.method != "fixed_kf" else 0.0
        )
        cost_improvement = (
            (fixed_lqg - row.lqg_cost) / fixed_lqg * 100 if row.method != "fixed_kf" else 0.0
        )
        suffix = (
            f"  RMSE↓{rmse_improvement:.1f}%  Cost↓{cost_improvement:.1f}%"
            if row.method != "fixed_kf"
            else "  [baseline]"
        )
        print(
            f"{row.label:25s} {row.rmse_m:8.4f} {row.mean_nees:8.2f} "
            f"{row.lqg_cost:12.1f} {row.mean_dt_err_m:8.4f}{suffix}"
        )

    plot_results(df, seq_name, out_dir)
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
