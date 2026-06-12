"""
isac_dt_sync_aerpaw.py
======================
Paper 3 AERPAW adapter for sensing-quality-aware digital twin synchronization.

This script does not reopen the 7.2 GB raw AERPAW zip.  It consumes the
prediction-gated association streams already produced by aerpaw_radar_replay_v2:

    *_v2_associated_radar_measurements.csv

Those streams are derived from the raw AERPAW/Fortem radar logs and contain the
per-timestep quantities needed by the DT sync manager:

    xhat_k : radar-derived position measurement (meas_x, meas_y, meas_z)
    P_k    : radar measurement covariance proxy (r_xx, r_yy, r_zz)
    q_k    : radar sensing quality from sinr_db
    x_k    : associated UAV ground truth (gt_x, gt_y, gt_z)

The synchronization policy is identical to isac_dt_sync_v3.py: SYNC/FUSE/HOLD
using confidence-weighted divergence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aerpaw_radar_replay_v2 import (
    DEFAULT_SIGMA_M,
    RadarAdaptiveKalmanFilter,
)
from aerpaw_radar_replay import quality_from_sinr
from isac_dt_sync_v2 import (
    DigitalTwin,
    build_lqr,
    ALPHA_CONF,
    DELTA_CONF,
    TAU_LOW,
    TAU_HIGH,
    FUSE_W,
    MAX_HOLD,
    T_PERIOD,
    TAU_EVENT,
    COLORS,
    LABELS,
    Q_MIN,
)

METHODS = ["proposed", "periodic", "event", "unconditional"]


def load_aerpaw_stream(path: Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f"Too few AERPAW samples in {path}")
    dt = float(np.median(np.diff(df["time"].to_numpy(float))))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.256
    return df, dt


def _radar_filter_mode(filter_mode: str) -> str:
    return "fixed_kf" if filter_mode == "fixed" else "adaptive_r_gate"


def radar_nominal_trace(stream: pd.DataFrame, dt: float, floor_sigma_m: float, filter_mode: str) -> float:
    meas = stream[["meas_x", "meas_y", "meas_z"]].to_numpy(float)
    gt_pos = stream[["gt_x", "gt_y", "gt_z"]].to_numpy(float)
    gt_vel = np.gradient(gt_pos, dt, axis=0)
    true_state = np.hstack([gt_pos, gt_vel])
    q = quality_from_sinr(stream["sinr_db"].fillna(stream["sinr_db"].median()).to_numpy(float))
    r_mats = [np.diag([row.r_xx, row.r_yy, row.r_zz]) for row in stream.itertuples()]
    filt = RadarAdaptiveKalmanFilter(
        dt=dt,
        mode=_radar_filter_mode(filter_mode),
        base_sigma_m=floor_sigma_m,
    )
    traces = []
    for k in range(len(stream)):
        filt.step(meas[k], q[k], r_mats[k], true_state[k])
        if k > 5:
            traces.append(float(np.trace(filt.P)))
    return float(np.median(traces)) if traces else float(np.trace(filt.P))


def run_one_stream(
    csv_path: Path,
    floor_sigma_m: float = DEFAULT_SIGMA_M,
    tau_low: float = 20.0,
    tau_high: float = 100.0,
    t_period: int = T_PERIOD,
    tau_event: float = TAU_EVENT,
    seed: int = 0,
    quality_noise: float = 0.10,
    filter_mode: str = "good",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stream, dt = load_aerpaw_stream(csv_path)
    tr_pnom = radar_nominal_trace(stream, dt, floor_sigma_m, filter_mode)
    flight = csv_path.name.split("_v2_")[0]

    meas = stream[["meas_x", "meas_y", "meas_z"]].to_numpy(float)
    gt_pos = stream[["gt_x", "gt_y", "gt_z"]].to_numpy(float)
    gt_vel = np.gradient(gt_pos, dt, axis=0)
    true_state = np.hstack([gt_pos, gt_vel])
    q_true = quality_from_sinr(stream["sinr_db"].fillna(stream["sinr_db"].median()).to_numpy(float))
    rng = np.random.default_rng(seed + 3000)
    q = np.clip(q_true * (1.0 + rng.normal(0.0, quality_noise, len(q_true))), Q_MIN, 1.0)
    r_mats = [np.diag([row.r_xx, row.r_yy, row.r_zz]) for row in stream.itertuples()]

    K_lqr, Q_lqg, R_lqg = build_lqr(dt)
    twins = {m: DigitalTwin(dt) for m in METHODS}
    radar_mode = _radar_filter_mode(filter_mode)
    filts = {m: RadarAdaptiveKalmanFilter(dt=dt, mode=radar_mode, base_sigma_m=floor_sigma_m)
             for m in METHODS}
    aoi = {m: 0 for m in METHODS}
    hold = {m: 0 for m in METHODS}
    sync_count = {m: 0 for m in METHODS}
    fuse_count = {m: 0 for m in METHODS}
    lqg_cost = {m: 0.0 for m in METHODS}
    records = []

    for k in range(len(stream)):
        for m in METHODS:
            twins[m].predict()
            out = filts[m].step(meas[k], q[k], r_mats[k], true_state[k])
            xe = np.asarray(out["state"], dtype=float)
            Pe = filts[m].P.copy()

            Ck = float(q[k] * np.exp(-ALPHA_CONF * np.trace(Pe) / max(tr_pnom, 1e-9)))
            Dk = float(np.linalg.norm(xe[:3] - twins[m].x[:3]))
            Dr = Dk / (Ck + DELTA_CONF)

            if k == 0:
                action = "SYNC"
                twins[m].sync(xe, Pe)
                aoi[m] = 0
            elif m == "unconditional":
                action = "SYNC"
                twins[m].sync(xe, Pe)
                aoi[m] = 0
                sync_count[m] += 1
            elif m == "periodic":
                if k % t_period == 0:
                    action = "SYNC"
                    twins[m].sync(xe, Pe)
                    aoi[m] = 0
                    sync_count[m] += 1
                else:
                    action = "HOLD"
                    aoi[m] += 1
            elif m == "event":
                if Dk > tau_event:
                    action = "SYNC"
                    twins[m].sync(xe, Pe)
                    aoi[m] = 0
                    sync_count[m] += 1
                else:
                    action = "HOLD"
                    aoi[m] += 1
            else:
                # AERPAW radar errors are tens of metres, unlike the metre-scale
                # DeepSense replay.  Here Dr is used in the intuitive direction:
                # low risk -> HOLD, medium risk -> FUSE, high risk -> SYNC.
                if Dr <= tau_low:
                    action = "HOLD"
                elif Dr <= tau_high:
                    action = "FUSE"
                else:
                    action = "SYNC"
                if action == "HOLD" and hold[m] >= MAX_HOLD:
                    action = "FUSE"
                if action == "SYNC":
                    twins[m].sync(xe, Pe)
                    aoi[m] = 0
                    sync_count[m] += 1
                    hold[m] = 0
                elif action == "FUSE":
                    twins[m].fuse(xe, Pe, w=FUSE_W)
                    aoi[m] = 0
                    fuse_count[m] += 1
                    hold[m] = 0
                else:
                    aoi[m] += 1
                    hold[m] += 1

            twin_err = float(np.linalg.norm(true_state[k, :3] - twins[m].x[:3]))
            ex = true_state[k] - xe
            eu = (-K_lqr @ xe) - (-K_lqr @ true_state[k])
            lqg_cost[m] += float(ex @ Q_lqg @ ex + eu @ R_lqg @ eu)
            records.append({
                "flight": flight,
                "seed": seed,
                "filter": filter_mode,
                "k": k,
                "t": float(stream["t"].iloc[k]) if "t" in stream else k * dt,
                "method": m,
                "action": action,
                "twin_err": twin_err,
                "aoi": aoi[m],
                "Ck": Ck,
                "Dk": Dk,
                "Dr": Dr,
                "q": float(q[k]),
                "q_true": float(q_true[k]),
                "sinr_db": float(stream["sinr_db"].iloc[k]),
                "assoc_error_m": float(stream["assoc_error_m"].iloc[k]),
            })

    ts = pd.DataFrame(records)
    rows = []
    for m in METHODS:
        sub = ts[ts["method"] == m]
        rows.append({
            "flight": flight,
            "seed": seed,
            "filter": filter_mode,
            "method": m,
            "rmse_twin": float(np.sqrt(np.mean(sub["twin_err"] ** 2))),
            "mean_aoi": float(sub["aoi"].mean()),
            "max_aoi": float(sub["aoi"].max()),
            "lqg_cost": float(lqg_cost[m]),
            "sync_count": float(sync_count[m]),
            "fuse_count": float(fuse_count[m]),
            "median_assoc_error_m": float(stream["assoc_error_m"].median()),
            "median_sinr_db": float(stream["sinr_db"].median()),
            "samples": int(len(stream)),
        })
    return pd.DataFrame(rows), ts


def plot_aerpaw_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    groups = [(flight, filt) for filt in sorted(summary["filter"].unique())
              for flight in sorted(summary["flight"].unique())]
    fig, axes = plt.subplots(2, len(groups), figsize=(4.3 * len(groups), 7.5))
    if len(groups) == 1:
        axes = axes.reshape(2, 1)
    for col, (flight, filt) in enumerate(groups):
        sub = summary[(summary["flight"] == flight) & (summary["filter"] == filt)]
        for row, metric in enumerate(["mean_aoi", "rmse_twin"]):
            ax = axes[row, col]
            vals = [float(sub[sub["method"] == m][metric].iloc[0]) for m in METHODS]
            ax.bar(np.arange(len(METHODS)), vals, color=[COLORS[m] for m in METHODS],
                   edgecolor="white", alpha=0.9)
            ax.set_xticks(np.arange(len(METHODS)))
            ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=15, ha="right", fontsize=7)
            ax.set_ylabel("Mean AoI (steps)" if metric == "mean_aoi" else "Twin RMSE (m)")
            if row == 0:
                ax.set_title(f"{flight} / {filt}", fontweight="bold", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "fig_aerpaw_dt_sync_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 3 AERPAW DT-sync adapter")
    parser.add_argument("--inputs", nargs="+", default=[
        "outputs_final_aerpaw_v2_opt1_20260610/Opt1_v2_associated_radar_measurements.csv",
        "outputs_final_aerpaw_v2_opt2_20260610/Opt2_v2_associated_radar_measurements.csv",
        "outputs_final_aerpaw_v2_opt3_nooffset_20260610/Opt3_v2_associated_radar_measurements.csv",
    ])
    parser.add_argument("--out", default="dt_sync_out_aerpaw")
    parser.add_argument("--floor-sigma-m", type=float, default=DEFAULT_SIGMA_M)
    parser.add_argument("--tau-low", type=float, default=20.0)
    parser.add_argument("--tau-high", type=float, default=100.0)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Monte Carlo seeds for q_k estimation noise")
    parser.add_argument("--quality-noise", type=float, default=0.10,
                        help="Relative sensing-quality estimation noise")
    parser.add_argument("--filter", default="good", choices=["good", "fixed", "both"],
                        help="Upstream radar filter mode")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = []
    all_ts = []
    filter_modes = ["good", "fixed"] if args.filter == "both" else [args.filter]
    for inp in args.inputs:
        path = Path(inp)
        for filter_mode in filter_modes:
            print(f"Running AERPAW DT sync: {path} ({filter_mode}, {args.seeds} seeds)")
            per_flight_summary = []
            per_flight_ts = []
            for seed in range(args.seeds):
                summary, ts = run_one_stream(
                    path,
                    floor_sigma_m=args.floor_sigma_m,
                    tau_low=args.tau_low,
                    tau_high=args.tau_high,
                    seed=seed,
                    quality_noise=args.quality_noise,
                    filter_mode=filter_mode,
                )
                per_flight_summary.append(summary)
                per_flight_ts.append(ts)
            summary = pd.concat(per_flight_summary, ignore_index=True)
            ts = pd.concat(per_flight_ts, ignore_index=True)
            flight = str(summary["flight"].iloc[0])
            tag = f"{flight}_{filter_mode}"
            summary.to_csv(out_dir / f"{tag}_dt_sync_summary.csv", index=False)
            ts.to_csv(out_dir / f"{tag}_dt_sync_timeseries.csv", index=False)
            all_summary.append(summary)
            all_ts.append(ts)
            mean_summary = summary.groupby("method", as_index=False).mean(numeric_only=True)
            prop = mean_summary[mean_summary["method"] == "proposed"].iloc[0]
            periodic = mean_summary[mean_summary["method"] == "periodic"].iloc[0]
            rmse_gain = (periodic.rmse_twin - prop.rmse_twin) / periodic.rmse_twin * 100
            sync_red = (periodic.sync_count - prop.sync_count) / periodic.sync_count * 100 if periodic.sync_count > 0 else 0.0
            print(f"  {flight}/{filter_mode}: RMSE {rmse_gain:.1f}% vs periodic | full-sync {sync_red:.1f}% vs periodic")

    combined = pd.concat(all_summary, ignore_index=True)
    combined_ts = pd.concat(all_ts, ignore_index=True)
    combined.to_csv(out_dir / "aerpaw_dt_sync_summary.csv", index=False)
    combined_ts.to_csv(out_dir / "aerpaw_dt_sync_timeseries.csv", index=False)
    plot_aerpaw_summary(
        combined.groupby(["flight", "filter", "method"], as_index=False).mean(numeric_only=True),
        out_dir,
    )
    print("\nCombined summary:")
    print(combined.groupby(["flight", "filter", "method"], as_index=False).mean(numeric_only=True).to_string(index=False))
    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()
