"""
isac_dt_sync_v3.py
==================
Integrity-Aware Digital Twin Synchronisation — Paper 3, real-data validation.

Extends isac_dt_sync_v2.py (synthetic Monte Carlo) with a DeepSense 6G replay
module that uses real measured 60 GHz beam-power as the sensing quality signal
q_k driving C_k in the synchronisation manager.

What is real vs simulated
--------------------------
  q_k        : REAL — derived from measured unit1_pwr_60ghz beam-power files
  trajectory : REAL — derived from unit2_loc GPS files (lat/lon → ENU metres)
  z_k (meas) : simulated from real trajectory with q_k-dependent noise
               (same as Paper 2 DeepSense replay — DeepSense has no raw radar pos)

Usage
-----
  python isac_dt_sync_v3.py \
      --data /path/to/scenario23_dev \
      --out  dt_sync_out_v3

  # or point at individual CSVs:
  python isac_dt_sync_v3.py \
      --seq  scenario23_seq20.csv scenario23_seq32.csv scenario23_seq33.csv \
      --data /path/to/scenario23_dev \
      --out  dt_sync_out_v3

Dependencies
------------
  isac_dt_sync_v2.py   (UpstreamFilter, DigitalTwin, build_lqr, one_trial,
                         ALPHA_CONF, DELTA_CONF, TAU_LOW, TAU_HIGH, FUSE_W,
                         MAX_HOLD, T_PERIOD, TAU_EVENT, Q_MIN, N_SEEDS)
  real_isac_replay.py  (load_deepsense_channel_quality,
                         load_deepsense_trajectory, _latlon_to_local_m)
  euroc_replay.py      (make_measurements, BASE_SIGMA_MEAS, MIN_QUALITY,
                         QUALITY_NOISE)
Both files must be in the same directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

# ── Import from v2 (sync logic, filter, twin) ─────────────────────────────
from isac_dt_sync_v2 import (
    UpstreamFilter, DigitalTwin, build_lqr,
    ALPHA_CONF, DELTA_CONF, TAU_LOW, TAU_HIGH, FUSE_W,
    MAX_HOLD, T_PERIOD, TAU_EVENT, Q_MIN, DT, N_SEEDS,
    tr_P_nominal_from_warmup,
    COLORS, LABELS,
)

# ── Import DeepSense loaders from real_isac_replay ─────────────────────────
from real_isac_replay import (
    load_deepsense_channel_quality,
    load_deepsense_trajectory,
)
from euroc_replay import (
    BASE_SIGMA_MEAS, MIN_QUALITY, QUALITY_NOISE,
    make_measurements,
)

METHODS  = ["proposed", "periodic", "event", "unconditional"]

# ── Load one DeepSense sequence ────────────────────────────────────────────
def load_deepsense_seq(
    csv_path: Path,
    dt: float = DT,
    seed: int = 0,
    q_ref_percentile: float = 90.0,
    quality_noise: float = QUALITY_NOISE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns
    -------
    pos      : (N,3) real ENU position from GPS
    vel      : (N,3) finite-difference velocity
    q_true   : (N,)  real q_k from measured beam power
    q_filter : (N,)  q_k + 10% estimation noise (used by filter)
    source   : human-readable description of the quality signal
    """
    cache_path = csv_path.with_name(f"{csv_path.stem}_v3_cache.csv")
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        pos = cache[["x", "y", "z"]].to_numpy(float)
        vel = np.gradient(pos, dt, axis=0)
        q_true = cache["q_true"].to_numpy(float)
        rng = np.random.default_rng(seed + 3000)
        q_filter = np.clip(
            q_true * (1.0 + rng.normal(0.0, quality_noise, len(q_true))),
            MIN_QUALITY,
            1.0,
        )
        return pos, vel, q_true, q_filter, f"cached DeepSense real beam-power quality ({cache_path.name})"

    # Real trajectory from GPS
    time_s, pos_3d = load_deepsense_trajectory(csv_path, dt)
    n = len(time_s)
    pos = pos_3d[:, :3]                               # (N,3) ENU
    vel = np.gradient(pos, dt, axis=0)                # finite-diff velocity

    # Real q_k from measured 60 GHz beam power
    cq = load_deepsense_channel_quality(
        csv_path,
        dt=dt,
        seed=seed,
        quality_noise=quality_noise,
        q_ref_percentile=q_ref_percentile,
    )
    # Align lengths (GPS and beam power may differ by ±1 row)
    n_common = min(len(pos), len(cq.quality_true))
    pos      = pos[:n_common]
    vel      = vel[:n_common]
    q_true   = cq.quality_true[:n_common]
    q_filter = cq.quality_filter[:n_common]

    return pos, vel, q_true, q_filter, cq.source


# ── One DeepSense trial (all four sync policies) ───────────────────────────
def one_trial_deepsense(
    csv_path: Path,
    filter_mode: str,
    tr_Pnom: float,
    K_lqr: np.ndarray,
    Ql: np.ndarray,
    Rl: np.ndarray,
    seed: int = 0,
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
    t_period: int = T_PERIOD,
    tau_event: float = TAU_EVENT,
    dt: float = DT,
    quality_noise: float = QUALITY_NOISE,
) -> dict:
    """
    Run all four sync policies on one DeepSense sequence.
    Returns dict of per-method metric dicts (same schema as one_trial in v2).
    """
    pos, vel, q_true, q_filter, _ = load_deepsense_seq(
        csv_path, dt=dt, seed=seed, quality_noise=quality_noise)
    n = len(pos)
    true_st = np.hstack([pos, vel])

    # Simulate noisy position measurements (same as Paper 2 DeepSense replay)
    rng_meas = np.random.default_rng(seed + 2000)
    meas, _outlier_mask = make_measurements(pos, q_true, rng_meas)   # (N,3)

    twins  = {m: DigitalTwin(dt) for m in METHODS}
    filts  = {m: UpstreamFilter(filter_mode, dt) for m in METHODS}
    aoi_s  = {m: 0  for m in METHODS}
    hold_s = {m: 0  for m in METHODS}
    sync_c = {m: 0  for m in METHODS}
    fuse_c = {m: 0  for m in METHODS}
    lqg_c  = {m: 0. for m in METHODS}
    records= {m: [] for m in METHODS}

    for k in range(n):
        q_k = float(q_filter[k])
        for m in METHODS:
            twins[m].predict()
            xe, Pe = filts[m].step(meas[k], q_k)

            # Integrity quantities
            Ck = q_k * np.exp(-ALPHA_CONF * np.trace(Pe) / tr_Pnom)
            Dk = float(np.linalg.norm(xe[:3] - twins[m].x[:3]))
            Dr = Dk / (Ck + DELTA_CONF)

            # ── Sync decision ─────────────────────────────────────
            if k == 0:
                twins[m].sync(xe, Pe); aoi_s[m] = 0; action = "SYNC"
            elif m == "unconditional":
                twins[m].sync(xe, Pe); aoi_s[m] = 0; sync_c[m] += 1
                action = "SYNC"
            elif m == "periodic":
                if k % t_period == 0:
                    twins[m].sync(xe, Pe); aoi_s[m] = 0; sync_c[m] += 1
                    action = "SYNC"
                else:
                    aoi_s[m] += 1; action = "HOLD"
            elif m == "event":
                if Dk > tau_event:
                    twins[m].sync(xe, Pe); aoi_s[m] = 0; sync_c[m] += 1
                    action = "SYNC"
                else:
                    aoi_s[m] += 1; action = "HOLD"
            else:  # proposed: low risk -> HOLD, medium -> FUSE, high -> SYNC
                if Dr <= tau_low:
                    action = "HOLD"
                elif Dr <= tau_high:
                    action = "FUSE"
                else:
                    action = "SYNC"
                if action == "HOLD" and hold_s[m] >= MAX_HOLD:
                    action = "FUSE"
                if action == "SYNC":
                    twins[m].sync(xe, Pe); aoi_s[m] = 0
                    sync_c[m] += 1; hold_s[m] = 0
                elif action == "FUSE":
                    twins[m].fuse(xe, Pe); aoi_s[m] = 0
                    fuse_c[m] += 1; hold_s[m] = 0
                else:
                    aoi_s[m] += 1; hold_s[m] += 1

            # ── Metrics ────────────────────────────────────────────
            terr = float(np.linalg.norm(true_st[k, :3] - twins[m].x[:3]))
            ex   = true_st[k] - xe
            eu   = (-K_lqr @ xe) - (-K_lqr @ true_st[k])
            lqg_c[m] += float(ex @ Ql @ ex + eu @ Rl @ eu)
            records[m].append({
                "k": k, "t": k * dt, "action": action,
                "twin_err": terr, "aoi": aoi_s[m],
                "Ck": Ck, "Dk": Dk, "Dr": Dr, "q": q_k,
                "q_true": float(q_true[k]),
            })

    out = {}
    for m in METHODS:
        df = pd.DataFrame(records[m])
        out[m] = {
            "rmse_twin":  float(np.sqrt(np.mean(df.twin_err ** 2))),
            "mean_aoi":   float(df.aoi.mean()),
            "max_aoi":    float(df.aoi.max()),
            "lqg_cost":   lqg_c[m],
            "sync_count": sync_c[m],
            "fuse_count": fuse_c[m],
            "df":         df,
        }
    return out


# ── Run all three sequences, multiple seeds ────────────────────────────────
def run_deepsense(
    data_dir: Path,
    seq_files: list[str],
    filter_mode: str,
    tr_Pnom: float,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    K, Ql, Rl = build_lqr()
    rows = []
    for seq_file in seq_files:
        csv_path = data_dir / seq_file
        seq_name = csv_path.stem   # e.g. "scenario23_seq20"
        print(f"  Sequence: {seq_name}")
        for seed in range(n_seeds):
            res = one_trial_deepsense(
                csv_path, filter_mode, tr_Pnom, K, Ql, Rl, seed=seed)
            for m, v in res.items():
                rows.append({
                    "seq": seq_name, "seed": seed, "method": m,
                    "filter": filter_mode,
                    "rmse_twin":  v["rmse_twin"],
                    "mean_aoi":   v["mean_aoi"],
                    "max_aoi":    v["max_aoi"],
                    "lqg_cost":   v["lqg_cost"],
                    "sync_count": v["sync_count"],
                    "fuse_count": v["fuse_count"],
                })
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_deepsense_results(df: pd.DataFrame, out_dir: Path) -> None:
    seqs    = sorted(df["seq"].unique())
    n_seqs  = len(seqs)
    fig, axes = plt.subplots(2, n_seqs, figsize=(5 * n_seqs, 8))
    if n_seqs == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(
        "DeepSense 6G Real-Channel DT Sync — AoI and Twin RMSE",
        fontweight="bold", fontsize=11)

    for col, seq in enumerate(seqs):
        sub = df[df.seq == seq]
        for row, (metric, ylabel) in enumerate([
            ("mean_aoi",  "Mean AoI (steps) ↓ better"),
            ("rmse_twin", "Twin RMSE (m) ↓ better"),
        ]):
            ax = axes[row, col]
            means = [float(sub[sub.method == m][metric].mean()) for m in METHODS]
            stds  = [float(sub[sub.method == m][metric].std())  for m in METHODS]
            colors= [COLORS[m] for m in METHODS]
            xpos  = np.arange(len(METHODS))
            ax.bar(xpos, means, yerr=stds, color=colors, capsize=3,
                   width=0.55, alpha=0.88, edgecolor="white")
            ax.set_xticks(xpos)
            ax.set_xticklabels([LABELS[m] for m in METHODS],
                               fontsize=7, rotation=12, ha="right")
            ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(seq.replace("scenario23_", "").upper(),
                             fontsize=9, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    p = out_dir / "fig_deepsense_aoi_rmse.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p.name}")


def plot_deepsense_timeline(
    csv_path: Path, filter_mode: str, tr_Pnom: float, out_dir: Path
) -> None:
    K, Ql, Rl = build_lqr()
    res = one_trial_deepsense(csv_path, filter_mode, tr_Pnom, K, Ql, Rl, seed=0)
    seq_name = csv_path.stem

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"DeepSense Timeline — {seq_name} — {filter_mode} filter",
        fontweight="bold", fontsize=11)

    for m in METHODS:
        df = pd.DataFrame(res[m]["df"])
        axes[0].plot(df.t, df.twin_err.rolling(8, min_periods=1).mean(),
                     color=COLORS[m], label=LABELS[m], lw=1.6)
        axes[1].plot(df.t, df.aoi, color=COLORS[m], lw=1.3)

    prop = pd.DataFrame(res["proposed"]["df"])
    axes[2].plot(prop.t, prop.q_true, color="#888", lw=1.2, ls=":",
                 label="q_k true (real 60 GHz)")
    axes[2].plot(prop.t, prop.Ck,    color="#2a9d8f", lw=1.5,
                 label="C_k (confidence)")
    Dr_norm = prop.Dr / max(float(prop.Dr.max()), 0.01)
    axes[2].plot(prop.t, Dr_norm, color="#2f6fed", lw=1.2, ls="--",
                 label="D_risk (normalised)")
    sync_t = prop.loc[prop.action == "SYNC", "t"]
    fuse_t = prop.loc[prop.action == "FUSE", "t"]
    if len(sync_t):
        axes[2].vlines(sync_t, 0, 1, color="#2f6fed", alpha=0.10, lw=0.7)
    if len(fuse_t):
        axes[2].vlines(fuse_t, 0, 1, color="#2a9d8f", alpha=0.15, lw=0.7)
    axes[2].plot([], [], "|", color="#2f6fed", alpha=0.5, ms=6, label="SYNC")
    axes[2].plot([], [], "|", color="#2a9d8f", alpha=0.6, ms=6, label="FUSE")

    axes[0].set_ylabel("Twin divergence (m)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].set_ylabel("AoI (steps)", fontsize=9)
    axes[2].set_ylabel("Quality / Decision signal", fontsize=9)
    axes[2].set_xlabel("Time (s)", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7.5, ncol=3)
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)

    fig.tight_layout()
    p = out_dir / f"fig_timeline_{seq_name}_{filter_mode}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p.name}")


# ── Summary table ──────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    seqs = sorted(df["seq"].unique())
    for seq in seqs:
        sub = df[df.seq == seq]
        prop_aoi  = float(sub[sub.method == "proposed"]["mean_aoi"].mean())
        event_aoi = float(sub[sub.method == "event"]["mean_aoi"].mean())
        prop_rmse = float(sub[sub.method == "proposed"]["rmse_twin"].mean())
        peri_rmse = float(sub[sub.method == "periodic"]["rmse_twin"].mean())
        prop_sync = float(sub[sub.method == "proposed"]["sync_count"].mean())
        fuse_cnt  = float(sub[sub.method == "proposed"]["fuse_count"].mean())
        print(f"\n  {seq}:")
        print(f"  {'Method':22} {'AoI':>8} {'RMSE':>8} {'Syncs':>7} {'Fuses':>7}")
        print(f"  {'-'*58}")
        for m in METHODS:
            s = sub[sub.method == m]
            print(f"  {m:22} {s['mean_aoi'].mean():8.2f} "
                  f"{s['rmse_twin'].mean():8.4f} "
                  f"{s['sync_count'].mean():7.0f} "
                  f"{s['fuse_count'].mean():7.0f}")
        aoi_red = (event_aoi - prop_aoi) / event_aoi * 100 if event_aoi > 0 else 0
        rmse_imp= (peri_rmse - prop_rmse) / peri_rmse * 100
        print(f"\n  Key: AoI {aoi_red:.0f}% lower than event-triggered | "
              f"RMSE {rmse_imp:.1f}% vs periodic")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DT sync v3 — DeepSense real-channel validation")
    parser.add_argument("--data", default="scenario23_dev",
                        help="Directory containing scenario23_seq*.csv files")
    parser.add_argument("--seq", nargs="+",
                        default=["scenario23_seq20.csv",
                                 "scenario23_seq32.csv",
                                 "scenario23_seq33.csv"],
                        help="CSV filenames within --data")
    parser.add_argument("--filter", default="good",
                        choices=["good", "fixed", "both"],
                        help="Upstream filter mode")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Monte Carlo seeds (10 is fast; 30 for final)")
    parser.add_argument("--out", default="dt_sync_out_v3")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    filter_modes = (["good", "fixed"] if args.filter == "both"
                    else [args.filter])

    print("Step 1: Computing tr(P_nominal)...")
    tr_Pnom = {}
    for mode in filter_modes:
        tr_Pnom[mode] = tr_P_nominal_from_warmup(mode)
        print(f"  {mode:5} filter: tr(P_nominal) = {tr_Pnom[mode]:.5f}")

    all_dfs = []
    for mode in filter_modes:
        print(f"\nStep 2: DeepSense replay — {mode} filter "
              f"({args.seeds} seeds × {len(args.seq)} sequences)...")
        df = run_deepsense(data_dir, args.seq, mode, tr_Pnom[mode],
                           n_seeds=args.seeds)
        df.to_csv(out_dir / f"deepsense_{mode}_mc.csv", index=False)
        print_summary(df, f"DeepSense — {mode} upstream filter")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "deepsense_combined.csv", index=False)

    print("\nStep 3: Plotting...")
    # Bar chart for each filter mode
    for mode in filter_modes:
        sub = combined[combined["filter"] == mode]
        plot_deepsense_results(sub, out_dir)

    # Timeline for first sequence, first filter mode
    first_csv = data_dir / args.seq[0]
    plot_deepsense_timeline(first_csv, filter_modes[0], tr_Pnom[filter_modes[0]], out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
