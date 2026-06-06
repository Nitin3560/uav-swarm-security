"""Replay the ISAC filter with measured wireless channel quality.

This script is the real-channel counterpart to ``euroc_replay.py``.  The
trajectory may come either from a TUM-format ground-truth file or from a channel
CSV that contains position columns.  The sensing-quality variable ``q_k`` is
derived from measured wireless data instead of the synthetic EuRoC speed proxy:

- explicit ``q`` / ``quality`` column, or
- ``sinr_db`` / ``snr_db`` column, or
- measured beam/power vector columns such as ``beam_0`` ... ``beam_63`` or
  ``power_0`` ... ``power_63``.

For DeepSense 6G Scenario 23, use the scenario CSV containing GPS/position and
the 64-element mmWave received-power vector.  The script uses the strongest beam
power as the real channel-quality proxy, converts it to SNR if a noise floor is
provided, and maps it to ``q_k`` for ``R_k = R0 / q_k``.

Important: public communication datasets usually provide real channel quality
and real trajectory, but not raw radar-derived position measurements.  This
script therefore still simulates noisy position measurements from the trajectory;
the authenticity improvement is that covariance adaptation is driven by real
wireless channel measurements.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from euroc_replay import (
    BASE_SIGMA_MEAS,
    DT_DELAY_STEPS,
    MIN_QUALITY,
    QUALITY_NOISE,
    ISACKalmanFilter,
    build_lqr,
    load_euroc_gt,
    make_measurements,
    resample_to_dt,
    summarise,
)


@dataclass(frozen=True)
class ChannelQuality:
    time_s: np.ndarray
    quality_true: np.ndarray
    quality_filter: np.ndarray
    raw_metric: np.ndarray
    source: str


def _normalise_columns(df: pd.DataFrame) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "", col.lower()): col for col in df.columns}


def _find_first(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in columns:
            return columns[key]
    return None


def _find_power_columns(df: pd.DataFrame) -> list[str]:
    """Find likely mmWave beam/power-vector columns."""
    power_cols = []
    patterns = [
        re.compile(r"^(beam|power|pwr|rxpower|rsrp|mmwave|unit1mmwave|unit1pwr)\D*\d+$", re.I),
        re.compile(r"^(unit1_)?(beam|power|pwr|rx_power|mmwave).*\d+$", re.I),
    ]
    for col in df.columns:
        col_clean = col.strip()
        if any(pattern.search(col_clean) for pattern in patterns):
            if pd.api.types.is_numeric_dtype(df[col]):
                power_cols.append(col)

    if power_cols:
        return power_cols

    # Some datasets store the 64-vector in consecutive numeric-looking columns.
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    excluded = {"time", "timestamp", "x", "y", "z", "lat", "latitude", "lon", "longitude", "height", "altitude"}
    return [
        col
        for col in numeric_cols
        if re.sub(r"[^a-z0-9]+", "", col.lower()) not in excluded
    ][-64:]


def _quality_from_metric(metric: np.ndarray, mode: str, noise_floor_dbm: float, q_ref_percentile: float) -> np.ndarray:
    metric = np.asarray(metric, dtype=float)
    finite = np.isfinite(metric)
    if not finite.any():
        raise ValueError("No finite channel-quality values found.")

    safe_metric = metric.copy()
    safe_metric[~finite] = np.nanmedian(metric[finite])

    if mode == "quality":
        q = safe_metric
        if np.nanmax(q) > 1.5 or np.nanmin(q) < 0.0:
            q = q / (np.nanpercentile(q, q_ref_percentile) + 1e-9)
    elif mode == "snr_db":
        snr_linear = 10.0 ** (safe_metric / 10.0)
        q = snr_linear / (np.nanpercentile(snr_linear, q_ref_percentile) + 1e-9)
    elif mode == "power_dbm":
        snr_db = safe_metric - float(noise_floor_dbm)
        snr_linear = 10.0 ** (snr_db / 10.0)
        q = snr_linear / (np.nanpercentile(snr_linear, q_ref_percentile) + 1e-9)
    elif mode == "relative_power_db":
        rel_linear = 10.0 ** ((safe_metric - np.nanpercentile(safe_metric, 10.0)) / 10.0)
        q = rel_linear / (np.nanpercentile(rel_linear, q_ref_percentile) + 1e-9)
    elif mode == "linear_power":
        q = safe_metric / (np.nanpercentile(safe_metric, q_ref_percentile) + 1e-12)
    else:
        raise ValueError(f"Unsupported quality mode: {mode}")

    return np.clip(q, MIN_QUALITY, 1.0)


def _is_deepsense_path_csv(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    return {"unit1_pwr_60ghz", "unit2_loc"}.issubset(columns)


def _resolve_dataset_path(root_csv: Path, value: str) -> Path:
    value = str(value).strip().strip("\"'")
    return (root_csv.parent / value).resolve()


def _load_numeric_file(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not read numeric DeepSense file: {path}") from exc


def _parse_deepsense_timestamp(value: str, fallback_index: int, fallback_dt: float) -> float:
    """Parse strings like "['16-58-42-142']" into seconds of day."""
    match = re.search(r"(\d+)-(\d+)-(\d+)-(\d+)", str(value))
    if not match:
        return float(fallback_index) * fallback_dt
    hour, minute, second, millis = [int(part) for part in match.groups()]
    return float(hour * 3600 + minute * 60 + second + millis / 1000.0)


def load_deepsense_channel_quality(
    channel_csv: Path,
    dt: float,
    seed: int,
    quality_noise: float,
    q_ref_percentile: float,
) -> ChannelQuality:
    df = pd.read_csv(channel_csv)
    if not _is_deepsense_path_csv(df):
        raise ValueError(f"{channel_csv} does not look like a DeepSense path-index CSV.")

    # Scenario 23 rows are ordered samples, but UTC timestamps can include gaps
    # between collection segments.  For filter replay we use sample time so the
    # dynamics are not stretched across pauses in data collection.
    time_s = np.arange(len(df), dtype=float) * float(dt)

    best_power = np.zeros(len(df), dtype=float)
    for idx, rel_path in enumerate(df["unit1_pwr_60ghz"]):
        power_vec = np.atleast_1d(_load_numeric_file(_resolve_dataset_path(channel_csv, rel_path)))
        best_power[idx] = float(np.nanmax(power_vec))

    q_true = _quality_from_metric(best_power, "linear_power", noise_floor_dbm=0.0, q_ref_percentile=q_ref_percentile)
    rng = np.random.default_rng(seed + 3000)
    q_filter = np.clip(
        q_true * (1.0 + rng.normal(0.0, quality_noise, len(q_true))),
        MIN_QUALITY,
        1.0,
    )
    return ChannelQuality(
        time_s=time_s,
        quality_true=q_true,
        quality_filter=q_filter,
        raw_metric=best_power,
        source="DeepSense Scenario 23: strongest measured 60 GHz beam power",
    )


def load_deepsense_trajectory(channel_csv: Path, dt: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(channel_csv)
    time_s = np.arange(len(df), dtype=float) * float(dt)

    lat = np.zeros(len(df), dtype=float)
    lon = np.zeros(len(df), dtype=float)
    z = np.zeros(len(df), dtype=float)
    has_height = "unit2_height" in df.columns
    for idx, rel_path in enumerate(df["unit2_loc"]):
        loc = np.atleast_1d(_load_numeric_file(_resolve_dataset_path(channel_csv, rel_path)))
        if len(loc) < 2:
            raise ValueError(f"Expected lat/lon in {rel_path}, got {loc}")
        lat[idx] = float(loc[0])
        lon[idx] = float(loc[1])
        if has_height:
            z[idx] = float(np.atleast_1d(_load_numeric_file(_resolve_dataset_path(channel_csv, df.loc[idx, "unit2_height"])))[0])
    return time_s, _latlon_to_local_m(lat, lon, z)


def load_channel_quality(
    channel_csv: Path,
    dt: float,
    seed: int,
    quality_noise: float,
    noise_floor_dbm: float,
    q_ref_percentile: float,
) -> ChannelQuality:
    df = pd.read_csv(channel_csv)
    if df.empty:
        raise ValueError(f"{channel_csv} is empty.")
    if _is_deepsense_path_csv(df):
        return load_deepsense_channel_quality(
            channel_csv=channel_csv,
            dt=dt,
            seed=seed,
            quality_noise=quality_noise,
            q_ref_percentile=q_ref_percentile,
        )

    columns = _normalise_columns(df)
    time_col = _find_first(columns, ["time", "time_s", "timestamp", "timestamp_s", "t"])
    if time_col is not None:
        time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        time_s = time_s - np.nanmin(time_s)
    else:
        time_s = np.arange(len(df), dtype=float) * dt

    q_col = _find_first(columns, ["q", "quality", "quality_true", "sensing_quality"])
    sinr_col = _find_first(columns, ["sinr_db", "sinr", "snr_db", "snr"])
    power_cols = _find_power_columns(df)

    if q_col is not None:
        raw_metric = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
        q_true = _quality_from_metric(raw_metric, "quality", noise_floor_dbm, q_ref_percentile)
        source = f"explicit quality column: {q_col}"
    elif sinr_col is not None:
        raw_metric = pd.to_numeric(df[sinr_col], errors="coerce").to_numpy(dtype=float)
        q_true = _quality_from_metric(raw_metric, "snr_db", noise_floor_dbm, q_ref_percentile)
        source = f"SINR/SNR column: {sinr_col}"
    elif power_cols:
        power_matrix = df[power_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        raw_metric = np.nanmax(power_matrix, axis=1)
        if np.nanmedian(raw_metric) < 0.0:
            mode = "power_dbm"
            source = f"best measured beam power from {len(power_cols)} columns, noise floor {noise_floor_dbm:g} dBm"
        else:
            mode = "relative_power_db"
            source = f"best measured beam/power metric from {len(power_cols)} columns, relative normalization"
        q_true = _quality_from_metric(raw_metric, mode, noise_floor_dbm, q_ref_percentile)
    else:
        raise ValueError(
            "Could not find a q/quality column, SINR/SNR column, or numeric power-vector columns."
        )

    rng = np.random.default_rng(seed + 3000)
    q_filter = np.clip(
        q_true * (1.0 + rng.normal(0.0, quality_noise, len(q_true))),
        MIN_QUALITY,
        1.0,
    )
    return ChannelQuality(time_s=time_s, quality_true=q_true, quality_filter=q_filter, raw_metric=raw_metric, source=source)


def _latlon_to_local_m(lat: np.ndarray, lon: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.deg2rad(lat[0])
    radius = 6_378_137.0
    x = np.deg2rad(lon - lon[0]) * radius * np.cos(lat0)
    y = np.deg2rad(lat - lat[0]) * radius
    if z is None:
        z = np.zeros_like(x)
    return np.column_stack([x, y, np.asarray(z, dtype=float) - float(np.asarray(z)[0])])


def load_channel_trajectory(channel_csv: Path, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Load trajectory from channel CSV, returning relative time and local XYZ."""
    df = pd.read_csv(channel_csv)
    if _is_deepsense_path_csv(df):
        return load_deepsense_trajectory(channel_csv, dt)
    columns = _normalise_columns(df)
    time_col = _find_first(columns, ["time", "time_s", "timestamp", "timestamp_s", "t"])
    if time_col is not None:
        time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        time_s = time_s - np.nanmin(time_s)
    else:
        time_s = np.arange(len(df), dtype=float) * dt

    x_col = _find_first(columns, ["x", "tx", "pos_x", "position_x", "gps_x", "east"])
    y_col = _find_first(columns, ["y", "ty", "pos_y", "position_y", "gps_y", "north"])
    z_col = _find_first(columns, ["z", "tz", "pos_z", "position_z", "height", "altitude", "alt"])
    if x_col and y_col:
        z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float) if z_col else np.zeros(len(df))
        pos = np.column_stack(
            [
                pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float),
                z,
            ]
        )
        pos = pos - pos[0]
        return time_s, pos

    lat_col = _find_first(columns, ["lat", "latitude", "gps_lat", "gps1_lat", "unit2_gps_lat"])
    lon_col = _find_first(columns, ["lon", "lng", "longitude", "gps_lon", "gps1_lon", "unit2_gps_lon"])
    if lat_col and lon_col:
        z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float) if z_col else None
        pos = _latlon_to_local_m(
            pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float),
            z,
        )
        return time_s, pos

    raise ValueError("No usable position columns found in channel CSV.")


def load_tum_trajectory(gt_path: Path, dt: float) -> tuple[np.ndarray, np.ndarray]:
    gt = resample_to_dt(load_euroc_gt(gt_path), dt)
    return gt[:, 0] - gt[0, 0], gt[:, 1:4]


def interpolate_to_time(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(target_time, source_time, values)
    out = np.zeros((len(target_time), values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(target_time, source_time, values[:, col])
    return out


def replay_real_channel(
    channel_csv: Path,
    gt_path: Path | None = None,
    dt: float = 0.05,
    seed: int = 0,
    quality_noise: float = QUALITY_NOISE,
    noise_floor_dbm: float = -94.0,
    q_ref_percentile: float = 90.0,
) -> tuple[pd.DataFrame, str]:
    channel = load_channel_quality(
        channel_csv=channel_csv,
        dt=dt,
        seed=seed,
        quality_noise=quality_noise,
        noise_floor_dbm=noise_floor_dbm,
        q_ref_percentile=q_ref_percentile,
    )

    if gt_path is not None:
        time_s, pos = load_tum_trajectory(gt_path, dt)
    else:
        time_s_raw, pos_raw = load_channel_trajectory(channel_csv, dt)
        time_s = np.arange(0.0, time_s_raw[-1] + 1e-9, dt)
        pos = interpolate_to_time(pos_raw, time_s_raw, time_s)

    q_true = interpolate_to_time(channel.quality_true, channel.time_s, time_s)
    q_filter = interpolate_to_time(channel.quality_filter, channel.time_s, time_s)
    raw_metric = interpolate_to_time(channel.raw_metric, channel.time_s, time_s)

    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    measurements, outlier_mask = make_measurements(pos, q_true, np.random.default_rng(seed + 2000))
    true_state = np.hstack([pos, vel])

    modes = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    labels = {
        "fixed_kf": "Fixed-R KF",
        "adaptive_r": "Real-channel adaptive R",
        "adaptive_r_gate": "Adaptive R + NIS gate",
    }
    filters = {mode: ISACKalmanFilter(dt=dt, mode=mode) for mode in modes}
    k_lqr, q_lqg, r_lqg = build_lqr(dt)
    lqg_cost = {mode: 0.0 for mode in modes}
    dt_buffer = {mode: [] for mode in modes}

    rows = []
    for k in range(len(time_s)):
        for mode, filt in filters.items():
            out = filt.step(measurements[k], q_filter[k], true_state=true_state[k])
            u_hat = -k_lqr @ out.state
            u_true = -k_lqr @ true_state[k]
            err_state = true_state[k] - out.state
            err_u = u_hat - u_true
            lqg_cost[mode] += float(err_state @ q_lqg @ err_state + err_u @ r_lqg @ err_u)

            dt_buffer[mode].append(out.state.copy())
            dt_state = dt_buffer[mode].pop(0) if len(dt_buffer[mode]) > DT_DELAY_STEPS else out.state
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
                    "dt_err_m": float(np.linalg.norm(true_state[k, :3] - dt_state[:3])),
                    "is_outlier": float(outlier_mask[k]),
                    "quality": q_filter[k],
                    "true_quality": q_true[k],
                    "raw_channel_metric": raw_metric[k],
                    "speed_ms": speed[k],
                }
            )

    df = pd.DataFrame(rows)
    for mode in modes:
        df.loc[df["mode"] == mode, "lqg_cost_total"] = lqg_cost[mode]
    return df, channel.source


def plot_real_channel_results(df: pd.DataFrame, name: str, out_dir: Path, source: str) -> None:
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
    fig.suptitle("Real-Channel ISAC Replay", fontweight="bold")

    ax = axes[0, 0]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["est_err_m"].rolling(15, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["nees"].rolling(25, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.axhline(6.0, color="k", ls="--", lw=1)
    ax.set_ylim(0, min(ax.get_ylim()[1], 150))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")

    ax = axes[1, 0]
    base = df[df["mode"] == "fixed_kf"]
    ax2 = ax.twinx()
    ax.plot(base["t"], base["true_quality"], color="#c43c39", lw=1.5, label="q_k from measured channel")
    ax2.plot(base["t"], base["raw_channel_metric"], color="#555", lw=1.0, alpha=0.6, ls="--", label="raw channel metric")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Quality q_k")
    ax2.set_ylabel("Raw channel metric")
    ax.set_title(source, fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)

    ax = axes[1, 1]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["dt_err_m"].rolling(15, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0], lw=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DT delay error (m)")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}_real_channel_replay.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}_real_channel_replay.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ISAC filter using measured wireless channel quality")
    parser.add_argument("--channel-csv", required=True, help="CSV with SINR/SNR, q, or mmWave power-vector columns")
    parser.add_argument("--gt", default=None, help="Optional TUM ground-truth trajectory file. If omitted, trajectory is read from channel CSV.")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-floor-dbm", type=float, default=-94.0)
    parser.add_argument("--q-ref-percentile", type=float, default=90.0)
    parser.add_argument("--quality-noise", type=float, default=QUALITY_NOISE)
    parser.add_argument("--out", default="outputs_real_isac_replay")
    args = parser.parse_args()

    channel_csv = Path(args.channel_csv)
    gt_path = Path(args.gt) if args.gt else None
    out_dir = Path(args.out)
    name = channel_csv.stem

    df, source = replay_real_channel(
        channel_csv=channel_csv,
        gt_path=gt_path,
        dt=args.dt,
        seed=args.seed,
        quality_noise=args.quality_noise,
        noise_floor_dbm=args.noise_floor_dbm,
        q_ref_percentile=args.q_ref_percentile,
    )
    summary = summarise(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_real_channel_timeseries.csv", index=False)
    summary.to_csv(out_dir / f"{name}_real_channel_summary.csv", index=False)
    plot_real_channel_results(df, name, out_dir, source)

    print(f"Channel source: {source}")
    print(f"\n{'Label':28s} {'RMSE':>8} {'NEES':>8} {'LQG cost':>12} {'DT err':>8}")
    print("-" * 72)
    fixed_rmse = float(summary.loc[summary["method"] == "fixed_kf", "rmse_m"].iloc[0])
    fixed_lqg = float(summary.loc[summary["method"] == "fixed_kf", "lqg_cost"].iloc[0])
    for _, row in summary.iterrows():
        if row.method == "fixed_kf":
            suffix = "  [baseline]"
        else:
            rmse_imp = (fixed_rmse - row.rmse_m) / fixed_rmse * 100.0
            cost_imp = (fixed_lqg - row.lqg_cost) / fixed_lqg * 100.0
            suffix = f"  RMSE↓{rmse_imp:.1f}%  Cost↓{cost_imp:.1f}%"
        print(
            f"{row.label:28s} {row.rmse_m:8.4f} {row.mean_nees:8.2f} "
            f"{row.lqg_cost:12.1f} {row.mean_dt_err_m:8.4f}{suffix}"
        )
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
