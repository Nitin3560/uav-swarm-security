"""Replay the ISAC filter on AERPAW Fortem radar-derived UAV measurements.

This adapter uses the Dryad/AERPAW AADM 2025 dataset:

    UAV-based wireless multi-modal measurements from AERPAW autonomous data mule
    (AADM) challenge in digital twin and real-world environments

Unlike ``real_isac_replay.py`` with DeepSense, this script does not simulate the
position measurement.  It parses Fortem R20 ``radar_data_*.json`` detections and
uses the radar-derived LLA target position as ``z_k``.  UAV ``vehicleOut.txt``
telemetry provides ground truth.  The radar detection nearest to the UAV ground
truth at each radar frame is selected as the target measurement; the selected
detection's ``sinrDb`` drives the quality variable for ``R_k = R0 / q_k``.

This is the closest public-data validation path for the "raw/radar-derived
measurement" limitation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from euroc_replay import DT_DELAY_STEPS, ISACKalmanFilter, build_lqr, summarise


RADAR_BASE_SIGMA_M = 8.0
MIN_QUALITY = 0.04


@dataclass(frozen=True)
class FlightPair:
    flight_id: str
    folder: str
    radar_json: str
    vehicle_out: str


def lla_to_local_m(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, origin: np.ndarray) -> np.ndarray:
    radius = 6_378_137.0
    lat0 = np.deg2rad(origin[0])
    x = np.deg2rad(lon - origin[1]) * radius * np.cos(lat0)
    y = np.deg2rad(lat - origin[0]) * radius
    z = alt - origin[2]
    return np.column_stack([x, y, z])


def parse_vehicle_timestamp(value: str) -> float:
    # vehicleOut logs are local EDT-like timestamps; Fortem globalTime is UTC.
    dt = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S.%f")
    return (dt + timedelta(hours=4)).replace(tzinfo=timezone.utc).timestamp()


def load_vehicle_out(zf: ZipFile, name: str) -> pd.DataFrame:
    text = zf.read(name).decode("utf-8", errors="ignore").splitlines()
    rows: list[dict[str, float]] = []
    for row in csv.reader(text):
        if len(row) < 8:
            continue
        try:
            idx = int(row[0])
            lon = float(row[1])
            lat = float(row[2])
            rel_alt = float(row[3])
            baro_alt = float(row[6])
            timestamp = parse_vehicle_timestamp(row[7])
        except Exception:
            continue
        rows.append(
            {
                "idx": idx,
                "time": timestamp,
                "lat": lat,
                "lon": lon,
                "alt": baro_alt + rel_alt,
            }
        )
    if len(rows) < 5:
        raise ValueError(f"Could not parse enough vehicleOut rows from {name}")
    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
    origin = df[["lat", "lon", "alt"]].iloc[0].to_numpy(dtype=float)
    local = lla_to_local_m(
        df["lat"].to_numpy(dtype=float),
        df["lon"].to_numpy(dtype=float),
        df["alt"].to_numpy(dtype=float),
        origin,
    )
    df[["x", "y", "z"]] = local
    return df


def iter_radar_frames(zf: ZipFile, name: str):
    for raw_line in zf.read(name).decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        params = obj.get("params", {})
        detections = obj.get("data", [])
        if not params or not detections:
            continue
        frame_time = float(params.get("globalTime", params.get("gpsSeconds", np.nan)))
        if not np.isfinite(frame_time):
            continue
        yield frame_time, detections


def interpolate_gt(vehicle: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    gt = np.zeros((len(times), 3), dtype=float)
    for col_idx, col in enumerate(["x", "y", "z"]):
        gt[:, col_idx] = np.interp(times, vehicle["time"], vehicle[col])
    return gt


def quality_from_sinr(sinr_db: np.ndarray, ref_percentile: float = 90.0) -> np.ndarray:
    sinr_linear = 10.0 ** (np.asarray(sinr_db, dtype=float) / 10.0)
    q = sinr_linear / (np.nanpercentile(sinr_linear, ref_percentile) + 1e-12)
    return np.clip(q, MIN_QUALITY, 1.0)


def associate_radar_to_gt(
    zf: ZipFile,
    radar_json: str,
    vehicle: pd.DataFrame,
    max_assoc_m: float,
) -> pd.DataFrame:
    origin = vehicle[["lat", "lon", "alt"]].iloc[0].to_numpy(dtype=float)
    out = []
    for frame_time, detections in iter_radar_frames(zf, radar_json):
        gt_pos = interpolate_gt(vehicle, np.array([frame_time]))[0]
        best = None
        best_dist = float("inf")
        for det in detections:
            lla = det.get("lla")
            if not isinstance(lla, list) or len(lla) < 3:
                continue
            try:
                lat, lon, alt = float(lla[0]), float(lla[1]), float(lla[2])
                if abs(lat) < 1e-9 or abs(lon) < 1e-9:
                    continue
                pos = lla_to_local_m(
                    np.array([lat]), np.array([lon]), np.array([alt]), origin
                )[0]
                dist = float(np.linalg.norm(pos - gt_pos))
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best = (det, pos)
        if best is None or best_dist > max_assoc_m:
            continue
        det, pos = best
        out.append(
            {
                "time": frame_time,
                "meas_x": pos[0],
                "meas_y": pos[1],
                "meas_z": pos[2],
                "gt_x": gt_pos[0],
                "gt_y": gt_pos[1],
                "gt_z": gt_pos[2],
                "assoc_error_m": best_dist,
                "sinr_db": float(det.get("sinrDb", np.nan)),
                "range_sigma": float(det.get("rangeSigma", np.nan)),
                "azimuth_sigma": float(det.get("azimuthSigma", np.nan)),
                "elevation_sigma": float(det.get("elevationSigma", np.nan)),
                "rcs_dbsm": float(det.get("rcsDbsm", np.nan)),
                "range_m": float(det.get("range", np.nan)),
            }
        )
    if not out:
        raise ValueError(f"No radar detections associated for {radar_json}")
    df = pd.DataFrame(out).sort_values("time").drop_duplicates("time")
    df["t"] = df["time"] - df["time"].iloc[0]
    return df


def list_flight_pairs(zip_path: Path) -> list[FlightPair]:
    with ZipFile(zip_path) as zf:
        names = zf.namelist()
    radars = [n for n in names if "RF Sensor and Radar/" in n and n.endswith(".json")]
    vehicles = [
        n
        for n in names
        if "RF Sensor and Radar/" in n and "vehicleOut" in Path(n).name and n.endswith(".txt")
    ]

    def folder(name: str) -> str:
        return name.rsplit("/", 1)[0].strip()

    pairs = []
    for radar in radars:
        rf = folder(radar)
        match = [v for v in vehicles if folder(v) == rf]
        if not match:
            continue
        if "rerun" in Path(radar).name.lower():
            rerun_match = [v for v in match if "rerun" in Path(v).name.lower()]
            if rerun_match:
                match = rerun_match
        flight_match = re.search(r"(AADM\d+|Opt\d+|flight\d+)", Path(radar).name, flags=re.I)
        flight_id = flight_match.group(1) if flight_match else Path(rf).name.strip()
        pairs.append(FlightPair(flight_id=flight_id, folder=rf, radar_json=radar, vehicle_out=match[0]))
    return pairs


def run_filter_on_radar(radar_df: pd.DataFrame, dt: float, radar_sigma_m: float) -> pd.DataFrame:
    times = radar_df["time"].to_numpy(dtype=float)
    measurements = radar_df[["meas_x", "meas_y", "meas_z"]].to_numpy(dtype=float)
    gt_pos = radar_df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
    gt_vel = np.gradient(gt_pos, dt, axis=0)
    true_state = np.hstack([gt_pos, gt_vel])
    q = quality_from_sinr(radar_df["sinr_db"].fillna(radar_df["sinr_db"].median()).to_numpy(dtype=float))

    modes = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    labels = {
        "fixed_kf": "Fixed-R KF",
        "adaptive_r": "Radar-SINR adaptive R",
        "adaptive_r_gate": "Adaptive R + NIS gate",
    }
    filters = {mode: ISACKalmanFilter(dt=dt, mode=mode) for mode in modes}
    for filt in filters.values():
        filt.R0 = np.eye(3) * float(radar_sigma_m) ** 2
    k_lqr, q_lqg, r_lqg = build_lqr(dt)
    lqg_cost = {mode: 0.0 for mode in modes}
    dt_buffer = {mode: [] for mode in modes}
    rows = []
    for k in range(len(radar_df)):
        is_outlier = float(radar_df["assoc_error_m"].iloc[k] > 3.0 * radar_sigma_m)
        for mode, filt in filters.items():
            out = filt.step(measurements[k], q[k], true_state=true_state[k])
            u_hat = -k_lqr @ out.state
            u_true = -k_lqr @ true_state[k]
            err_state = true_state[k] - out.state
            err_u = u_hat - u_true
            lqg_cost[mode] += float(err_state @ q_lqg @ err_state + err_u @ r_lqg @ err_u)
            dt_buffer[mode].append(out.state.copy())
            dt_state = dt_buffer[mode].pop(0) if len(dt_buffer[mode]) > DT_DELAY_STEPS else out.state
            rows.append(
                {
                    "t": radar_df["t"].iloc[k],
                    "time": times[k],
                    "mode": mode,
                    "label": labels[mode],
                    "est_err_m": float(np.linalg.norm(out.state[:3] - true_state[k, :3])),
                    "measurement_error_m": float(radar_df["assoc_error_m"].iloc[k]),
                    "is_outlier": is_outlier,
                    "nis": out.nis,
                    "nees": out.nees,
                    "pos_nees": out.pos_nees,
                    "soft_rew": float(out.soft_reweighted),
                    "hard_rej": float(out.hard_rejected),
                    "dt_err_m": float(np.linalg.norm(true_state[k, :3] - dt_state[:3])),
                    "quality": q[k],
                    "sinr_db": radar_df["sinr_db"].iloc[k],
                    "range_m": radar_df["range_m"].iloc[k],
                }
            )
    df = pd.DataFrame(rows)
    for mode in modes:
        df.loc[df["mode"] == mode, "lqg_cost_total"] = lqg_cost[mode]
    return df


def plot_results(df: pd.DataFrame, flight_id: str, out_dir: Path) -> None:
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
    fig.suptitle(f"AERPAW Fortem Radar Replay: {flight_id}", fontweight="bold")
    ax = axes[0, 0]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["est_err_m"].rolling(7, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["nees"].rolling(7, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0])
    ax.axhline(6.0, color="k", ls="--", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.set_ylim(0, min(300, ax.get_ylim()[1]))

    base = df[df["mode"] == "fixed_kf"]
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(base["t"], base["quality"], color="#c43c39", lw=1.5, label="q from radar SINR")
    ax2.plot(base["t"], base["measurement_error_m"], color="#555", lw=1.0, ls="--", alpha=0.7, label="radar meas. error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Quality q")
    ax2.set_ylabel("Radar measurement error (m)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)

    ax = axes[1, 1]
    for mode, sub in df.groupby("mode"):
        ax.plot(sub["t"], sub["dt_err_m"].rolling(7, center=True, min_periods=1).mean(), color=colors[mode], label=sub["label"].iloc[0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DT delay error (m)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{flight_id}_aerpaw_radar_replay.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{flight_id}_aerpaw_radar_replay.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ISAC filter using AERPAW Fortem radar measurements")
    parser.add_argument("--zip", default="data/aerpaw_aadm2025/AADM2025Dryad.zip")
    parser.add_argument("--flight", default="AADM8", help="Flight id substring, e.g. AADM8, AADM1, flight8")
    parser.add_argument("--list-flights", action="store_true")
    parser.add_argument("--max-assoc-m", type=float, default=250.0)
    parser.add_argument("--radar-sigma-m", type=float, default=8.0)
    parser.add_argument("--out", default="outputs_aerpaw_radar_replay")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    pairs = list_flight_pairs(zip_path)
    if args.list_flights:
        for pair in pairs:
            print(f"{pair.flight_id:12s} | {pair.folder}")
        return
    flight_query = args.flight.lower()
    candidates = [pair for pair in pairs if pair.flight_id.lower() == flight_query]
    if not candidates:
        candidates = [
            pair
            for pair in pairs
            if flight_query in pair.radar_json.lower() or flight_query in pair.folder.lower()
        ]
    if not candidates:
        raise SystemExit(f"No flight matched {args.flight!r}. Use --list-flights.")
    pair = candidates[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as zf:
        vehicle = load_vehicle_out(zf, pair.vehicle_out)
        radar_df = associate_radar_to_gt(zf, pair.radar_json, vehicle, max_assoc_m=args.max_assoc_m)

    dt = float(np.median(np.diff(radar_df["time"].to_numpy(dtype=float))))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.256
    replay_df = run_filter_on_radar(radar_df, dt=dt, radar_sigma_m=args.radar_sigma_m)
    summary = summarise(replay_df)

    flight_label = re.sub(r"[^A-Za-z0-9_]+", "_", pair.flight_id)
    radar_df.to_csv(out_dir / f"{flight_label}_associated_radar_measurements.csv", index=False)
    replay_df.to_csv(out_dir / f"{flight_label}_aerpaw_radar_timeseries.csv", index=False)
    summary.to_csv(out_dir / f"{flight_label}_aerpaw_radar_summary.csv", index=False)
    plot_results(replay_df, flight_label, out_dir)

    print(f"Flight: {pair.flight_id}")
    print(f"Radar file: {pair.radar_json}")
    print(f"Ground truth: {pair.vehicle_out}")
    print(f"Associated radar measurements: {len(radar_df)}")
    print(f"Median radar dt: {dt:.3f}s")
    print(f"Median raw radar measurement error: {radar_df['assoc_error_m'].median():.2f} m")
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
            f"{row.label:28s} {row.rmse_m:8.3f} {row.mean_nees:8.2f} "
            f"{row.lqg_cost:12.1f} {row.mean_dt_err_m:8.3f}{suffix}"
        )
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
