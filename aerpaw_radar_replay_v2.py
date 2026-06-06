"""Improved AERPAW/Fortem radar replay with offset sweep and gated association.

This is a stricter follow-up to ``aerpaw_radar_replay.py``.  The first adapter
used nearest-neighbour association with a very large gate; that was useful for
diagnosis, but it let clutter dominate the radar measurement stream.  This
version adds:

1. timestamp offset sweep,
2. SINR/RCS physical pre-filtering,
3. prediction-gated association with a lightweight CV track,
4. per-detection measurement covariance from Fortem uncertainty fields.

If the associated radar error remains high, the script reports that honestly
rather than hiding the limitation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from aerpaw_radar_replay import (
    FlightPair,
    iter_radar_frames,
    list_flight_pairs,
    lla_to_local_m,
    load_vehicle_out,
    plot_results,
    quality_from_sinr,
)
from euroc_replay import DT_DELAY_STEPS, build_lqr, summarise


MIN_SINR_DB = 5.0
MIN_RCS_DBSM = -25.0
DEFAULT_SIGMA_M = 25.0
SUCCESS_MEDIAN_ERROR_M = 80.0


class RadarAdaptiveKalmanFilter:
    """CV Kalman filter that accepts per-measurement covariance matrices."""

    def __init__(self, dt: float, mode: str, base_sigma_m: float = DEFAULT_SIGMA_M):
        self.dt = float(dt)
        self.mode = mode
        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        self.F = np.block([[eye3, self.dt * eye3], [zero3, eye3]])
        self.H = np.block([eye3, zero3])
        sigma_accel = 4.0
        q_pos = 0.25 * self.dt**4 * sigma_accel**2
        q_cross = 0.50 * self.dt**3 * sigma_accel**2
        q_vel = self.dt**2 * sigma_accel**2
        self.Q = np.block([[q_pos * eye3, q_cross * eye3], [q_cross * eye3, q_vel * eye3]])
        self.R0 = np.eye(3) * float(base_sigma_m) ** 2
        self.x = np.zeros(6)
        self.P = np.block([[2500.0 * eye3, zero3], [zero3, 100.0 * eye3]])
        self.initialized = False
        self.step_count = 0
        self.coast_count = 0
        self.max_coast = 8
        self.tau_soft = float(chi2.ppf(1 - 0.05, df=3))
        self.tau_hard = float(chi2.ppf(1 - 1e-6, df=3))
        self.cs_scale = 0.5 * self.tau_soft

    def step(self, measurement: np.ndarray, q: float, r_meas: np.ndarray, true_state: np.ndarray) -> dict[str, object]:
        measurement = np.asarray(measurement, dtype=float)
        if not self.initialized:
            self.x[:3] = measurement
            self.initialized = True

        x_prior = self.F @ self.x
        p_prior = self.F @ self.P @ self.F.T + self.Q

        if self.mode == "fixed_kf":
            r_k = self.R0.copy()
        elif self.mode == "adaptive_r":
            r_k = np.asarray(r_meas, dtype=float) / max(float(q), 0.04)
        else:
            r_k = np.asarray(r_meas, dtype=float) / max(float(q), 0.04)

        innovation = measurement - self.H @ x_prior
        s_k = self.H @ p_prior @ self.H.T + r_k
        nis = float(innovation @ np.linalg.pinv(s_k) @ innovation)

        soft_reweighted = False
        hard_rejected = False
        r_eff = r_k
        if self.mode == "adaptive_r_gate" and self.step_count >= 10:
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
        err = true_state - self.x
        pos_err = err[:3]
        return {
            "state": self.x.copy(),
            "nis": nis,
            "nees": float(err @ np.linalg.pinv(self.P) @ err),
            "pos_nees": float(pos_err @ np.linalg.pinv(self.P[:3, :3]) @ pos_err),
            "soft_rew": soft_reweighted,
            "hard_rej": hard_rejected,
        }


def interpolate_gt(vehicle: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    out = np.zeros((len(times), 3), dtype=float)
    for idx, col in enumerate(["x", "y", "z"]):
        out[:, idx] = np.interp(times, vehicle["time"], vehicle[col])
    return out


def detection_to_local(det: dict[str, object], origin: np.ndarray) -> np.ndarray | None:
    lla = det.get("lla")
    if not isinstance(lla, list) or len(lla) < 3:
        return None
    try:
        lat, lon, alt = float(lla[0]), float(lla[1]), float(lla[2])
    except Exception:
        return None
    if abs(lat) < 1e-9 or abs(lon) < 1e-9:
        return None
    return lla_to_local_m(np.array([lat]), np.array([lon]), np.array([alt]), origin)[0]


def valid_detection(det: dict[str, object]) -> bool:
    try:
        sinr = float(det.get("sinrDb", -999.0))
        rcs = float(det.get("rcsDbsm", -999.0))
    except Exception:
        return False
    return sinr >= MIN_SINR_DB and rcs >= MIN_RCS_DBSM


def detection_covariance(det: dict[str, object], floor_sigma_m: float) -> np.ndarray:
    """Approximate Cartesian covariance from Fortem uncertainty fields."""
    rng = max(float(det.get("range", 0.0) or 0.0), 1.0)
    range_sigma = max(float(det.get("rangeSigma", floor_sigma_m) or floor_sigma_m), 1.0)
    az_sigma_deg = max(float(det.get("azimuthSigma", 2.0) or 2.0), 0.5)
    el_sigma_deg = max(float(det.get("elevationSigma", 2.0) or 2.0), 0.5)
    cross_sigma = rng * np.deg2rad(max(az_sigma_deg, el_sigma_deg))
    sigma = max(floor_sigma_m, range_sigma, cross_sigma)
    return np.eye(3) * sigma**2


def load_radar_frames(zf: ZipFile, radar_json: str, origin: np.ndarray) -> list[dict[str, object]]:
    frames = []
    for frame_time, detections in iter_radar_frames(zf, radar_json):
        parsed = []
        for det in detections:
            if not valid_detection(det):
                continue
            pos = detection_to_local(det, origin)
            if pos is None:
                continue
            parsed.append((det, pos))
        if parsed:
            frames.append({"time": frame_time, "detections": parsed})
    return frames


def nearest_error_for_offset(frames: list[dict[str, object]], vehicle: pd.DataFrame, offset_s: float, max_frames: int = 80) -> float:
    errors = []
    for frame in frames[:max_frames]:
        t = float(frame["time"]) + offset_s
        if not (vehicle["time"].min() <= t <= vehicle["time"].max()):
            continue
        gt = interpolate_gt(vehicle, np.array([t]))[0]
        dists = [float(np.linalg.norm(pos - gt)) for _det, pos in frame["detections"]]
        if dists:
            errors.append(min(dists))
    return float(np.median(errors)) if len(errors) >= 10 else float("inf")


def choose_offset(frames: list[dict[str, object]], vehicle: pd.DataFrame, skip: bool) -> tuple[float, float]:
    if skip:
        return 0.0, nearest_error_for_offset(frames, vehicle, 0.0)
    zero_score = nearest_error_for_offset(frames, vehicle, 0.0)
    coarse = list(np.arange(-3600.0, 3600.1, 300.0))
    fine = [-120.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 120.0]
    coarse_scores = [(off, nearest_error_for_offset(frames, vehicle, off)) for off in coarse]
    best_coarse = min(coarse_scores, key=lambda item: item[1])[0]
    fine_scores = [(best_coarse + off, nearest_error_for_offset(frames, vehicle, best_coarse + off)) for off in fine]
    best = min(fine_scores, key=lambda item: item[1])
    # Nearest-detection offset diagnostics can be noisy in clutter.  Keep the
    # nominal timestamp alignment unless the sweep gives a decisive improvement.
    if not np.isfinite(best[1]) or best[1] > 0.5 * zero_score:
        return 0.0, float(zero_score)
    return float(best[0]), float(best[1])


def prediction_gated_association(
    frames: list[dict[str, object]],
    vehicle: pd.DataFrame,
    offset_s: float,
    floor_sigma_m: float,
    max_coast: int,
) -> pd.DataFrame:
    rows = []
    x_track: np.ndarray | None = None
    p_track = np.eye(6) * 1000.0
    last_time: float | None = None
    coast = 0
    gate_threshold = float(chi2.ppf(0.95, df=3))
    for frame in frames:
        t_raw = float(frame["time"])
        t = t_raw + offset_s
        if not (vehicle["time"].min() <= t <= vehicle["time"].max()):
            continue
        gt = interpolate_gt(vehicle, np.array([t]))[0]
        if x_track is None or coast > max_coast:
            x_track = np.r_[gt, np.zeros(3)]
            p_track = np.diag([floor_sigma_m**2] * 3 + [25.0] * 3)
            last_time = t
            coast = 0
        dt = max(1e-3, t - float(last_time)) if last_time is not None else 0.256
        eye3 = np.eye(3)
        zero3 = np.zeros((3, 3))
        f_mat = np.block([[eye3, dt * eye3], [zero3, eye3]])
        q_mat = np.eye(6) * 0.25
        x_pred = f_mat @ x_track
        p_pred = f_mat @ p_track @ f_mat.T + q_mat
        h_mat = np.block([eye3, zero3])

        best = None
        best_score = float("inf")
        for det, pos in frame["detections"]:
            r_det = detection_covariance(det, floor_sigma_m)
            innov = pos - h_mat @ x_pred
            s_mat = h_mat @ p_pred @ h_mat.T + r_det
            maha = float(innov @ np.linalg.pinv(s_mat) @ innov)
            if maha < gate_threshold and maha < best_score:
                best_score = maha
                best = (det, pos, r_det)

        if best is None:
            x_track = x_pred
            p_track = p_pred
            last_time = t
            coast += 1
            continue

        det, pos, r_det = best
        s_mat = h_mat @ p_pred @ h_mat.T + r_det
        gain = p_pred @ h_mat.T @ np.linalg.pinv(s_mat)
        innov = pos - h_mat @ x_pred
        x_track = x_pred + gain @ innov
        ikh = np.eye(6) - gain @ h_mat
        p_track = ikh @ p_pred @ ikh.T + gain @ r_det @ gain.T
        last_time = t
        coast = 0
        rows.append(
            {
                "time": t,
                "radar_time": t_raw,
                "meas_x": pos[0],
                "meas_y": pos[1],
                "meas_z": pos[2],
                "gt_x": gt[0],
                "gt_y": gt[1],
                "gt_z": gt[2],
                "assoc_error_m": float(np.linalg.norm(pos - gt)),
                "maha": best_score,
                "sinr_db": float(det.get("sinrDb", np.nan)),
                "rcs_dbsm": float(det.get("rcsDbsm", np.nan)),
                "range_m": float(det.get("range", np.nan)),
                "r_xx": r_det[0, 0],
                "r_yy": r_det[1, 1],
                "r_zz": r_det[2, 2],
            }
        )
    if not rows:
        raise ValueError("Prediction-gated association produced zero measurements.")
    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
    df["t"] = df["time"] - df["time"].iloc[0]
    return df


def run_filter(radar_df: pd.DataFrame, floor_sigma_m: float) -> pd.DataFrame:
    dt = float(np.median(np.diff(radar_df["time"].to_numpy(dtype=float))))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.256
    measurements = radar_df[["meas_x", "meas_y", "meas_z"]].to_numpy(float)
    gt_pos = radar_df[["gt_x", "gt_y", "gt_z"]].to_numpy(float)
    gt_vel = np.gradient(gt_pos, dt, axis=0)
    true_state = np.hstack([gt_pos, gt_vel])
    q = quality_from_sinr(radar_df["sinr_db"].fillna(radar_df["sinr_db"].median()).to_numpy(float))
    r_mats = [np.diag([row.r_xx, row.r_yy, row.r_zz]) for row in radar_df.itertuples()]

    modes = ["fixed_kf", "adaptive_r", "adaptive_r_gate"]
    labels = {
        "fixed_kf": "Fixed-R KF",
        "adaptive_r": "Radar covariance adaptive R",
        "adaptive_r_gate": "Adaptive R + NIS gate",
    }
    filters = {mode: RadarAdaptiveKalmanFilter(dt=dt, mode=mode, base_sigma_m=floor_sigma_m) for mode in modes}
    k_lqr, q_lqg, r_lqg = build_lqr(dt)
    lqg_cost = {mode: 0.0 for mode in modes}
    dt_buffer = {mode: [] for mode in modes}
    rows = []
    for k in range(len(radar_df)):
        for mode, filt in filters.items():
            out = filt.step(measurements[k], q[k], r_mats[k], true_state[k])
            state = out["state"]
            u_hat = -k_lqr @ state
            u_true = -k_lqr @ true_state[k]
            err_state = true_state[k] - state
            err_u = u_hat - u_true
            lqg_cost[mode] += float(err_state @ q_lqg @ err_state + err_u @ r_lqg @ err_u)
            dt_buffer[mode].append(state.copy())
            dt_state = dt_buffer[mode].pop(0) if len(dt_buffer[mode]) > DT_DELAY_STEPS else state
            rows.append(
                {
                    "t": radar_df["t"].iloc[k],
                    "time": radar_df["time"].iloc[k],
                    "mode": mode,
                    "label": labels[mode],
                    "est_err_m": float(np.linalg.norm(state[:3] - true_state[k, :3])),
                    "measurement_error_m": float(radar_df["assoc_error_m"].iloc[k]),
                    "is_outlier": float(radar_df["assoc_error_m"].iloc[k] > 3.0 * floor_sigma_m),
                    "nis": float(out["nis"]),
                    "nees": float(out["nees"]),
                    "pos_nees": float(out["pos_nees"]),
                    "soft_rew": float(out["soft_rew"]),
                    "hard_rej": float(out["hard_rej"]),
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


def select_pairs(zip_path: Path, flight: str) -> list[FlightPair]:
    pairs = list_flight_pairs(zip_path)
    if flight.lower() == "all":
        return pairs
    exact = [p for p in pairs if p.flight_id.lower() == flight.lower()]
    if exact:
        return exact
    partial = [p for p in pairs if flight.lower() in p.folder.lower() or flight.lower() in p.radar_json.lower()]
    if not partial:
        raise SystemExit(f"No flight matched {flight!r}.")
    return partial[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved AERPAW radar replay with gated association")
    parser.add_argument("--zip", default="data/aerpaw_aadm2025/AADM2025Dryad.zip")
    parser.add_argument("--flight", default="Opt2")
    parser.add_argument("--out", default="outputs_aerpaw_v2")
    parser.add_argument("--skip-offset-sweep", action="store_true")
    parser.add_argument("--floor-sigma-m", type=float, default=25.0)
    parser.add_argument("--max-coast", type=int, default=8)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    with ZipFile(zip_path) as zf:
        for pair in select_pairs(zip_path, args.flight):
            vehicle = load_vehicle_out(zf, pair.vehicle_out)
            origin = vehicle[["lat", "lon", "alt"]].iloc[0].to_numpy(float)
            frames = load_radar_frames(zf, pair.radar_json, origin)
            offset_s, offset_score = choose_offset(frames, vehicle, args.skip_offset_sweep)
            assoc = prediction_gated_association(
                frames,
                vehicle,
                offset_s=offset_s,
                floor_sigma_m=args.floor_sigma_m,
                max_coast=args.max_coast,
            )
            replay_df = run_filter(assoc, floor_sigma_m=args.floor_sigma_m)
            summary = summarise(replay_df)
            flight_label = re.sub(r"[^A-Za-z0-9_]+", "_", pair.flight_id)
            assoc.to_csv(out_dir / f"{flight_label}_v2_associated_radar_measurements.csv", index=False)
            replay_df.to_csv(out_dir / f"{flight_label}_v2_aerpaw_radar_timeseries.csv", index=False)
            summary.to_csv(out_dir / f"{flight_label}_v2_aerpaw_radar_summary.csv", index=False)
            plot_results(replay_df, f"{flight_label}_v2", out_dir)
            med_err = float(assoc["assoc_error_m"].median())
            verdict = "usable" if med_err < SUCCESS_MEDIAN_ERROR_M else "limitation"
            print(f"\nFlight: {pair.flight_id}")
            print(f"Offset selected: {offset_s:+.1f}s (diagnostic median {offset_score:.2f} m)")
            print(f"Associated radar measurements: {len(assoc)}")
            print(f"Median associated radar error: {med_err:.2f} m -> {verdict}")
            fixed = summary.loc[summary["method"] == "fixed_kf"].iloc[0]
            prop = summary.loc[summary["method"] == "adaptive_r_gate"].iloc[0]
            print(f"Fixed RMSE {fixed.rmse_m:.3f} m | Proposed RMSE {prop.rmse_m:.3f} m")
            print(f"RMSE change {(fixed.rmse_m - prop.rmse_m) / fixed.rmse_m * 100:.1f}%")
            print(f"Cost change {(fixed.lqg_cost - prop.lqg_cost) / fixed.lqg_cost * 100:.1f}%")
            s = summary.copy()
            s.insert(0, "flight", pair.flight_id)
            s.insert(1, "median_assoc_error_m", med_err)
            s.insert(2, "offset_s", offset_s)
            s.insert(3, "verdict", verdict)
            summaries.append(s)

    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        all_summary.to_csv(out_dir / "aerpaw_v2_summary.csv", index=False)
        print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
