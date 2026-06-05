from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paper_sim.controllers import FailureAwareSupervisor, GenericSupervisor, PositionPID, SupervisorCommand
from paper_sim.attack_injection import (
    buffer_neighbor_states,
    inject_jamming,
    inject_replay,
    inject_spoofing,
    reset_attack_state,
)
from paper_sim.diagnosis import FaultDiagnoser
from paper_sim.digital_twin import DigitalTwin
from paper_sim.env import make_env
from paper_sim.estimation import ConstantVelocityEstimator
from paper_sim.faults import FaultModel
from paper_sim.ids import IDS, AttackClass
from paper_sim.metrics import (
    connectivity_rate,
    degradation_pct,
    formation_error,
    max_pairwise_spacing_error,
    mean_neighbor_count,
    recovery_time,
    rmse,
    settling_time,
    spacing_violation_count,
    time_above_threshold,
    tracking_errors,
)
from paper_sim.security_metrics import SecurityMetrics
from paper_sim.trust_mpc import TrustMPC
from paper_sim.trajectory import formation_offsets, reference_state


CONDITION_SPECS = {
    "nominal": {"fault_scenario": "nominal", "attack": None},
    "wind": {"fault_scenario": "wind", "attack": None},
    "sensor": {"fault_scenario": "sensor", "attack": None},
    "comm": {"fault_scenario": "comm", "attack": None},
    "wind_comm": {"fault_scenario": "wind_comm", "attack": None},
    "jamming": {"fault_scenario": "jamming", "attack": "jamming", "jam_power": 1.0},
    "jamming_full": {"fault_scenario": "jamming", "attack": "jamming", "jam_power": 1.0},
    "jamming_partial": {"fault_scenario": "jamming", "attack": "jamming", "jam_power": 0.5},
    "spoofing": {"fault_scenario": "spoofing", "attack": "spoofing", "d_spoof": [1.5, 0.0, 0.0]},
    "spoofing_strong": {"fault_scenario": "spoofing", "attack": "spoofing", "d_spoof": [1.5, 0.0, 0.0]},
    "spoofing_subtle": {"fault_scenario": "spoofing", "attack": "spoofing", "d_spoof": [0.5, 0.0, 0.0]},
    "replay": {"fault_scenario": "replay", "attack": "replay", "replay_delay_s": 5.0},
    "compound": {"fault_scenario": "wind", "attack": "jamming", "jam_power": 1.0},
}
SCENARIOS = list(CONDITION_SPECS)
ATTACK_CLASS_BY_SCENARIO = {
    "nominal": AttackClass.H0_NONE,
    "wind": AttackClass.H1_WIND,
    "sensor": AttackClass.H0_NONE,
    "comm": AttackClass.H0_NONE,
    "wind_comm": AttackClass.H1_WIND,
    "jamming": AttackClass.H2_JAMMING,
    "jamming_full": AttackClass.H2_JAMMING,
    "jamming_partial": AttackClass.H2_JAMMING,
    "spoofing": AttackClass.H3_SPOOF,
    "spoofing_strong": AttackClass.H3_SPOOF,
    "spoofing_subtle": AttackClass.H3_SPOOF,
    "replay": AttackClass.H4_REPLAY,
    "compound": AttackClass.H2_JAMMING,
}
CONTROLLER_ALIASES = {
    "pid_baseline": "pid",
    "prior_supervisory": "generic",
    "proposed_ids_mpc": "failure_aware",
}
CONTROLLERS = ["pid", "generic", "failure_aware", *CONTROLLER_ALIASES]
ABLATIONS = [
    "full",
    "no_diagnosis",
    "no_filter",
    "no_comm_fallback",
    "no_wind_comp",
    "no_recovery_schedule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--controller", required=True, choices=CONTROLLERS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default="paper_sim/outputs")
    parser.add_argument("--ablation", default="full", choices=ABLATIONS)
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def controller_label(controller_name: str, ablation: str) -> str:
    controller_name = CONTROLLER_ALIASES.get(controller_name, controller_name)
    if controller_name != "failure_aware" or ablation == "full":
        return controller_name
    return f"{controller_name}:{ablation}"


def window_mean(df: pd.DataFrame, start_s: float, end_s: float, col: str) -> float:
    seg = df[(df["t"] >= start_s) & (df["t"] < end_s)]
    return float(seg[col].mean()) if not seg.empty else float("nan")


def _neighbor_state_from_broadcasts(
    broadcast_pos: dict[int, np.ndarray],
    broadcast_vel: dict[int, np.ndarray],
    t_s: float,
    num: int,
) -> dict[int, dict[int, dict[str, np.ndarray | float]]]:
    neighbor_states: dict[int, dict[int, dict[str, np.ndarray | float]]] = {}
    for i in range(num):
        neighbor_states[i] = {}
        for j in range(num):
            if i == j:
                continue
            neighbor_states[i][j] = {
                "pos": np.asarray(broadcast_pos[j], dtype=float).copy(),
                "vel": np.asarray(broadcast_vel[j], dtype=float).copy(),
                "timestamp": float(t_s),
            }
    return neighbor_states


def run_once(cfg: dict, scenario: str, controller_name: str, seed: int, output_root: Path, ablation: str = "full") -> tuple[pd.DataFrame, dict]:
    np.random.seed(seed)
    controller_name = CONTROLLER_ALIASES.get(controller_name, controller_name)
    condition_spec = CONDITION_SPECS[scenario]
    fault_scenario = str(condition_spec["fault_scenario"])
    attack_type = condition_spec.get("attack")
    env_handles = make_env(cfg, gui=False)
    swarm = env_handles.env
    num = int(cfg["sim"]["num_drones"])
    ctrl_decim = max(1, int(round(env_handles.dt_ctrl / env_handles.dt_sim)))
    total_steps = int(round(float(cfg["sim"]["duration_s"]) / env_handles.dt_sim))
    comm_range_m = float(cfg["network"]["comm_range_m"])
    formation_gain = float(cfg["sim"].get("formation_consensus_gain", 0.18))
    min_spacing_m = float(cfg["safety"]["min_spacing_m"])
    safety_error_threshold_m = float(cfg["analysis"]["safety_error_threshold_m"])

    offsets_nominal = formation_offsets(cfg, num)
    pid = PositionPID(cfg)
    pid.reset(num)
    generic = GenericSupervisor(cfg)
    failure_aware = FailureAwareSupervisor(cfg)
    faults = FaultModel(cfg, fault_scenario, env_handles.dt_ctrl, seed, num)
    diagnoser = FaultDiagnoser(cfg)
    twin_cfg = cfg.get("security", {}).get("digital_twin", {})
    twin = DigitalTwin(
        n_agents=num,
        dt=env_handles.dt_ctrl,
        sigma_p=float(twin_cfg.get("sigma_p", 0.02)),
        sigma_v=float(twin_cfg.get("sigma_v", 0.05)),
        sigma_meas=float(twin_cfg.get("sigma_meas", 0.02)),
    )
    twin.initialize({idx: swarm.positions[idx] for idx in range(num)})
    ids_mod = IDS(n_agents=num, dt=env_handles.dt_ctrl)
    trust_mpc = TrustMPC(twin, n_agents=num, dt=env_handles.dt_ctrl)
    reset_attack_state(n_agents=num)

    est_cfg = cfg["estimation"]
    estimators = {
        idx: ConstantVelocityEstimator(
            dt=env_handles.dt_ctrl,
            base_gain=float(est_cfg["base_gain"]),
            min_gain=float(est_cfg["min_gain"]),
            max_gain=float(est_cfg["max_gain"]),
            outlier_gate_m=float(est_cfg["outlier_gate_m"]),
            nis_gate=float(est_cfg["nis_gate"]),
            predicted_var_xy=float(est_cfg["predicted_var_xy"]),
            measurement_var_xy=float(est_cfg["measurement_var_xy"]),
            bias_gain=float(est_cfg["bias_gain"]),
            bias_decay=float(est_cfg["bias_decay"]),
            bias_enable_conf=float(est_cfg["bias_enable_conf"]),
            bias_limit_m=float(est_cfg["bias_limit_m"]),
        )
        for idx in range(num)
    }
    for idx in range(num):
        estimators[idx].reset(swarm.positions[idx], swarm.velocities[idx])

    last_ref_view = {idx: swarm.positions[idx].copy() for idx in range(num)}
    last_vel_view = {idx: np.zeros(3, dtype=float) for idx in range(num)}
    last_comm_quality = 1.0
    last_mean_error = 0.0
    filtered_pos = {idx: swarm.positions[idx].copy() for idx in range(num)}
    filtered_vel = {idx: swarm.velocities[idx].copy() for idx in range(num)}
    prev_true_vel = {idx: swarm.velocities[idx].copy() for idx in range(num)}
    disturbance_est = {idx: np.zeros(3, dtype=float) for idx in range(num)}
    last_disturbance_norm = 0.0
    dob_cfg = cfg["wind_observer"]

    rows: list[dict] = []
    accel_cmd = np.zeros((num, 3), dtype=float)
    target_pos_cmd = swarm.positions.copy()
    target_vel_cmd = np.zeros((num, 3), dtype=float)
    active_wind = np.zeros((num, 3), dtype=float)
    pre0, pre1 = cfg["analysis"]["pre_fault_window_s"]
    fault0, fault1 = cfg["analysis"]["fault_window_s"]
    post0, post1 = cfg["analysis"]["post_fault_window_s"]
    sec_metrics = SecurityMetrics(
        attack_start=float(fault0),
        attack_end=float(fault1),
        nominal_start=float(pre0),
        nominal_end=float(pre1),
        true_attack_class=ATTACK_CLASS_BY_SCENARIO.get(scenario, AttackClass.H0_NONE),
    )

    for step in range(total_steps):
        t_s = step * env_handles.dt_sim
        if step % ctrl_decim == 0:
            pos_true = swarm.positions.copy()
            vel_true = swarm.velocities.copy()

            measured_pos = np.zeros_like(pos_true)
            raw_sensor_residuals = []
            measurements_missing = 0
            for i in range(num):
                raw = faults.measurement(pos_true[i], t_s, i)
                measured_pos[i] = raw
                if np.any(~np.isfinite(raw)):
                    measurements_missing += 1
                    raw_sensor_residuals.append(float(est_cfg["outlier_gate_m"]))
                else:
                    raw_sensor_residuals.append(float(np.linalg.norm((raw - pos_true[i])[:2])))

            group_drift_m = float(np.linalg.norm((np.mean(pos_true, axis=0) - reference_state(cfg, t_s)[0])[:2]))
            comm_probe = []
            if fault_scenario in {"comm", "wind_comm"} and faults.fault_active(t_s):
                for i in range(num):
                    probe = faults.comm_view(np.array([1.0]), i, -99, t_s)
                    comm_probe.append(0.0 if probe is None else 1.0)
            else:
                comm_probe = [1.0] * num
            comm_quality = float(np.mean(comm_probe)) if comm_probe else 1.0
            diagnosis = diagnoser.step(
                sensor_residual_m=float(np.mean(raw_sensor_residuals)),
                wind_residual_m=max(abs(group_drift_m - last_mean_error), last_disturbance_norm),
                comm_quality=comm_quality,
            )

            measured_or_estimated = np.zeros_like(pos_true)
            estimated_vel_array = np.zeros_like(vel_true)
            innovation_norms = []
            measurement_weights = []
            nis_values = []
            accept_flags = []
            bias_norms = []
            for i in range(num):
                measurement = None if np.any(~np.isfinite(measured_pos[i])) else measured_pos[i]
                trust_scale = 1.0
                if ablation == "no_filter":
                    trust_scale = 1.0
                snapshot = estimators[i].step(
                    measurement=measurement,
                    sensor_confidence=0.0 if ablation == "no_filter" else diagnosis.confidences.sensor,
                    trust_scale=trust_scale,
                    control_input=accel_cmd[i],
                )
                filtered_pos[i] = snapshot.filtered_pos
                filtered_vel[i] = snapshot.filtered_vel
                innovation_norms.append(snapshot.innovation_norm)
                measurement_weights.append(snapshot.measurement_weight)
                nis_values.append(snapshot.nis)
                accept_flags.append(float(snapshot.measurement_accepted))
                bias_norms.append(float(np.linalg.norm(snapshot.bias_estimate_xy)))
                if controller_name == "failure_aware" and ablation != "no_filter":
                    measured_or_estimated[i] = snapshot.filtered_pos
                    estimated_vel_array[i] = snapshot.filtered_vel
                else:
                    if measurement is None:
                        measured_or_estimated[i] = snapshot.filtered_pos
                        estimated_vel_array[i] = snapshot.filtered_vel
                    else:
                        measured_or_estimated[i] = measurement
                        estimated_vel_array[i] = snapshot.filtered_vel

            p_ref, v_ref = reference_state(cfg, t_s)
            references_nominal = p_ref[None, :] + offsets_nominal
            broadcast_pos = {idx: measured_or_estimated[idx].copy() for idx in range(num)}
            broadcast_vel = {idx: estimated_vel_array[idx].copy() for idx in range(num)}
            attack_cfg = cfg.get("security", {}).get("attacks", {})

            neighbor_states = _neighbor_state_from_broadcasts(broadcast_pos, broadcast_vel, t_s, num)
            neighbor_states_clean = {
                i: {j: dict(state) for j, state in neighbors.items()}
                for i, neighbors in neighbor_states.items()
            }
            qcomm = {idx: 1.0 for idx in range(num)}
            buffer_neighbor_states(neighbor_states, t_s)
            if attack_type == "spoofing":
                spoof_cfg = attack_cfg.get("spoofing", {})
                neighbor_states = inject_spoofing(
                    neighbor_states,
                    t_s,
                    target_agent=int(spoof_cfg.get("target_agent", 0)),
                    d_spoof=np.asarray(condition_spec.get("d_spoof", spoof_cfg.get("offset_m", [1.5, 0.0, 0.0])), dtype=float),
                    start_t=float(fault0),
                    end_t=float(fault1),
                )
            elif attack_type == "jamming":
                jam_cfg = attack_cfg.get("jamming", {})
                neighbor_states, qcomm = inject_jamming(
                    neighbor_states,
                    qcomm,
                    t_s,
                    start_t=float(fault0),
                    end_t=float(fault1),
                    jam_power=float(condition_spec.get("jam_power", jam_cfg.get("jam_power", 1.0))),
                    n_agents=num,
                    run_seed=seed,
                )
            elif attack_type == "replay":
                replay_cfg = attack_cfg.get("replay", {})
                neighbor_states = inject_replay(
                    neighbor_states,
                    t_s,
                    replay_delay=float(condition_spec.get("replay_delay_s", replay_cfg.get("replay_delay_s", 5.0))),
                    start_t=float(fault0),
                    end_t=float(fault1),
                    n_agents=num,
                )

            twin.predict(accel_cmd)
            twin.update(measured_pos)
            ids_out = ids_mod.step(twin, neighbor_states, qcomm, t_s, attack_start_t=float(fault0))

            mean_err_nominal, max_err_nominal = tracking_errors(pos_true, references_nominal)
            form_err_nominal = formation_error(pos_true, offsets_nominal)
            connectivity = connectivity_rate(pos_true, comm_range_m)
            center_xy = np.mean(pos_true[:, :2], axis=0)
            group_error_xy = center_xy - p_ref[:2]

            if controller_name == "pid":
                sup = generic.step(-1.0, -1.0)
            elif controller_name == "generic":
                sup = generic.step(mean_err_nominal, form_err_nominal)
            else:
                if cfg.get("sim", {}).get("backend") == "pybullet_drones":
                    sup = SupervisorCommand()
                else:
                    sup = failure_aware.step(diagnosis, group_error_xy, ablation=ablation)

            references_cmd = np.zeros_like(pos_true)
            target_vel_cmd = np.zeros_like(pos_true)
            effective_offsets = offsets_nominal * sup.formation_scale
            trust_delta = trust_mpc.step(references_nominal, effective_offsets, ids_out, neighbor_states, t_s)
            delta_norms = {idx: float(np.linalg.norm(trust_delta[idx])) for idx in range(num)}
            comm_health_samples = []
            control_effort = 0.0
            compensation_norm = 0.0
            integral_norms = []
            saturation_flags = []

            for i in range(num):
                ref_pos = p_ref.copy()
                ref_vel = v_ref.copy()
                if fault_scenario in {"comm", "wind_comm"} and faults.fault_active(t_s):
                    ref_view = faults.comm_view(p_ref, i, -1, t_s)
                    vel_view = faults.comm_view(v_ref, i, -2, t_s)
                    if ref_view is not None:
                        last_ref_view[i] = ref_view.copy()
                        comm_health_samples.append(1.0)
                    else:
                        comm_health_samples.append(0.0)
                    if vel_view is not None:
                        last_vel_view[i] = vel_view.copy()
                        comm_health_samples.append(1.0)
                    else:
                        comm_health_samples.append(0.0)
                    ref_pos = last_ref_view[i].copy()
                    ref_vel = last_vel_view[i].copy()
                else:
                    comm_health_samples.extend([1.0, 1.0])

                ref_pos[:2] += sup.connectivity_bias_xy
                target_pos = ref_pos + effective_offsets[i]

                consensus_sum = np.zeros(2, dtype=float)
                consensus_count = 0
                for j in range(num):
                    if i == j:
                        continue
                    neighbor_view = measured_or_estimated[j].copy()
                    if attack_type in {"jamming", "spoofing", "replay"}:
                        state = neighbor_states.get(i, {}).get(j)
                        if controller_name == "failure_aware" and ids_out["k_hat"] != AttackClass.H0_NONE:
                            comm_health_samples.append(float(qcomm.get(i, 1.0)))
                            neighbor_view = twin.get_state(j)[:3]
                        elif state is None:
                            comm_health_samples.append(0.0)
                            if controller_name == "failure_aware" and ablation != "no_comm_fallback":
                                neighbor_view = filtered_pos[j].copy()
                            else:
                                continue
                        else:
                            comm_health_samples.append(float(qcomm.get(i, 1.0)))
                            neighbor_view = np.asarray(state["pos"], dtype=float)
                    elif fault_scenario in {"comm", "wind_comm"} and faults.fault_active(t_s):
                        delayed_view = faults.comm_view(measured_or_estimated[j], i, j, t_s)
                        if delayed_view is None:
                            comm_health_samples.append(0.0)
                            if controller_name == "failure_aware" and ablation != "no_comm_fallback":
                                neighbor_view = filtered_pos[j].copy()
                            else:
                                continue
                        else:
                            comm_health_samples.append(1.0)
                            neighbor_view = delayed_view
                    else:
                        comm_health_samples.append(1.0)
                    desired_delta = effective_offsets[j][:2] - effective_offsets[i][:2]
                    actual_delta = neighbor_view[:2] - measured_or_estimated[i][:2]
                    consensus_sum += actual_delta - desired_delta
                    consensus_count += 1
                if consensus_count > 0:
                    target_pos[:2] += formation_gain * sup.consensus_scale * consensus_sum / consensus_count
                if controller_name == "failure_aware" and attack_type in {"jamming", "spoofing", "replay"}:
                    target_pos += trust_delta[i]

                references_cmd[i] = target_pos
                target_pos_cmd[i] = target_pos
                target_vel_cmd[i] = ref_vel * sup.speed_scale
                observed_accel = (vel_true[i] - prev_true_vel[i]) / env_handles.dt_ctrl
                dob_residual = observed_accel - accel_cmd[i]
                disturbance_est[i] = (
                    float(dob_cfg["lpf_alpha"]) * disturbance_est[i]
                    + (1.0 - float(dob_cfg["lpf_alpha"])) * dob_residual
                )
                disturbance_ff = np.zeros(3, dtype=float)
                if controller_name == "failure_aware" and diagnosis.active_fault == "wind" and ablation != "no_wind_comp":
                    disturbance_ff = -float(dob_cfg["ff_gain"]) * disturbance_est[i]
                    ff_norm = np.linalg.norm(disturbance_ff[:2])
                    ff_cap = float(dob_cfg["ff_limit_xy"])
                    if ff_norm > ff_cap:
                        disturbance_ff[:2] *= ff_cap / (ff_norm + 1e-9)
                    disturbance_ff[2] = float(np.clip(disturbance_ff[2], -float(dob_cfg["ff_limit_z"]), float(dob_cfg["ff_limit_z"])))
                accel_out, pid_debug = pid.compute_accel(
                    drone_id=i,
                    measured_pos=measured_or_estimated[i],
                    measured_vel=estimated_vel_array[i],
                    target_pos=target_pos,
                    target_vel=ref_vel * sup.speed_scale,
                    extra_damping=sup.damping_gain,
                )
                accel_cmd[i] = accel_out + disturbance_ff
                control_effort += float(np.linalg.norm(accel_cmd[i]))
                compensation_norm += float(np.linalg.norm(disturbance_ff))
                integral_norms.append(pid_debug["integral_norm"])
                saturation_flags.append(pid_debug["saturation_flag"])
                prev_true_vel[i] = vel_true[i].copy()

            active_wind = faults.wind_accel(num, t_s)
            mean_err_cmd, max_err_cmd = tracking_errors(pos_true, references_cmd)
            comm_health = float(np.mean(comm_health_samples)) if comm_health_samples else 1.0
            sec_metrics.log(t_s, ids_out, pos_true, references_cmd, effective_offsets)
            agent0_broadcast_seen_by_1 = neighbor_states.get(1, {}).get(0) if num > 1 else None
            agent0_clean_seen_by_1 = neighbor_states_clean.get(1, {}).get(0) if num > 1 else None
            attacked_ts_10 = (
                float(agent0_broadcast_seen_by_1.get("timestamp", float("nan")))
                if agent0_broadcast_seen_by_1 is not None
                else float("nan")
            )
            clean_ts_10 = (
                float(agent0_clean_seen_by_1.get("timestamp", float("nan")))
                if agent0_clean_seen_by_1 is not None
                else float("nan")
            )
            attacked_pos_10 = (
                np.asarray(agent0_broadcast_seen_by_1.get("pos"), dtype=float)
                if agent0_broadcast_seen_by_1 is not None
                else np.full(3, np.nan, dtype=float)
            )
            clean_pos_10 = (
                np.asarray(agent0_clean_seen_by_1.get("pos"), dtype=float)
                if agent0_clean_seen_by_1 is not None
                else np.full(3, np.nan, dtype=float)
            )
            twin_pos_0 = twin.get_state(0)[:3]
            omega = getattr(trust_mpc, "_last_omega", np.ones(num, dtype=float) / max(num, 1))
            last_comm_quality = comm_health
            last_mean_error = mean_err_nominal
            rows.append(
                {
                    "t": t_s,
                    "scenario": scenario,
                    "controller": controller_name,
                    "controller_label": controller_label(controller_name, ablation),
                    "ablation": ablation,
                    "seed": seed,
                    "mean_err_nominal_m": mean_err_nominal,
                    "max_err_nominal_m": max_err_nominal,
                    "mean_err_cmd_m": mean_err_cmd,
                    "max_err_cmd_m": max_err_cmd,
                    "formation_error_m": form_err_nominal,
                    "max_formation_deformation_m": max_pairwise_spacing_error(pos_true, offsets_nominal),
                    "spacing_violations": spacing_violation_count(pos_true, min_spacing_m),
                    "connectivity_rate": connectivity,
                    "comm_health_rate": comm_health,
                    "neighbor_count_mean": mean_neighbor_count(pos_true, comm_range_m),
                    "group_drift_m": group_drift_m,
                    "sensor_residual_m": float(np.mean(raw_sensor_residuals)),
                    "innovation_norm_m": float(np.mean(innovation_norms)),
                    "nis": float(np.mean(nis_values)),
                    "measurement_accept_rate": float(np.mean(accept_flags)),
                    "measurement_weight": float(np.mean(measurement_weights)),
                    "bias_estimate_norm_m": float(np.mean(bias_norms)),
                    "diagnosed_fault": diagnosis.active_fault,
                    "sensor_confidence": diagnosis.confidences.sensor,
                    "wind_confidence": diagnosis.confidences.wind,
                    "comm_confidence": diagnosis.confidences.comm,
                    "supervisor_mode": sup.detected_mode,
                    "speed_scale": sup.speed_scale,
                    "formation_scale": sup.formation_scale,
                    "consensus_scale": sup.consensus_scale,
                    "control_effort": control_effort,
                    "supervisor_compensation_norm": compensation_norm,
                    "disturbance_estimate_norm": float(np.mean([np.linalg.norm(disturbance_est[j]) for j in range(num)])),
                    "integral_norm": float(np.mean(integral_norms)) if integral_norms else 0.0,
                    "saturation_rate": float(np.mean(saturation_flags)) if saturation_flags else 0.0,
                    "measurement_dropout_count": measurements_missing,
                    "ids_chi2_swarm": ids_out["chi2_swarm"],
                    "ids_anomaly_flag": float(ids_out["anomaly_flag_swarm"]),
                    "ids_k_hat_raw": int(ids_out["k_hat_raw"]),
                    "ids_k_hat": int(ids_out["k_hat"]),
                    "ids_ttd_s": float("nan") if ids_out["ttd"] is None else float(ids_out["ttd"]),
                    "ids_cusum_wind": float(ids_out["cusum_g"].get(AttackClass.H1_WIND, 0.0)),
                    "ids_cusum_jamming": float(ids_out["cusum_g"].get(AttackClass.H2_JAMMING, 0.0)),
                    "ids_cusum_spoofing": float(ids_out["cusum_g"].get(AttackClass.H3_SPOOF, 0.0)),
                    "ids_cusum_replay": float(ids_out["cusum_g"].get(AttackClass.H4_REPLAY, 0.0)),
                    "security_true_attack": int(ATTACK_CLASS_BY_SCENARIO.get(scenario, AttackClass.H0_NONE)),
                    "debug_agent0_meas_err_m": float(np.linalg.norm(measured_pos[0] - pos_true[0])),
                    "debug_agent0_twin_err_m": float(np.linalg.norm(twin_pos_0 - pos_true[0])),
                    "debug_qcomm_min": float(min(qcomm.values())) if qcomm else 1.0,
                    "debug_qcomm_mean": float(np.mean(list(qcomm.values()))) if qcomm else 1.0,
                    "debug_target_k3": -1 if ids_out["target_k3"] is None else int(ids_out["target_k3"]),
                    "debug_agent0_broadcast_offset_m": float(np.linalg.norm((attacked_pos_10 - pos_true[0])[:2])),
                    "debug_replay_clean_ts_10": clean_ts_10,
                    "debug_replay_attacked_ts_10": attacked_ts_10,
                    "debug_replay_staleness_10_s": float(t_s - attacked_ts_10) if np.isfinite(attacked_ts_10) else float("nan"),
                    "debug_replay_pos_delta_10_m": float(np.linalg.norm((clean_pos_10 - attacked_pos_10)[:2])),
                    "debug_mpc_mode": getattr(trust_mpc, "_last_mode", "nominal"),
                    "debug_delta_norm_0": delta_norms.get(0, 0.0),
                    "debug_delta_norm_1": delta_norms.get(1, 0.0),
                    "debug_delta_norm_2": delta_norms.get(2, 0.0),
                    "debug_delta_norm_3": delta_norms.get(3, 0.0),
                    "debug_omega_0": float(omega[0]) if len(omega) > 0 else float("nan"),
                    "debug_omega_1": float(omega[1]) if len(omega) > 1 else float("nan"),
                    "debug_omega_2": float(omega[2]) if len(omega) > 2 else float("nan"),
                    "debug_omega_3": float(omega[3]) if len(omega) > 3 else float("nan"),
                }
            )
            last_disturbance_norm = float(np.mean([np.linalg.norm(disturbance_est[j][:2]) for j in range(num)]))

        swarm.step(target_pos_cmd, active_wind, target_vel_cmd)

    df = pd.DataFrame(rows)
    pre_mean = window_mean(df, float(pre0), float(pre1), "mean_err_nominal_m")
    fault_mean = window_mean(df, float(fault0), float(fault1), "mean_err_nominal_m")
    post_mean = window_mean(df, float(post0), float(post1), "mean_err_nominal_m")
    threshold = float(cfg["analysis"]["recovery_threshold_mult"]) * pre_mean
    settling_threshold = float(cfg["analysis"]["settling_threshold_mult"]) * post_mean
    label = controller_label(controller_name, ablation)
    security_summary = sec_metrics.summary()
    summary = {
        "scenario": scenario,
        "controller": controller_name,
        "controller_label": label,
        "ablation": ablation,
        "seed": seed,
        "pre_fault_error_m": pre_mean,
        "fault_error_m": fault_mean,
        "post_fault_error_m": post_mean,
        "degradation_pct": degradation_pct(pre_mean, fault_mean),
        "post_degradation_pct": degradation_pct(pre_mean, post_mean),
        "peak_error_spike_m": float(df[(df["t"] >= fault0) & (df["t"] < fault1)]["max_err_nominal_m"].max()),
        "recovery_time_s": recovery_time(
            df["t"].to_numpy(),
            df["mean_err_nominal_m"].to_numpy(),
            float(fault0),
            threshold,
            float(cfg["analysis"]["recovery_sustain_s"]),
        ),
        "settling_time_s": settling_time(
            df["t"].to_numpy(),
            df["mean_err_nominal_m"].to_numpy(),
            float(post0),
            settling_threshold,
            float(cfg["analysis"]["recovery_sustain_s"]),
        ),
        "rmse_m": rmse(df["mean_err_nominal_m"].to_numpy()),
        "max_formation_deformation_m": float(df["max_formation_deformation_m"].max()),
        "spacing_violation_count": int(df["spacing_violations"].sum()),
        "time_above_safety_s": time_above_threshold(
            df["mean_err_nominal_m"].to_numpy(),
            safety_error_threshold_m,
            env_handles.dt_ctrl,
        ),
        "control_effort_total": float(df["control_effort"].sum() * env_handles.dt_ctrl),
        "mean_connectivity": float(df["connectivity_rate"].mean()),
        "mean_comm_health": float(df["comm_health_rate"].mean()),
        "stable_run": int(float(df["mean_err_nominal_m"].max()) < 2.5 * safety_error_threshold_m),
        **security_summary,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "csv").mkdir(parents=True, exist_ok=True)
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(":", "_")
    df.to_csv(output_root / "csv" / f"{safe_label}_seed{seed}.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_root / "summary" / f"{safe_label}_seed{seed}.csv", index=False)
    return df, summary


def plot_run(df: pd.DataFrame, output_root: Path, label: str, seed: int) -> None:
    fig_dir = output_root / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(":", "_")
    plt.figure(figsize=(9, 4))
    plt.plot(df["t"], df["mean_err_nominal_m"], label="tracking error")
    plt.plot(df["t"], df["sensor_confidence"], label="sensor conf")
    plt.plot(df["t"], df["wind_confidence"], label="wind conf")
    plt.plot(df["t"], df["comm_confidence"], label="comm conf")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{safe_label}_seed{seed}_timeline.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    output_root = Path(args.output_root) / args.scenario
    df, _ = run_once(cfg, args.scenario, args.controller, int(args.seed), output_root, ablation=args.ablation)
    plot_run(df, output_root, controller_label(args.controller, args.ablation), int(args.seed))


if __name__ == "__main__":
    main()
