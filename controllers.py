from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SupervisorCommand:
    speed_scale: float = 1.0
    formation_scale: float = 1.0
    consensus_scale: float = 1.0
    connectivity_bias_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    smoothing_alpha: float = 1.0
    damping_gain: float = 1.0
    detected_mode: str = "nominal"


class PositionPID:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["pid"]
        self.ctrl_hz = float(cfg["sim"]["ctrl_hz"])
        self.int_err: dict[int, np.ndarray] = {}
        self.last_int_norm: dict[int, float] = {}
        self.last_saturation: dict[int, int] = {}
        self.accel_limit_xy = float(cfg["sim"].get("accel_limit_xy", 4.0))
        self.accel_limit_z = float(cfg["sim"].get("accel_limit_z", 2.0))
        self.antiwindup_gain = float(self.cfg.get("antiwindup_gain", 0.25))

    def reset(self, num_drones: int) -> None:
        self.int_err = {idx: np.zeros(3, dtype=float) for idx in range(num_drones)}
        self.last_int_norm = {idx: 0.0 for idx in range(num_drones)}
        self.last_saturation = {idx: 0 for idx in range(num_drones)}

    def compute_accel(
        self,
        drone_id: int,
        measured_pos: np.ndarray,
        measured_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        extra_damping: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        dt = 1.0 / self.ctrl_hz
        err = np.asarray(target_pos, dtype=float) - np.asarray(measured_pos, dtype=float)
        vel_err = np.asarray(target_vel, dtype=float) - np.asarray(measured_vel, dtype=float)

        self.int_err[drone_id] += err * dt
        self.int_err[drone_id][:2] = np.clip(
            self.int_err[drone_id][:2],
            -float(self.cfg["integral_limit_xy"]),
            float(self.cfg["integral_limit_xy"]),
        )
        self.int_err[drone_id][2] = float(
            np.clip(self.int_err[drone_id][2], -float(self.cfg["integral_limit_z"]), float(self.cfg["integral_limit_z"]))
        )

        accel_unsat = np.zeros(3, dtype=float)
        accel_unsat[:2] = (
            float(self.cfg["kp_xy"]) * err[:2]
            + float(self.cfg["ki_xy"]) * self.int_err[drone_id][:2]
            + float(self.cfg["kd_xy"]) * vel_err[:2] * extra_damping
        )
        accel_unsat[2] = (
            float(self.cfg["kp_z"]) * err[2]
            + float(self.cfg["ki_z"]) * self.int_err[drone_id][2]
            + float(self.cfg["kd_z"]) * vel_err[2]
        )
        accel = accel_unsat.copy()
        xy_norm = np.linalg.norm(accel[:2])
        if xy_norm > self.accel_limit_xy:
            accel[:2] *= self.accel_limit_xy / (xy_norm + 1e-9)
        accel[2] = float(np.clip(accel[2], -self.accel_limit_z, self.accel_limit_z))

        sat_error = accel - accel_unsat
        self.int_err[drone_id][:2] += self.antiwindup_gain * sat_error[:2] * dt
        self.int_err[drone_id][2] += self.antiwindup_gain * sat_error[2] * dt
        self.int_err[drone_id][:2] = np.clip(
            self.int_err[drone_id][:2],
            -float(self.cfg["integral_limit_xy"]),
            float(self.cfg["integral_limit_xy"]),
        )
        self.int_err[drone_id][2] = float(
            np.clip(self.int_err[drone_id][2], -float(self.cfg["integral_limit_z"]), float(self.cfg["integral_limit_z"]))
        )
        sat_flag = int(np.linalg.norm(sat_error) > 1e-9)
        self.last_saturation[drone_id] = sat_flag
        self.last_int_norm[drone_id] = float(np.linalg.norm(self.int_err[drone_id]))
        debug = {
            "integral_norm": self.last_int_norm[drone_id],
            "saturation_flag": float(sat_flag),
            "unsat_minus_sat_norm": float(np.linalg.norm(sat_error)),
        }
        return accel, debug


class GenericSupervisor:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["generic_supervisor"]

    def step(self, mean_error_m: float, formation_error_m: float) -> SupervisorCommand:
        cmd = SupervisorCommand()
        if mean_error_m > float(self.cfg["error_trigger_m"]) or formation_error_m > float(self.cfg["error_trigger_m"]):
            cmd.speed_scale = float(self.cfg["speed_scale_active"])
            cmd.formation_scale = float(self.cfg["formation_scale_active"])
            cmd.smoothing_alpha = float(self.cfg["smoothing_alpha_active"])
            cmd.detected_mode = "generic_reactive"
        return cmd


class FailureAwareSupervisor:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["failure_aware"]
        self._last_cmd = SupervisorCommand()

    def step(
        self,
        diagnosis: Any,
        group_error_xy: np.ndarray,
        ablation: str = "full",
    ) -> SupervisorCommand:
        mode = "nominal" if ablation == "no_diagnosis" else diagnosis.active_fault
        cmd = SupervisorCommand(detected_mode=mode)
        sensor_conf = 0.0 if ablation == "no_diagnosis" else float(diagnosis.confidences.sensor)
        wind_conf = 0.0 if ablation == "no_diagnosis" else float(diagnosis.confidences.wind)
        comm_conf = 0.0 if ablation == "no_diagnosis" else float(diagnosis.confidences.comm)
        if mode == "wind":
            if ablation != "no_wind_comp":
                cmd.speed_scale = 1.0 - (1.0 - float(self.cfg["wind_speed_scale"])) * wind_conf
                cmd.damping_gain = 1.0 + (float(self.cfg["wind_damping_gain"]) - 1.0) * wind_conf
                cmd.connectivity_bias_xy = -np.asarray(group_error_xy, dtype=float) * float(self.cfg["wind_bias_gain"]) * wind_conf
        elif mode == "sensor":
            if ablation != "no_filter":
                cmd.smoothing_alpha = 1.0 - (1.0 - float(self.cfg["sensor_smoothing_alpha"])) * sensor_conf
            cmd.speed_scale = 1.0 - (1.0 - float(self.cfg["sensor_speed_scale"])) * sensor_conf
            cmd.consensus_scale = 1.0 - (1.0 - float(self.cfg["sensor_consensus_scale"])) * sensor_conf
            cmd.damping_gain = 1.0 + (float(self.cfg["sensor_damping_gain"]) - 1.0) * sensor_conf
        elif mode == "comm":
            cmd.speed_scale = float(self.cfg["comm_speed_scale"])
            cmd.formation_scale = float(self.cfg["comm_formation_scale"])
            if ablation != "no_comm_fallback":
                cmd.consensus_scale = 1.0 - (1.0 - float(self.cfg["comm_consensus_scale"])) * comm_conf
                group_error = np.asarray(group_error_xy, dtype=float)
                group_norm = np.linalg.norm(group_error)
                if group_norm > 1e-9:
                    cmd.connectivity_bias_xy = (
                        -group_error / group_norm * float(self.cfg["comm_connectivity_bias_m"]) * comm_conf
                    )
        if mode == "nominal" and ablation != "no_recovery_schedule":
            blend = float(self.cfg["recovery_blend_rate"])
            cmd.speed_scale = blend * cmd.speed_scale + (1.0 - blend) * self._last_cmd.speed_scale
            cmd.formation_scale = blend * cmd.formation_scale + (1.0 - blend) * self._last_cmd.formation_scale
            cmd.consensus_scale = blend * cmd.consensus_scale + (1.0 - blend) * self._last_cmd.consensus_scale
            cmd.smoothing_alpha = blend * cmd.smoothing_alpha + (1.0 - blend) * self._last_cmd.smoothing_alpha
            cmd.damping_gain = blend * cmd.damping_gain + (1.0 - blend) * self._last_cmd.damping_gain
            cmd.connectivity_bias_xy = (
                blend * cmd.connectivity_bias_xy + (1.0 - blend) * self._last_cmd.connectivity_bias_xy
            )
        self._last_cmd = cmd
        return cmd
