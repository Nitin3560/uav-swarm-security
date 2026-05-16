from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScenarioFaultState:
    sensor_bias_xy: dict[int, np.ndarray]
    sensor_drift_xy: dict[int, np.ndarray]
    frozen_measurement: dict[int, np.ndarray | None]


class FaultModel:
    def __init__(self, cfg: dict[str, Any], scenario: str, dt_ctrl: float, seed: int, num_drones: int):
        self.cfg = cfg
        self.analysis_cfg = cfg["analysis"]
        self.fault_cfg = cfg["faults"][scenario]
        self.dt_ctrl = float(dt_ctrl)
        self.rng = np.random.default_rng(seed)
        base_sensor_bias = np.asarray(self.fault_cfg.get("sensor_bias_xy_m", [0.0, 0.0]), dtype=float)
        self.state = ScenarioFaultState(
            sensor_bias_xy={
                i: base_sensor_bias
                * self.rng.uniform(0.7, 1.3, size=2)
                * self.rng.choice(np.array([-1.0, 1.0]), size=2)
                for i in range(num_drones)
            },
            sensor_drift_xy={i: np.zeros(2, dtype=float) for i in range(num_drones)},
            frozen_measurement={i: None for i in range(num_drones)},
        )
        self.delay_steps = int(self.fault_cfg.get("packet_delay_steps", 0))
        self.packet_drop_prob = float(self.fault_cfg.get("packet_drop_prob", 0.0))
        self.comm_mode = str(self.fault_cfg.get("comm_mode", "nominal")).lower()
        self.sensor_mode = str(self.fault_cfg.get("sensor_mode", "nominal")).lower()
        self.wind_mode = str(self.fault_cfg.get("wind_mode", "constant")).lower()
        self.isolate_agent_id = int(self.fault_cfg.get("isolate_agent_id", -1))
        self._comm_buffers: dict[tuple[int, int], deque[np.ndarray]] = {}

    @property
    def fault_window(self) -> tuple[float, float]:
        start_s, end_s = self.analysis_cfg["fault_window_s"]
        return float(start_s), float(end_s)

    def fault_active(self, t_s: float) -> bool:
        start_s, end_s = self.fault_window
        return start_s <= t_s < end_s

    def wind_accel(self, num_drones: int, t_s: float) -> np.ndarray:
        accel = np.zeros((num_drones, 3), dtype=float)
        if not self.fault_active(t_s):
            return accel
        base = np.asarray(self.fault_cfg.get("wind_force_n", [0.0, 0.0, 0.0]), dtype=float)
        gust = np.asarray(self.fault_cfg.get("wind_gust_force_n", [0.0, 0.0, 0.0]), dtype=float)
        interval_s = float(self.fault_cfg.get("wind_gust_interval_s", 0.0))
        varying_std = float(self.fault_cfg.get("wind_variation_std_n", 0.0))
        gust_now = np.zeros(3, dtype=float)
        if self.wind_mode in {"gust", "mixed"} and interval_s > 0.0:
            phase = (t_s - self.fault_window[0]) % interval_s
            if phase < min(0.4, interval_s / 2.0):
                gust_now = gust
        varying = np.zeros((num_drones, 3), dtype=float)
        if self.wind_mode in {"varying", "mixed"} and varying_std > 0.0:
            phase = t_s - self.fault_window[0]
            varying[:] = np.array(
                [
                    varying_std * np.sin(0.35 * phase),
                    varying_std * np.cos(0.2 * phase),
                    0.0,
                ]
            )
        accel[:] = base + gust_now + varying
        accel += self.rng.normal(0.0, 0.03, size=accel.shape)
        return accel

    def measurement(self, true_pos: np.ndarray, t_s: float, drone_id: int) -> np.ndarray:
        measured = np.asarray(true_pos, dtype=float).copy()
        if self.fault_active(t_s):
            noise_std = float(self.fault_cfg.get("sensor_noise_std_m", 0.0))
            drift_std = float(self.fault_cfg.get("sensor_drift_std_mps", 0.0))
            spike_std = float(self.fault_cfg.get("sensor_spike_std_m", 0.0))
            freeze_prob = float(self.fault_cfg.get("sensor_freeze_prob", 0.0))
            dropout_prob = float(self.fault_cfg.get("sensor_dropout_prob", 0.0))
            if dropout_prob > 0.0 and self.rng.uniform() < dropout_prob:
                return np.full(3, np.nan, dtype=float)
            if self.sensor_mode in {"drift", "mixed"}:
                self.state.sensor_drift_xy[drone_id] += self.rng.normal(0.0, drift_std * self.dt_ctrl, size=2)
            if self.sensor_mode in {"bias", "mixed", "frozen"}:
                measured[:2] += self.state.sensor_bias_xy[drone_id]
            measured[:2] += self.state.sensor_drift_xy[drone_id]
            measured[:2] += self.rng.normal(0.0, noise_std, size=2)
            if self.sensor_mode in {"spike", "mixed"} and spike_std > 0.0 and self.rng.uniform() < 0.05:
                measured[:2] += self.rng.normal(0.0, spike_std, size=2)
            if self.sensor_mode in {"frozen", "mixed"} and freeze_prob > 0.0 and self.rng.uniform() < freeze_prob:
                if self.state.frozen_measurement[drone_id] is None:
                    self.state.frozen_measurement[drone_id] = measured.copy()
                return self.state.frozen_measurement[drone_id].copy()
            self.state.frozen_measurement[drone_id] = None
        else:
            self.state.frozen_measurement[drone_id] = None
            measured[:2] += self.rng.normal(0.0, 0.02, size=2)
        return measured

    def comm_view(self, payload: np.ndarray, receiver_id: int, sender_id: int, t_s: float) -> np.ndarray | None:
        key = (receiver_id, sender_id)
        buf = self._comm_buffers.setdefault(key, deque())
        buf.append(np.asarray(payload, dtype=float).copy())
        if len(buf) > max(1, self.delay_steps + 1):
            buf.popleft()
        if not self.fault_active(t_s):
            return np.asarray(payload, dtype=float).copy()
        if self.comm_mode == "isolation" and (receiver_id == self.isolate_agent_id or sender_id == self.isolate_agent_id):
            return None
        if self.rng.uniform() < self.packet_drop_prob:
            return None
        idx = max(0, len(buf) - 1 - self.delay_steps)
        return np.asarray(list(buf)[idx], dtype=float)
