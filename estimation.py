from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EstimatorSnapshot:
    predicted_pos: np.ndarray
    predicted_vel: np.ndarray
    filtered_pos: np.ndarray
    filtered_vel: np.ndarray
    innovation: np.ndarray
    innovation_norm: float
    measurement_weight: float
    measurement_accepted: bool
    nis: float
    bias_estimate_xy: np.ndarray


class ConstantVelocityEstimator:
    """Constant-velocity estimator with innovation gating and explicit bias tracking."""

    def __init__(
        self,
        dt: float,
        base_gain: float,
        min_gain: float,
        max_gain: float,
        outlier_gate_m: float,
        nis_gate: float,
        predicted_var_xy: float,
        measurement_var_xy: float,
        bias_gain: float,
        bias_decay: float,
        bias_enable_conf: float,
        bias_limit_m: float,
    ):
        self.dt = float(dt)
        self.base_gain = float(base_gain)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.outlier_gate_m = float(outlier_gate_m)
        self.nis_gate = float(nis_gate)
        self.predicted_var_xy = float(predicted_var_xy)
        self.measurement_var_xy = float(measurement_var_xy)
        self.bias_gain = float(bias_gain)
        self.bias_decay = float(bias_decay)
        self.bias_enable_conf = float(bias_enable_conf)
        self.bias_limit_m = float(bias_limit_m)
        self.pos = np.zeros(3, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.bias_xy = np.zeros(2, dtype=float)
        self.initialized = False

    def reset(self, pos: np.ndarray, vel: np.ndarray | None = None) -> None:
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.zeros(3, dtype=float) if vel is None else np.asarray(vel, dtype=float).copy()
        self.bias_xy = np.zeros(2, dtype=float)
        self.initialized = True

    def step(
        self,
        measurement: np.ndarray | None,
        sensor_confidence: float,
        trust_scale: float = 1.0,
        control_input: np.ndarray | None = None,
    ) -> EstimatorSnapshot:
        if not self.initialized:
            init = np.zeros(3, dtype=float) if measurement is None else np.asarray(measurement, dtype=float)
            self.reset(init)

        u = np.zeros(3, dtype=float) if control_input is None else np.asarray(control_input, dtype=float)
        predicted_pos = self.pos + self.dt * self.vel + 0.5 * (self.dt**2) * u
        predicted_vel = self.vel + self.dt * u
        if measurement is None:
            innovation = np.zeros(3, dtype=float)
            meas_gain = 0.0
            measurement_accepted = False
            nis = 0.0
            self.bias_xy *= max(0.0, 1.0 - self.bias_decay)
        else:
            corrected_measurement = np.asarray(measurement, dtype=float).copy()
            corrected_measurement[:2] -= self.bias_xy
            innovation = corrected_measurement - predicted_pos
            innovation_norm = float(np.linalg.norm(innovation[:2]))
            nis = float(np.sum(np.square(innovation[:2])) / max(self.predicted_var_xy + self.measurement_var_xy, 1e-9))
            measurement_accepted = bool(innovation_norm <= self.outlier_gate_m and nis <= self.nis_gate)
            raw_bias_residual = np.asarray(measurement, dtype=float)[:2] - predicted_pos[:2] - self.bias_xy
            if sensor_confidence >= self.bias_enable_conf and measurement_accepted:
                self.bias_xy += self.bias_gain * raw_bias_residual
                self.bias_xy = np.clip(self.bias_xy, -self.bias_limit_m, self.bias_limit_m)
            else:
                self.bias_xy *= max(0.0, 1.0 - self.bias_decay)
            if not measurement_accepted:
                meas_gain = 0.0
            else:
                meas_gain = self.base_gain * trust_scale * (1.0 - sensor_confidence)
                meas_gain = float(np.clip(meas_gain, self.min_gain, self.max_gain))
        filtered_pos = predicted_pos + meas_gain * innovation
        filtered_vel = predicted_vel + (meas_gain / max(self.dt, 1e-9)) * innovation
        self.pos = filtered_pos
        self.vel = filtered_vel
        return EstimatorSnapshot(
            predicted_pos=predicted_pos,
            predicted_vel=predicted_vel,
            filtered_pos=filtered_pos,
            filtered_vel=filtered_vel,
            innovation=innovation,
            innovation_norm=float(np.linalg.norm(innovation[:2])),
            measurement_weight=meas_gain,
            measurement_accepted=measurement_accepted,
            nis=nis,
            bias_estimate_xy=self.bias_xy.copy(),
        )
