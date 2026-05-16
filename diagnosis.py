from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaultConfidence:
    sensor: float
    wind: float
    comm: float


@dataclass
class DiagnosisSnapshot:
    confidences: FaultConfidence
    active_fault: str
    sensor_counter: int
    wind_counter: int
    comm_counter: int


class FaultDiagnoser:
    """Residual-based diagnosis with persistence to avoid chattering."""

    def __init__(self, cfg: dict):
        dcfg = cfg["diagnosis"]
        self.sensor_residual_threshold = float(dcfg["sensor_residual_threshold_m"])
        self.wind_residual_threshold = float(dcfg["wind_residual_threshold_m"])
        self.comm_quality_threshold = float(dcfg["comm_quality_threshold"])
        self.persistence_steps = int(dcfg["persistence_steps"])
        self.release_steps = int(dcfg["release_steps"])
        self._sensor_counter = 0
        self._wind_counter = 0
        self._comm_counter = 0
        self._sensor_release = 0
        self._wind_release = 0
        self._comm_release = 0

    def _update_counter(self, active: bool, counter: int, release: int) -> tuple[int, int]:
        if active:
            counter += 1
            release = 0
        else:
            release += 1
            if release >= self.release_steps:
                counter = max(0, counter - 1)
        return counter, release

    def step(
        self,
        sensor_residual_m: float,
        wind_residual_m: float,
        comm_quality: float,
    ) -> DiagnosisSnapshot:
        self._sensor_counter, self._sensor_release = self._update_counter(
            sensor_residual_m > self.sensor_residual_threshold,
            self._sensor_counter,
            self._sensor_release,
        )
        self._wind_counter, self._wind_release = self._update_counter(
            wind_residual_m > self.wind_residual_threshold,
            self._wind_counter,
            self._wind_release,
        )
        self._comm_counter, self._comm_release = self._update_counter(
            comm_quality < self.comm_quality_threshold,
            self._comm_counter,
            self._comm_release,
        )

        sensor_conf = float(np.clip(self._sensor_counter / max(self.persistence_steps, 1), 0.0, 1.0))
        wind_conf = float(np.clip(self._wind_counter / max(self.persistence_steps, 1), 0.0, 1.0))
        comm_conf = float(np.clip(self._comm_counter / max(self.persistence_steps, 1), 0.0, 1.0))

        confs = FaultConfidence(sensor=sensor_conf, wind=wind_conf, comm=comm_conf)
        active_fault = "nominal"
        if max(sensor_conf, wind_conf, comm_conf) > 0.0:
            active_fault = max(
                [("sensor", sensor_conf), ("wind", wind_conf), ("comm", comm_conf)],
                key=lambda x: x[1],
            )[0]
        return DiagnosisSnapshot(
            confidences=confs,
            active_fault=active_fault,
            sensor_counter=self._sensor_counter,
            wind_counter=self._wind_counter,
            comm_counter=self._comm_counter,
        )
