from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from paper_sim.trajectory import formation_offsets, reference_state


@dataclass
class SwarmDynamics:
    positions: np.ndarray
    velocities: np.ndarray
    dt: float
    drag_coeff: float
    accel_limit_xy: float
    accel_limit_z: float
    speed_limit_xy: float
    speed_limit_z: float

    def step(self, accel_cmd: np.ndarray, wind_accel: np.ndarray) -> None:
        accel = np.asarray(accel_cmd, dtype=float) + np.asarray(wind_accel, dtype=float)
        accel = accel.copy()

        xy_norm = np.linalg.norm(accel[:, :2], axis=1)
        over_xy = xy_norm > self.accel_limit_xy
        if np.any(over_xy):
            accel[over_xy, :2] *= (self.accel_limit_xy / (xy_norm[over_xy] + 1e-9))[:, None]
        accel[:, 2] = np.clip(accel[:, 2], -self.accel_limit_z, self.accel_limit_z)

        accel -= self.drag_coeff * self.velocities
        self.velocities += self.dt * accel

        vel_xy_norm = np.linalg.norm(self.velocities[:, :2], axis=1)
        over_speed_xy = vel_xy_norm > self.speed_limit_xy
        if np.any(over_speed_xy):
            self.velocities[over_speed_xy, :2] *= (
                self.speed_limit_xy / (vel_xy_norm[over_speed_xy] + 1e-9)
            )[:, None]
        self.velocities[:, 2] = np.clip(self.velocities[:, 2], -self.speed_limit_z, self.speed_limit_z)

        self.positions += self.dt * self.velocities


@dataclass
class PaperEnv:
    env: SwarmDynamics
    dt_sim: float
    dt_ctrl: float


def make_env(cfg: dict[str, Any]) -> PaperEnv:
    sim_cfg = cfg["sim"]
    num = int(sim_cfg["num_drones"])
    dt_sim = 1.0 / float(sim_cfg["freq_hz"])
    dt_ctrl = 1.0 / float(sim_cfg["ctrl_hz"])

    p_ref0, _ = reference_state(cfg, 0.0)
    offsets0 = formation_offsets(cfg, num)
    positions = p_ref0 + offsets0
    velocities = np.zeros((num, 3), dtype=float)

    dynamics = SwarmDynamics(
        positions=positions.astype(float),
        velocities=velocities,
        dt=dt_sim,
        drag_coeff=float(sim_cfg.get("drag_coeff", 0.55)),
        accel_limit_xy=float(sim_cfg.get("accel_limit_xy", 4.0)),
        accel_limit_z=float(sim_cfg.get("accel_limit_z", 2.0)),
        speed_limit_xy=float(sim_cfg.get("speed_limit_xy", 4.0)),
        speed_limit_z=float(sim_cfg.get("speed_limit_z", 1.5)),
    )
    return PaperEnv(env=dynamics, dt_sim=dt_sim, dt_ctrl=dt_ctrl)
