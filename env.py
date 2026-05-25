from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from paper_sim.trajectory_v2 import formation_offsets, reference_state


class PaperEnv:
    """PyBullet-Drones backend with the interface expected by run_study.py."""

    def __init__(self, cfg: dict[str, Any], gui: bool = False):
        self.cfg = cfg
        sim_cfg = cfg["sim"]
        self.n = int(sim_cfg["num_drones"])
        self.sim_hz = float(sim_cfg["freq_hz"])
        self.ctrl_hz = float(sim_cfg["ctrl_hz"])
        self.dt_sim = 1.0 / self.sim_hz
        self.dt_ctrl = 1.0 / self.ctrl_hz

        p_ref0, _ = reference_state(cfg, 0.0)
        offsets0 = formation_offsets(cfg, self.n)
        init_pos = (p_ref0 + offsets0).astype(float)

        # run_study.py owns the 240 Hz outer loop and updates commands every
        # ctrl_hz interval, so CtrlAviary is stepped at pyb_freq here.
        self._aviary = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.n,
            initial_xyzs=init_pos,
            initial_rpys=np.zeros((self.n, 3), dtype=float),
            physics=Physics.PYB,
            pyb_freq=int(self.sim_hz),
            ctrl_freq=int(self.sim_hz),
            gui=gui,
            record=False,
            user_debug_gui=False,
        )
        reset_out = self._aviary.reset(seed=int(sim_cfg.get("seed", 1)))
        self._obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        self.positions = init_pos.copy()
        self.velocities = np.zeros((self.n, 3), dtype=float)
        self._target_pos = init_pos.copy()
        self._target_vel = np.zeros((self.n, 3), dtype=float)
        self._mass = float(self._aviary.M)
        self._client = int(self._aviary.CLIENT)
        self._controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.n)]
        self._last_rpm = np.ones((self.n, 4), dtype=float) * float(self._aviary.HOVER_RPM)
        self._last_pos_error = np.zeros((self.n, 3), dtype=float)
        self._last_yaw_error = np.zeros(self.n, dtype=float)
        self._update_state()

        # Backward compatibility with the old PaperEnv dataclass wrapper.
        self.env = self

    def get_pos(self, i: int) -> np.ndarray:
        return self.positions[i].copy()

    def get_vel(self, i: int) -> np.ndarray:
        return self.velocities[i].copy()

    def get_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_velocities(self) -> np.ndarray:
        return self.velocities.copy()

    def set_accel_cmd(self, i: int, accel: np.ndarray) -> None:
        # Backward-compatible alias: older callers supplied acceleration
        # commands. The PyBullet-Drones backend now uses position targets.
        self._target_vel[i] = np.asarray(accel, dtype=float)

    def set_target(self, i: int, target_pos: np.ndarray, target_vel: np.ndarray | None = None) -> None:
        self._target_pos[i] = np.asarray(target_pos, dtype=float)
        if target_vel is None:
            self._target_vel[i] = 0.0
        else:
            self._target_vel[i] = np.asarray(target_vel, dtype=float)

    def step_simulation(self) -> None:
        self.step_targets(self._target_pos, self._target_vel, np.zeros_like(self._target_pos))

    def step(self, target_pos: np.ndarray, wind_accel: np.ndarray, target_vel: np.ndarray | None = None) -> None:
        self.step_targets(target_pos, target_vel, wind_accel)

    def step_targets(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray | None,
        wind_accel: np.ndarray,
    ) -> None:
        self._target_pos = np.asarray(target_pos, dtype=float).copy()
        if target_vel is None:
            self._target_vel = np.zeros_like(self._target_pos)
        else:
            self._target_vel = np.asarray(target_vel, dtype=float).copy()
        wind = np.asarray(wind_accel, dtype=float)
        action = np.zeros((self.n, 4), dtype=float)
        for i in range(self.n):
            rpm, pos_e, yaw_e = self._controllers[i].computeControlFromState(
                control_timestep=self.dt_sim,
                state=self._state(i),
                target_pos=self._target_pos[i],
                target_vel=self._target_vel[i],
            )
            action[i] = np.asarray(rpm, dtype=float)
            self._last_pos_error[i] = np.asarray(pos_e, dtype=float)
            self._last_yaw_error[i] = float(yaw_e)

            force = self._mass * wind[i]
            if np.linalg.norm(force) > 0.0:
                p.applyExternalForce(
                    objectUniqueId=self._aviary.DRONE_IDS[i],
                    linkIndex=-1,
                    forceObj=force.tolist(),
                    posObj=self.positions[i].tolist(),
                    flags=p.WORLD_FRAME,
                    physicsClientId=self._client,
                )

        self._last_rpm = action.copy()
        out = self._aviary.step(action)
        self._obs = out[0]
        self._update_state()

    def close(self) -> None:
        self._aviary.close()

    def _update_state(self) -> None:
        obs = self._obs
        for i in range(self.n):
            obs_i = obs[str(i)] if isinstance(obs, dict) else obs[i]
            self.positions[i] = np.asarray(obs_i[0:3], dtype=float)
            self.velocities[i] = np.asarray(obs_i[10:13], dtype=float)

    def _state(self, i: int) -> np.ndarray:
        obs = self._obs
        return np.asarray(obs[str(i)] if isinstance(obs, dict) else obs[i], dtype=float)

    def _bounded_accel(self, accels: np.ndarray) -> np.ndarray:
        bounded = np.asarray(accels, dtype=float).copy()
        accel_limit_xy = float(self.cfg["sim"].get("accel_limit_xy", 4.0))
        accel_limit_z = float(self.cfg["sim"].get("accel_limit_z", 2.0))
        for i in range(self.n):
            xy_norm = float(np.linalg.norm(bounded[i, :2]))
            if xy_norm > accel_limit_xy:
                bounded[i, :2] *= accel_limit_xy / (xy_norm + 1e-9)
            bounded[i, 2] = float(np.clip(bounded[i, 2], -accel_limit_z, accel_limit_z))
        return bounded


def make_env(cfg: dict[str, Any], gui: bool = False) -> PaperEnv:
    return PaperEnv(cfg, gui=gui)
