from __future__ import annotations

from typing import Any

import numpy as np


def formation_offsets(cfg: dict[str, Any], n: int) -> np.ndarray:
    formation_cfg = cfg["formation"]
    spacing = float(formation_cfg["spacing_m"])
    if formation_cfg["type"].lower() == "square" and n == 4:
        return np.array(
            [
                [-spacing / 2.0, -spacing / 2.0, 0.0],
                [-spacing / 2.0, spacing / 2.0, 0.0],
                [spacing / 2.0, -spacing / 2.0, 0.0],
                [spacing / 2.0, spacing / 2.0, 0.0],
            ],
            dtype=float,
        )
    offsets = np.zeros((n, 3), dtype=float)
    for idx in range(n):
        offsets[idx] = np.array([(idx - (n - 1) / 2.0) * spacing, 0.0, 0.0], dtype=float)
    return offsets


def reference_state(cfg: dict[str, Any], t_s: float) -> tuple[np.ndarray, np.ndarray]:
    traj_cfg = cfg["trajectory"]
    kind = str(traj_cfg["type"]).lower()
    radius = float(traj_cfg["radius_m"])
    altitude = float(traj_cfg["altitude_m"])
    period_s = float(traj_cfg["period_s"])
    center_x, center_y = traj_cfg["center_xy"]

    if kind != "circle":
        raise ValueError(f"Unsupported trajectory type for paper simulation: {kind}")

    omega = 2.0 * np.pi / period_s
    x = center_x + radius * np.cos(omega * t_s)
    y = center_y + radius * np.sin(omega * t_s)
    vx = -radius * omega * np.sin(omega * t_s)
    vy = radius * omega * np.cos(omega * t_s)
    return np.array([x, y, altitude], dtype=float), np.array([vx, vy, 0.0], dtype=float)
