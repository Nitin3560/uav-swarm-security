"""trajectory_v2.py — Formation geometry for arbitrary N.

Extension 2 support: scalability study (N = 4, 8, 16)
------------------------------------------------------
The original formation_offsets() had a hardcoded `if n == 4` square branch.
This version adds:

  N=4  : 2×2 square      (original behaviour preserved)
  N=8  : two parallel rows of 4  (rectangular double-line)
  N=16 : 4×4 grid
  N=any: linear fallback  (original fallback)

All formations are centred at the swarm centroid so the reference trajectory
logic (which targets the centroid) is unchanged.

The reference_state() function is identical to the original.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def formation_offsets(cfg: dict[str, Any], n: int) -> np.ndarray:
    """Return (n, 3) array of formation offsets relative to swarm centroid.

    Supported formations:
      n=4  : 2×2 square
      n=8  : 2×4 double-line (2 rows, 4 columns)
      n=16 : 4×4 grid
      other: linear row (original fallback)
    """
    formation_cfg = cfg["formation"]
    spacing = float(formation_cfg["spacing_m"])

    if n == 4 and formation_cfg["type"].lower() == "square":
        # Original 2×2 square — identical to v1
        half = spacing / 2.0
        return np.array(
            [
                [-half, -half, 0.0],
                [-half,  half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
            ],
            dtype=float,
        )

    if n == 8:
        # 2 rows × 4 columns centred at origin
        # Row separation = spacing, column separation = spacing
        offsets = np.zeros((8, 3), dtype=float)
        cols = 4
        rows = 2
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                offsets[idx, 0] = (c - (cols - 1) / 2.0) * spacing  # x
                offsets[idx, 1] = (r - (rows - 1) / 2.0) * spacing  # y
        return offsets

    if n == 16:
        # 4×4 grid centred at origin
        offsets = np.zeros((16, 3), dtype=float)
        side = 4
        for r in range(side):
            for c in range(side):
                idx = r * side + c
                offsets[idx, 0] = (c - (side - 1) / 2.0) * spacing
                offsets[idx, 1] = (r - (side - 1) / 2.0) * spacing
        return offsets

    # Linear fallback for arbitrary N (original behaviour)
    offsets = np.zeros((n, 3), dtype=float)
    for idx in range(n):
        offsets[idx] = np.array(
            [(idx - (n - 1) / 2.0) * spacing, 0.0, 0.0],
            dtype=float,
        )
    return offsets


def reference_state(cfg: dict[str, Any], t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, velocity) for the swarm centroid reference trajectory.

    Unchanged from v1 — circular trajectory only.
    """
    traj_cfg = cfg["trajectory"]
    kind = str(traj_cfg["type"]).lower()
    radius = float(traj_cfg["radius_m"])
    altitude = float(traj_cfg["altitude_m"])
    period_s = float(traj_cfg["period_s"])
    center_x, center_y = traj_cfg["center_xy"]

    if kind != "circle":
        raise ValueError(f"Unsupported trajectory type: {kind}")

    omega = 2.0 * np.pi / period_s
    x = center_x + radius * np.cos(omega * t_s)
    y = center_y + radius * np.sin(omega * t_s)
    vx = -radius * omega * np.sin(omega * t_s)
    vy = radius * omega * np.cos(omega * t_s)
    return np.array([x, y, altitude], dtype=float), np.array([vx, vy, 0.0], dtype=float)
