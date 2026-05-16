from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np


NeighborStates = dict[int, dict[int, dict[str, Any] | None]]

_replay_buffers: dict[int, deque[tuple[float, dict[int, dict[str, Any] | None]]]] = {}
_jam_active: dict[int, bool] = {}
_spoof_active: dict[int, bool] = {}
_replay_active: dict[int, bool] = {}


def reset_attack_state(n_agents: int = 4, replay_buffer_len: int = 300) -> None:
    """Reset module-level attack state at the start of each simulation run."""
    global _replay_buffers, _jam_active, _spoof_active, _replay_active
    _jam_active = {i: False for i in range(n_agents)}
    _spoof_active = {i: False for i in range(n_agents)}
    _replay_active = {i: False for i in range(n_agents)}
    _replay_buffers = {i: deque(maxlen=replay_buffer_len) for i in range(n_agents)}


def _copy_neighbor_states(neighbor_states: NeighborStates) -> NeighborStates:
    return {
        int(i): {int(j): (None if state is None else dict(state)) for j, state in neighbors.items()}
        for i, neighbors in neighbor_states.items()
    }


def inject_jamming(
    neighbor_states: NeighborStates,
    qcomm: dict[int, float],
    t: float,
    agent_ids: str | list[int] | range = "all",
    start_t: float = 20.0,
    end_t: float = 30.0,
    jam_power: float = 1.0,
    n_agents: int = 4,
) -> tuple[NeighborStates, dict[int, float]]:
    """Drop inter-agent messages for targeted receivers during the attack window."""
    if not (start_t <= t < end_t):
        return neighbor_states, qcomm

    ns = _copy_neighbor_states(neighbor_states)
    qc = dict(qcomm)
    ids = range(n_agents) if agent_ids == "all" else agent_ids
    rng = np.random.default_rng(seed=int(t * 1000) % (2**31 - 1))
    drop_prob = float(np.clip(jam_power, 0.0, 1.0))

    for i in ids:
        _jam_active[int(i)] = True
        suppressed: dict[int, dict[str, Any] | None] = {}
        for j, state in ns.get(int(i), {}).items():
            suppressed[j] = None if rng.random() < drop_prob else state
        ns[int(i)] = suppressed
        qc[int(i)] = 1.0 - drop_prob
    return ns, qc


def inject_spoofing(
    neighbor_states: NeighborStates,
    t: float,
    target_agent: int = 0,
    d_spoof: np.ndarray | None = None,
    start_t: float = 20.0,
    end_t: float = 30.0,
) -> NeighborStates:
    """Corrupt one agent's broadcast position without mutating onboard measurements."""
    if not (start_t <= t < end_t):
        return neighbor_states
    if d_spoof is None:
        d_spoof = np.array([1.5, 0.0, 0.0], dtype=float)
    ns = _copy_neighbor_states(neighbor_states)
    target = int(target_agent)
    offset = np.asarray(d_spoof, dtype=float)
    for receiver_id, neighbors in ns.items():
        if receiver_id == target:
            continue
        state = neighbors.get(target)
        if state is None:
            continue
        spoofed_state = dict(state)
        spoofed_state["pos"] = np.asarray(spoofed_state["pos"], dtype=float) + offset
        neighbors[target] = spoofed_state
    _spoof_active[int(target_agent)] = True
    return ns


def buffer_neighbor_states(neighbor_states: NeighborStates, t: float) -> None:
    """Store pre-attack neighbor messages so replay can inject stale packets later."""
    for i, neighbors in neighbor_states.items():
        if i in _replay_buffers:
            _replay_buffers[i].append((float(t), deepcopy(neighbors)))


def inject_replay(
    neighbor_states: NeighborStates,
    t: float,
    target_links: list[tuple[int, int]] | None = None,
    replay_delay: float = 5.0,
    start_t: float = 20.0,
    end_t: float = 30.0,
    n_agents: int = 4,
) -> NeighborStates:
    """Replace current messages with packets captured replay_delay seconds earlier."""
    if not (start_t <= t < end_t):
        return neighbor_states

    ns = _copy_neighbor_states(neighbor_states)
    target_set = None if target_links is None else {(int(i), int(j)) for i, j in target_links}
    for i in range(n_agents):
        buf = list(_replay_buffers.get(i, ()))
        if not buf:
            continue
        target_time = float(t) - float(replay_delay)
        stale_t, stale_neighbors = min(buf, key=lambda item: abs(item[0] - target_time))
        for j, state in stale_neighbors.items():
            if target_set is not None and (i, j) not in target_set:
                continue
            if state is None or j not in ns.get(i, {}):
                continue
            ns[i][j] = dict(state)
            ns[i][j]["timestamp"] = float(stale_t)
            _replay_active[i] = True
    return ns


def get_attack_flags() -> dict[str, dict[int, bool]]:
    return {
        "jamming": dict(_jam_active),
        "spoofing": dict(_spoof_active),
        "replay": dict(_replay_active),
    }
