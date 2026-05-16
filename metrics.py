from __future__ import annotations

import numpy as np


def connectivity_rate(positions: np.ndarray, comm_range_m: float) -> float:
    n = positions.shape[0]
    if n <= 1:
        return 1.0
    good = 0
    for i in range(n):
        ok = False
        for j in range(n):
            if i == j:
                continue
            if np.linalg.norm(positions[i, :2] - positions[j, :2]) <= comm_range_m:
                ok = True
                break
        good += int(ok)
    return float(good / n)


def mean_neighbor_count(positions: np.ndarray, comm_range_m: float) -> float:
    n = positions.shape[0]
    if n <= 1:
        return 0.0
    counts = []
    for i in range(n):
        count = 0
        for j in range(n):
            if i != j and np.linalg.norm(positions[i, :2] - positions[j, :2]) <= comm_range_m:
                count += 1
        counts.append(count)
    return float(np.mean(counts))


def formation_error(positions: np.ndarray, offsets: np.ndarray) -> float:
    center = np.mean(positions, axis=0)
    rel = positions - center
    return float(np.mean(np.linalg.norm((rel - offsets)[:, :2], axis=1)))


def tracking_errors(positions: np.ndarray, references: np.ndarray) -> tuple[float, float]:
    errs = np.linalg.norm((positions - references)[:, :2], axis=1)
    return float(np.mean(errs)), float(np.max(errs))


def rmse(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(vals))))


def degradation_pct(pre_fault_mean: float, fault_mean: float) -> float:
    if abs(pre_fault_mean) < 1e-9:
        return float("nan")
    return float((fault_mean - pre_fault_mean) / pre_fault_mean * 100.0)


def recovery_time(
    times: np.ndarray,
    errors: np.ndarray,
    fault_start_s: float,
    threshold: float,
    sustain_s: float,
) -> float:
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    for idx, t_s in enumerate(times):
        if t_s < fault_start_s:
            continue
        end_t = t_s + sustain_s
        mask = (times >= t_s) & (times < end_t)
        if np.any(mask) and np.all(errors[mask] <= threshold):
            return float(t_s - fault_start_s)
    return float("nan")


def settling_time(
    times: np.ndarray,
    values: np.ndarray,
    start_s: float,
    threshold: float,
    sustain_s: float,
) -> float:
    return recovery_time(times, values, start_s, threshold, sustain_s)


def spacing_violation_count(positions: np.ndarray, min_spacing_m: float) -> int:
    n = positions.shape[0]
    violations = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i, :2] - positions[j, :2]) < min_spacing_m:
                violations += 1
    return violations


def max_pairwise_spacing_error(positions: np.ndarray, offsets: np.ndarray) -> float:
    n = positions.shape[0]
    max_err = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            actual = np.linalg.norm(positions[i, :2] - positions[j, :2])
            desired = np.linalg.norm(offsets[i, :2] - offsets[j, :2])
            max_err = max(max_err, abs(actual - desired))
    return float(max_err)


def time_above_threshold(values: np.ndarray, threshold: float, dt: float) -> float:
    vals = np.asarray(values, dtype=float)
    return float(np.sum(vals > threshold) * dt)


def confidence_interval_95(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(vals.size))
