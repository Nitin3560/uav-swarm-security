# Paper Simulation

Standalone research simulation for the failure-aware UAV swarm study.

This project intentionally does not reuse the old `case study1` controller or
fault-handling code. It only borrows the environment backend idea from the
existing workspace and rebuilds the study logic from scratch.

Core comparisons:

- `pid`
- `generic`
- `failure_aware`

Core scenarios:

- `nominal`
- `wind`
- `sensor`
- `comm`
- `wind_comm`

Main robustness metrics:

- recovery time
- degradation percentage
- peak error spike

Outputs are written under `paper_sim/outputs`.

## Architecture

- `env.py`: transparent swarm dynamics with direct position/velocity state
- `faults.py`: seeded wind, sensor, and communication fault injection
- `estimation.py`: constant-velocity predictor/corrector used under sensing faults
- `diagnosis.py`: residual-based fault diagnosis with configurable persistence
- `controllers.py`: PID baseline, generic supervisor, and diagnosis-driven failure-aware controller
- `run_study.py`: single-scenario runner with logging, metrics, and ablations
- `run_matrix.py`: multi-seed orchestration and summary aggregation

## Diagnosis Logic

The diagnosis layer is deliberately transparent:

- sensor confidence rises when measurement innovation/residual stays above threshold for several control steps
- wind confidence rises when sustained unmodeled tracking drift exceeds a configured threshold
- communication confidence rises when packet quality drops or stale/missing packets persist
- hysteresis/persistence counters suppress chattering and smooth recovery

Each control step logs:

- active diagnosed fault
- sensor/wind/comm confidence in `[0, 1]`

## Estimator Logic

Each UAV uses a lightweight constant-velocity estimator:

- predict next position from filtered state and velocity
- compute innovation from the latest measurement
- reduce measurement trust as sensor-fault confidence grows
- gate large outliers
- fall back to prediction when measurements drop out or freeze

This keeps the sensor-fault mitigation mathematically inspectable and publication-friendly.

## Reconfiguration Rules

The enhanced failure-aware controller responds to diagnosis output:

- `wind`: increase damping and add drift-canceling bias
- `sensor`: trust filtered state more, reduce aggressive motion, and reduce noisy consensus coupling
- `comm`: reduce dependence on unreliable neighbors, hold last valid shared state, and soften consensus coupling
- `recovery`: blend back toward nominal settings instead of switching abruptly

## New Metrics

In addition to pre/fault/post mean error, the runner now logs:

- RMSE
- peak error spike
- recovery time
- settling time after fault
- maximum formation deformation
- inter-UAV spacing violation count
- time above safety threshold
- total control effort
- stable-run indicator
- diagnosis confidences and estimator innovation statistics

`run_matrix.py` aggregates means, standard deviations, and 95% confidence intervals across seeds.

## Ablation Modes

Failure-aware runs support:

- `full`
- `no_diagnosis`
- `no_filter`
- `no_comm_fallback`
- `no_wind_comp`
- `no_recovery_schedule`

## Example Commands

Baseline study:

```bash
python -m paper_sim.run_matrix \
  --config /Users/nitin/Desktop/failure/paper_sim/configs/base.yaml \
  --scenarios nominal wind sensor comm \
  --controllers pid generic failure_aware \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --output-root /Users/nitin/Desktop/failure/paper_sim/outputs_study
```

Single upgraded run:

```bash
python -m paper_sim.run_study \
  --config /Users/nitin/Desktop/failure/paper_sim/configs/base.yaml \
  --scenario sensor \
  --controller failure_aware \
  --seed 1 \
  --output-root /Users/nitin/Desktop/failure/paper_sim/outputs_example
```

Ablation study:

```bash
python -m paper_sim.run_matrix \
  --config /Users/nitin/Desktop/failure/paper_sim/configs/base.yaml \
  --scenarios sensor comm \
  --controllers failure_aware \
  --ablations full no_diagnosis no_filter no_comm_fallback no_wind_comp no_recovery_schedule \
  --seeds 1 2 3 4 5 \
  --output-root /Users/nitin/Desktop/failure/paper_sim/outputs_ablation
```
