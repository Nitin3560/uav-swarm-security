# Digital Twin-Assisted UAV Swarm Security Simulation

This package contains the simulation code for a digital twin-assisted UAV swarm
security study. The core idea is to run a stochastic digital twin alongside the
physical swarm and use twin divergence as an intrusion signal for low-altitude
multi-UAV coordination.

The implementation extends the original fault-aware thesis simulation with:

- a Kalman-filter digital twin for each UAV
- chi-squared normalized innovation detection
- GLRT-style attack attribution for wind, jamming, spoofing, and replay
- CUSUM sequential confirmation
- finite-horizon trust-weighted MPC solved with OSQP
- attack injection for RF jamming, GPS/broadcast spoofing, replay, and compound wind+jamming

## Threat Model

The physical-fault cases remain part of the study because they are important
confounders for attack detection:

- `nominal`: no fault or attack
- `wind`: correlated physical drift
- `sensor`: onboard measurement corruption
- `comm`: degraded communication quality from the thesis setup
- `wind_comm`: combined thesis fault condition

The security-paper scenarios are:

- `jamming_full`: RF link suppression with `jam_power = 1.0`
- `jamming_partial`: lower-intensity jamming with `jam_power = 0.5`
- `spoofing_strong`: broadcast spoofing with `d_spoof = [1.5, 0, 0]`
- `spoofing_subtle`: broadcast spoofing with `d_spoof = [0.5, 0, 0]`
- `replay`: stale neighbor-state replay with a 5 s delay
- `compound`: wind plus full jamming

Spoofing corrupts the position broadcast received by neighbors, not the
spoofed UAV's own sensor measurement. Replay buffers clean pre-attack packets
and injects stale versions during the attack window.

## Architecture

Per control step, the intended data flow is:

```text
swarm dynamics
  -> fault injection
  -> attack injection
  -> digital twin predict/update
  -> chi-squared + GLRT + CUSUM IDS
  -> trust-weighted horizon MPC
  -> PID tracking of modified references
  -> next swarm state
```

Main modules:

- `env.py`: transparent swarm dynamics and state integration
- `faults.py`: seeded wind, sensor, and communication fault injection
- `attack_injection.py`: jamming, spoofing, and replay attack models
- `digital_twin.py`: stochastic twin and Kalman innovation statistics
- `ids.py`: chi-squared detector, GLRT attribution, and CUSUM confirmation
- `trust_mpc.py`: finite-horizon attack-conditioned MPC supervisor
- `security_metrics.py`: detection rate, false alarm rate, attribution accuracy, TTD, and disambiguation
- `run_study.py`: single-scenario runner
- `run_matrix.py`: multi-seed orchestration and aggregate summaries

## Controllers

Implemented controller labels:

- `pid`: fixed PID baseline
- `generic`: legacy thesis supervisory controller
- `failure_aware`: proposed digital-twin IDS + trust MPC controller

Paper-facing aliases are also accepted:

- `pid_baseline -> pid`
- `prior_supervisory -> generic`
- `proposed_ids_mpc -> failure_aware`

Planned external baselines for the paper comparison are not yet implemented in
this package: threshold-only IDS, CUSUM-only IDS without GLRT attribution, and
always-on Byzantine-resilient consensus.

## Key Parameters

IDS parameters are fixed in `ids.py`:

- `ALPHA = 0.01`
- `DF = 3`
- `RHO = 1e-3`
- `B_CUSUM[H1_WIND] = 0.8`
- `B_CUSUM[H2_JAMMING] = 1.5`
- `B_CUSUM[H3_SPOOF] = 1.2`
- `B_CUSUM[H4_REPLAY] = 0.6`

MPC parameters are fixed in `trust_mpc.py`:

- `HORIZON = 5`
- `DELTA_MAX = 0.05`
- `SMOOTH_ALPHA = 0.20`
- terminal cost enters as `Q_TRACK + P_inf` in the final horizon block

`SMOOTH_ALPHA` is a conservative response smoother. It damps rapid MPC changes,
which helps sustained attacks but can slow response and release under brief or
intermittent attacks.

## Example Commands

Single proposed security run:

```bash
python -m paper_sim.run_study \
  --config /Users/nitin/Desktop/failure/paper_sim/configs/base.yaml \
  --scenario spoofing_subtle \
  --controller proposed_ids_mpc \
  --seed 1 \
  --output-root /Users/nitin/Desktop/failure/paper_sim/outputs_example
```

Full 30-seed security matrix used for the current results:

```bash
python -m paper_sim.run_matrix \
  --config /Users/nitin/Desktop/failure/paper_sim/configs/base.yaml \
  --scenarios nominal wind sensor jamming_full jamming_partial spoofing_strong spoofing_subtle replay compound \
  --controllers pid_baseline prior_supervisory proposed_ids_mpc \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --output-root /Users/nitin/Desktop/failure/paper_sim/results_security_30seed
```

Aggregate outputs are written under the chosen output root. The current full
matrix summaries are:

- `results_security_30seed/combined_concise_summary.csv`
- `results_security_30seed/combined_aggregate_summary.csv`

## Interpretation Notes

The proposed controller is strongest when the attack affects shared coordination
signals, such as spoofing, replay, and jamming. Sensor corruption is the hard
case: the twin's measurement update is itself corrupted, so the twin degrades
along with the onboard sensor. This is an expected limitation and should be
reported honestly in the paper discussion.

Under `H0_NONE`, the trust MPC intentionally returns zero reference correction.
Any nominal improvement should therefore be attributed to the broader supervisor
and estimator behavior, not to attack-mode MPC intervention.

## Paper 3: ISAC Digital-Twin Synchronisation

The Paper 3 extension evaluates an integrity-aware digital-twin synchronisation
policy for ISAC-enabled UAV systems. The synchronisation layer assumes that an
upstream ISAC estimator exports the triple `(x_hat, P, q)` at each timestep:
state estimate, covariance, and sensing-quality indicator. The policy then
chooses among `HOLD`, `FUSE`, and `SYNC` actions to balance twin freshness,
tracking error, and communication load.

Current real-dataset validation outputs are stored in:

- `paper3_real_dataset_results_30seed/paper3_real_dataset_results_30seed_20260612.csv`
- `paper3_real_dataset_results_30seed/paper3_real_dataset_summary_30seed_20260612.csv`
- `paper3_real_dataset_results_30seed/paper3_real_dataset_results_30seed.png`
- `dt_sync_v3_30seed_seq20_seq32_seq33_means.csv`
- `dt_sync_aerpaw_30seed_summary_means.csv`

DeepSense Scenario 23 validation uses real 60 GHz beam-power measurements for
`q_k` and real GPS trajectories; position measurements are simulated from the
real trajectory because DeepSense does not provide radar-derived position
measurements. Across 30 seeds on seq20, seq32, and seq33, the proposed policy
reduces AoI by 86.3% on average versus event-triggered synchronisation. Seq33
is a boundary case: AoI improves, but RMSE is 9.7% worse than periodic
synchronisation, and 125 of 494 beam-power reads were interpolated after local
file-system timeouts.

AERPAW validation uses the processed Fortem radar association streams generated
from the raw AADM dataset. Across 30 seeds on Opt1, Opt2, and Opt3, the
proposed policy improves RMSE by 8.5% on average versus periodic
synchronisation while reducing full synchronisations by 67.4%.
