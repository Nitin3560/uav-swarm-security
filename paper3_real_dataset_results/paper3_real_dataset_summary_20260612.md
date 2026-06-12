# Paper 3 Real-Dataset Validation Summary

Generated on 2026-06-12 from DeepSense Scenario 23 and AERPAW/Fortem replay outputs.

## Dataset-Level Means

| dataset | cases | mean_rmse_gain_vs_periodic_pct | mean_aoi_reduction_vs_event_pct | mean_sync_reduction_vs_periodic_pct | mean_proposed_fuses |
| --- | --- | --- | --- | --- | --- |
| AERPAW AADM/Fortem radar | 3 | 8.49 | -4552.43 | 67.44 | 880.33 |
| DeepSense Scenario 23 | 3 | -1.51 | 90.39 | -164.20 | 20.44 |

## Per-Case Results

| dataset | case | proposed_rmse_m | periodic_rmse_m | rmse_gain_vs_periodic_pct | proposed_mean_aoi | event_mean_aoi | aoi_reduction_vs_event_pct | proposed_full_syncs | periodic_full_syncs | sync_reduction_vs_periodic_pct | proposed_fuses | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSense Scenario 23 | scenario23_seq20 | 58.15 | 58.85 | 1.20 | 7.75 | 111.37 | 93.04 | 55.33 | 26.00 | -112.82 | 12.33 | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq32 | 101.78 | 102.48 | 0.69 | 6.71 | 134.37 | 95.01 | 98.33 | 31.00 | -217.20 | 13.33 | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq33 | 14.22 | 13.37 | -6.41 | 6.93 | 41.06 | 83.13 | 128.67 | 49.00 | -162.59 | 35.67 | seq33 uses v3 cache; 125/494 beam-power reads interpolated after local file timeouts |
| AERPAW AADM/Fortem radar | Opt1 | 118.39 | 120.74 | 1.95 | 1.60 | 0.03 | -6108.16 | 71.00 | 190.00 | 62.63 | 1033.00 | real radar-associated position stream; q_k from radar SINR/confidence proxy |
| AERPAW AADM/Fortem radar | Opt2 | 40.33 | 46.20 | 12.71 | 1.88 | 0.09 | -1991.98 | 33.00 | 180.00 | 81.67 | 783.00 | real radar-associated position stream; q_k from radar SINR/confidence proxy |
| AERPAW AADM/Fortem radar | Opt3 | 46.83 | 52.52 | 10.82 | 1.46 | 0.03 | -5557.14 | 68.00 | 162.00 | 58.02 | 825.00 | real radar-associated position stream; q_k from radar SINR/confidence proxy |
