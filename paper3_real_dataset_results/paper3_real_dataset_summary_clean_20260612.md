# Paper 3 Real-Dataset Validation Summary

This clean summary keeps the DeepSense and AERPAW primary claims separate because the datasets stress different parts of the synchronisation policy.

## Dataset-Level Summary

| dataset | cases | mean_rmse_gain_vs_periodic_pct | primary_benefit_metric | mean_primary_benefit_pct | best_case | boundary_case |
| --- | --- | --- | --- | --- | --- | --- |
| AERPAW AADM/Fortem radar | 3 | 8.49 | Full-sync reduction vs periodic (%) | 67.44 | Opt2 | none |
| DeepSense Scenario 23 | 3 | -1.51 | AoI reduction vs event-triggered (%) | 90.39 | scenario23_seq32 | scenario23_seq33 |

## Per-Case Summary

| dataset | case | rmse_gain_vs_periodic_pct | primary_benefit_metric | primary_benefit_pct | proposed_rmse_m | periodic_rmse_m | proposed_mean_aoi | event_mean_aoi | proposed_full_syncs | periodic_full_syncs | proposed_fuses | paper_interpretation | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSense Scenario 23 | scenario23_seq20 | 1.20 | AoI reduction vs event-triggered (%) | 93.04 | 58.15 | 58.85 | 7.75 | 111.37 | 55.33 | 26.00 | 12.33 | Freshness gain from real mmWave q_k with near-neutral/slightly better RMSE | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq32 | 0.69 | AoI reduction vs event-triggered (%) | 95.01 | 101.78 | 102.48 | 6.71 | 134.37 | 98.33 | 31.00 | 13.33 | Freshness gain from real mmWave q_k with near-neutral/slightly better RMSE | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq33 | -6.41 | AoI reduction vs event-triggered (%) | 83.13 | 14.22 | 13.37 | 6.93 | 41.06 | 128.67 | 49.00 | 35.67 | Freshness gain from real mmWave q_k; seq33 is a boundary case for RMSE | seq33 uses v3 cache; 125/494 beam-power reads interpolated after local file timeouts |
| AERPAW AADM/Fortem radar | Opt1 | 1.95 | Full-sync reduction vs periodic (%) | 62.63 | 118.39 | 120.74 | 1.60 | 0.03 | 71.00 | 190.00 | 1033.00 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy |
| AERPAW AADM/Fortem radar | Opt2 | 12.71 | Full-sync reduction vs periodic (%) | 81.67 | 40.33 | 46.20 | 1.88 | 0.09 | 33.00 | 180.00 | 783.00 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy |
| AERPAW AADM/Fortem radar | Opt3 | 10.82 | Full-sync reduction vs periodic (%) | 58.02 | 46.83 | 52.52 | 1.46 | 0.03 | 68.00 | 162.00 | 825.00 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy |
