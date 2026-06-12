# Paper 3 Real-Dataset Validation Summary (30 Seeds)

DeepSense and AERPAW are both replayed over 30 sensing-quality/noise seeds. DeepSense uses real 60 GHz beam-power q_k and real GPS trajectories; AERPAW uses processed real Fortem radar association streams.

## Dataset-Level Summary

| dataset | cases | n_seeds | mean_rmse_gain_vs_periodic_pct | mean_primary_benefit_pct | primary_benefit_metric |
| --- | --- | --- | --- | --- | --- |
| AERPAW AADM/Fortem radar | 3 | 30 | 8.51 | 67.42 | Full-sync reduction vs periodic (%) |
| DeepSense Scenario 23 | 3 | 30 | -2.64 | 86.29 | AoI reduction vs event-triggered (%) |

## Per-Case Summary

| dataset | case | n_seeds | rmse_gain_vs_periodic_pct | primary_benefit_metric | primary_benefit_pct | proposed_rmse_m | periodic_rmse_m | proposed_mean_aoi | event_mean_aoi | proposed_full_syncs | periodic_full_syncs | sync_reduction_vs_periodic_pct | proposed_fuses | paper_interpretation | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSense Scenario 23 | scenario23_seq20 | 30 | 1.06 | AoI reduction vs event-triggered (%) | 93.16 | 49.88 | 50.41 | 7.49 | 109.39 | 61.17 | 26.00 | -135.26 | 12.87 | Freshness gain from real mmWave q_k with near-neutral/slightly better RMSE | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq32 | 30 | 0.67 | AoI reduction vs event-triggered (%) | 95.04 | 101.16 | 101.85 | 6.71 | 135.30 | 99.00 | 31.00 | -219.35 | 12.97 | Freshness gain from real mmWave q_k with near-neutral/slightly better RMSE | real beam-power q_k and real GPS trajectory; z_k simulated from trajectory |
| DeepSense Scenario 23 | scenario23_seq33 | 30 | -9.66 | AoI reduction vs event-triggered (%) | 70.67 | 4.72 | 4.30 | 5.68 | 19.37 | 182.03 | 49.00 | -271.50 | 39.70 | Freshness gain from real mmWave q_k; seq33 is a boundary case for RMSE | seq33 uses v3 cache; 125/494 beam-power reads interpolated after local file timeouts |
| AERPAW AADM/Fortem radar | Opt1 | 30 | 1.96 | Full-sync reduction vs periodic (%) | 62.37 | 118.34 | 120.72 | 1.56 | 0.02 | 71.50 | 190.00 | 62.37 | 1036.33 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy with 10% quality-estimation noise |
| AERPAW AADM/Fortem radar | Opt2 | 30 | 12.67 | Full-sync reduction vs periodic (%) | 82.70 | 40.33 | 46.19 | 1.85 | 0.09 | 31.13 | 180.00 | 82.70 | 790.27 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy with 10% quality-estimation noise |
| AERPAW AADM/Fortem radar | Opt3 | 30 | 10.89 | Full-sync reduction vs periodic (%) | 57.18 | 46.84 | 52.57 | 1.48 | 0.03 | 69.37 | 162.00 | 57.18 | 824.73 | Communication-efficient radar DT sync: better RMSE than periodic with fewer full synchronizations | real radar-associated position stream; q_k from radar SINR/confidence proxy with 10% quality-estimation noise |
