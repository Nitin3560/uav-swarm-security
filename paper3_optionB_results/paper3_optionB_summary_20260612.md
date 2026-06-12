# Paper 3 Option B Results (30 seeds)

Option B compares a good upstream filter against a fixed-R upstream filter. The intended claim is that integrity-aware synchronization matters most when upstream sensing integrity is weaker.

## Dataset/filter summary

| dataset | filter | cases | n_seeds | mean_rmse_gain_vs_periodic_pct | mean_aoi_reduction_vs_event_pct | mean_sync_reduction_vs_periodic_pct | mean_rmse_gap_vs_unconditional_pct | mean_proposed_syncs | mean_periodic_syncs | mean_proposed_fuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AERPAW/Fortem radar | fixed | 3 | 30 | 11.66 | -22896.66 | -64.77 | 0.66 | 290.74 | 177.33 | 937.94 |
| AERPAW/Fortem radar | good | 3 | 30 | 8.51 | -4610.93 | 67.42 | 0.92 | 57.33 | 177.33 | 883.78 |
| DeepSense Scenario 23 | fixed | 3 | 30 | 49.41 | 88.71 | -70.57 | 9.57 | 53.56 | 35.33 | 236.62 |
| DeepSense Scenario 23 | good | 3 | 30 | 1.75 | 89.45 | 62.11 | 0.05 | 13.27 | 35.33 | 62.67 |
| Synthetic Option B | fixed | 1 | 30 | 29.39 | 76.93 | 32.46 | 6.51 | 54.03 | 80.00 | 382.10 |
| Synthetic Option B | good | 1 | 30 | 5.85 | 79.33 | 93.83 | 2.01 | 4.93 | 80.00 | 157.27 |

## Per-case summary

| dataset | case | filter | n_seeds | rmse_gain_vs_periodic_pct | aoi_reduction_vs_event_pct | sync_reduction_vs_periodic_pct | rmse_gap_vs_unconditional_pct | proposed_syncs | periodic_syncs | proposed_fuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic Option B | aggregate | fixed | 30 | 29.39 | 76.93 | 32.46 | 6.51 | 54.03 | 80.00 | 382.10 |
| Synthetic Option B | aggregate | good | 30 | 5.85 | 79.33 | 93.83 | 2.01 | 4.93 | 80.00 | 157.27 |
| DeepSense Scenario 23 | scenario23_seq20 | fixed | 30 | 48.67 | 96.61 | -157.82 | 9.07 | 67.03 | 26.00 | 186.30 |
| DeepSense Scenario 23 | scenario23_seq32 | fixed | 30 | 49.57 | 87.39 | -70.97 | 9.18 | 53.00 | 31.00 | 213.67 |
| DeepSense Scenario 23 | scenario23_seq33 | fixed | 30 | 50.00 | 82.14 | 17.07 | 10.46 | 40.63 | 49.00 | 309.90 |
| DeepSense Scenario 23 | scenario23_seq20 | good | 30 | 0.07 | 91.76 | 52.95 | 0.00 | 12.23 | 26.00 | 15.00 |
| DeepSense Scenario 23 | scenario23_seq32 | good | 30 | 0.00 | 93.13 | 71.83 | 0.00 | 8.73 | 31.00 | 16.67 |
| DeepSense Scenario 23 | scenario23_seq33 | good | 30 | 5.19 | 83.46 | 61.56 | 0.16 | 18.83 | 49.00 | 156.33 |
| AERPAW/Fortem radar | Opt1 | fixed | 30 | 3.47 | -19874.17 | -115.30 | 0.08 | 409.07 | 190.00 | 986.73 |
| AERPAW/Fortem radar | Opt2 | fixed | 30 | 11.77 | -9143.60 | 38.07 | 1.16 | 111.47 | 180.00 | 982.83 |
| AERPAW/Fortem radar | Opt3 | fixed | 30 | 19.74 | -39672.22 | -117.10 | 0.74 | 351.70 | 162.00 | 844.27 |
| AERPAW/Fortem radar | Opt1 | good | 30 | 1.96 | -6211.80 | 62.37 | 0.14 | 71.50 | 190.00 | 1036.33 |
| AERPAW/Fortem radar | Opt2 | good | 30 | 12.67 | -1887.63 | 82.70 | 1.84 | 31.13 | 180.00 | 790.27 |
| AERPAW/Fortem radar | Opt3 | good | 30 | 10.89 | -5733.36 | 57.18 | 0.79 | 69.37 | 162.00 | 824.73 |
