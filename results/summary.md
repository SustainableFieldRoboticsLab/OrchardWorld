# OrchardWorld-Sim results summary

- Dataset: 21943 decisions, 5440 trees, 340 seeds, splits {'train': 17597, 'val': 2203, 'test': 2143}, 85.8 simulated robot-hours.

## Offline prediction (test trees)

| predictor | detach AUROC | success AP | contact AUROC | ECE | depth MAE cm | ms |
|---|---|---|---|---|---|---|
| Reactive baseline | 0.664 | 0.245 | 0.553 | 0.229 | nan | 0.07 |
| World model (RGB-D + robot state) | 0.778 | 0.469 | 0.772 | 0.149 | 45.2 | 9.70 |
| World model (no action input) | 0.757 | 0.279 | 0.719 | 0.166 | 46.2 | 9.82 |
| World model (RGB) | 0.744 | 0.398 | 0.717 | 0.160 | 46.6 | 9.59 |
| World model (RGB + robot state) | 0.760 | 0.467 | 0.751 | 0.151 | 46.1 | 9.67 |
| World model (RGB-D) | 0.756 | 0.442 | 0.748 | 0.147 | 45.9 | 9.68 |
| TD-MPC2-style world model | 0.701 | 0.359 | 0.721 | 0.148 | 47.7 | 11.38 |
| TD-MPC2-style + V-JEPA 2 features | 0.756 | 0.387 | 0.758 | 0.141 | 47.8 | 77.03 |

## Closed-loop control (main condition, mean over seeds)

| controller | seeds | harvest % | detach % | contact % | damage % | actions/fruit | fruit/robot/min | safe return | skip % |
|---|---|---|---|---|---|---|---|---|---|
| Detect-and-pick | 8 | 30.9 | 72.9 | 60.4 | 31.3 | 3.47 | 0.727 | -0.526 | 20.7 |
| Reactive retry policy | 8 | 29.4 | 64.0 | 62.7 | 28.5 | 3.63 | 0.914 | -0.623 | 6.6 |
| OrchardWorld-Sim | 8 | 45.6 | 92.9 | 43.2 | 20.0 | 2.47 | 0.461 | 0.030 | 56.0 |
| Without robot state | 8 | 41.8 | 85.0 | 44.0 | 28.5 | 2.63 | 0.422 | 0.001 | 62.0 |
| Without depth input | 8 | 45.6 | 87.1 | 45.0 | 14.8 | 2.15 | 0.352 | 0.144 | 73.5 |
| Without action conditioning | 8 | nan | nan | 0.0 | 0.0 | nan | 0.000 | -0.005 | 94.4 |
| Without skip option | 8 | 28.4 | 65.5 | 62.4 | 28.3 | 3.83 | 0.891 | -0.540 | 0.0 |
| TD-MPC2-style world model | 8 | 39.1 | 83.8 | 49.5 | 27.8 | 3.28 | 0.453 | -0.041 | 59.7 |
| TD-MPC2-style + V-JEPA 2 | 8 | 44.6 | 82.4 | 35.0 | 22.6 | 2.36 | 0.398 | 0.090 | 64.8 |

Paired safe-return difference OrchardWorld-Sim - Detect-and-pick: +0.556 (s.d. 0.490, 7/8 seeds better)

Paired safe-return difference OrchardWorld-Sim - Reactive retry policy: +0.653 (s.d. 0.544, 8/8 seeds better)

Paired safe-return difference TD-MPC2-style + V-JEPA 2 - Detect-and-pick: +0.616 (s.d. 0.458, 8/8 seeds better)

Paired safe-return difference TD-MPC2-style + V-JEPA 2 - Reactive retry policy: +0.713 (s.d. 0.518, 8/8 seeds better)

Paired safe-return difference TD-MPC2-style + V-JEPA 2 - OrchardWorld-Sim: +0.060 (s.d. 0.351, 3/8 seeds better)

All numbers are from simulation only (OrchardBench); no real-orchard claims.
