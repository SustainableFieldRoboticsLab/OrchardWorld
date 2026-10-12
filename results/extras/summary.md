# Phase 3 extras: calibration + selectivity control

## Trade-off points (nominal, 8 seeds; mean over seeds)

| controller | θ | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |
|---|---|---|---|---|---|---|---|---|---|
| detect-and-pick | - | 8 | 30.9 | 60.4 | 31.3 | 20.7 | 0.727 | -0.526 | 0 |
| reactive retry | - | 8 | 29.4 | 62.7 | 28.5 | 6.6 | 0.914 | -0.623 | 0 |
| selective D&P | -0.342 | 8 | 28.1 | 59.0 | 24.4 | 19.5 | 0.734 | -0.395 | 0 |
| selective D&P | -0.227 | 8 | 30.6 | 56.9 | 22.5 | 35.2 | 0.610 | -0.287 | 1 |
| selective D&P | -0.104 | 8 | 33.2 | 50.2 | 20.2 | 55.5 | 0.422 | -0.082 | 0 |
| selective D&P | -0.050 (val) | 8 | 39.5 | 44.7 | 25.5 | 64.4 | 0.375 | -0.081 | 0 |
| selective D&P | +0.025 | 8 | 37.1 | 44.7 | 22.4 | 77.6 | 0.203 | -0.081 | 0 |
| world model | −∞ | 8 | 28.4 | 62.4 | 28.3 | 0.0 | 0.891 | -0.540 | 0 |
| world model | -0.30 | 8 | 27.0 | 60.9 | 29.7 | 14.0 | 0.734 | -0.546 | 0 |
| world model | -0.15 | 8 | 33.1 | 60.9 | 28.6 | 30.7 | 0.727 | -0.297 | 0 |
| world model | -0.05 | 8 | 39.8 | 54.7 | 31.4 | 38.3 | 0.711 | -0.158 | 0 |
| world model | +0.05 | 8 | 48.3 | 46.8 | 27.9 | 51.0 | 0.633 | 0.030 | 0 |
| world model | +0.15 | 8 | 45.7 | 46.7 | 33.6 | 57.2 | 0.500 | -0.073 | 0 |
| world model | +0.25 | 8 | 57.1 | 43.2 | 24.7 | 59.9 | 0.531 | 0.121 | 0 |
| world model | +0.40 | 8 | 55.3 | 29.3 | 22.6 | 79.1 | 0.289 | 0.133 | 0 |
| world model (main run, val θ) | val | 8 | 45.6 | 43.2 | 20.0 | 56.0 | 0.461 | 0.030 | 0 |
| world model calibrated | val (cal) | 8 | 52.2 | 39.1 | 24.9 | 63.4 | 0.523 | 0.109 | 0 |

## Paired comparisons (per seed)

- world model (val θ) − selective D&P (val θ), safe return: +0.111 (s.d. 0.288, t=+1.09, 6/8 seeds positive)
- same, harvest success per attempt: +6.055 (s.d. 18.335, t=+0.93, 5/8 seeds positive)
- same, contact rate: -1.578 (s.d. 21.169, t=-0.21, 4/8 seeds positive)
- selective D&P (val θ) − detect-and-pick, safe return: +0.445 (s.d. 0.494, t=+2.55, 6/8 seeds positive)
- calibrated − uncalibrated world model, safe return: +0.079 (s.d. 0.368, t=+0.61, 3/8 seeds positive)

## Matched throughput (world-model θ point closest in fruit/robot/min)

- selective D&P θ=-0.342 (0.734 fpm): HS 28.1 %, CR 59.0 %, SR -0.395  vs  world model θ=-0.30 (0.734 fpm): HS 27.0 %, CR 60.9 %, SR -0.546
- selective D&P θ=-0.227 (0.610 fpm): HS 30.6 %, CR 56.9 %, SR -0.287  vs  world model θ=+0.05 (0.633 fpm): HS 48.3 %, CR 46.8 %, SR 0.030
- selective D&P θ=-0.104 (0.422 fpm): HS 33.2 %, CR 50.2 %, SR -0.082  vs  world model θ=+0.15 (0.500 fpm): HS 45.7 %, CR 46.7 %, SR -0.073
- selective D&P θ=-0.050 (0.375 fpm): HS 39.5 %, CR 44.7 %, SR -0.081  vs  world model θ=+0.40 (0.289 fpm): HS 55.3 %, CR 29.3 %, SR 0.133
- selective D&P θ=+0.025 (0.203 fpm): HS 37.1 %, CR 44.7 %, SR -0.081  vs  world model θ=+0.40 (0.289 fpm): HS 55.3 %, CR 29.3 %, SR 0.133

## Calibration (test trees, ECE before → after, per run)

- full_s0: detach 0.044→0.038, success 0.365→0.030, contact 0.065→0.048; θ 0.15 → 0.0
- full_s1: detach 0.051→0.033, success 0.358→0.026, contact 0.032→0.037; θ 0.1 → -0.1
- full_s2: detach 0.054→0.034, success 0.330→0.034, contact 0.047→0.049; θ 0.0 → -0.125
- no_action_s0: detach 0.205→0.040, success 0.263→0.026, contact 0.015→0.046; θ 0.4 → -0.225
- no_action_s1: detach 0.171→0.043, success 0.273→0.019, contact 0.029→0.044; θ -0.175 → -0.225
- no_action_s2: detach 0.186→0.021, success 0.287→0.017, contact 0.065→0.041; θ 0.4 → -0.225
- rgb_s0: detach 0.073→0.027, success 0.407→0.044, contact 0.083→0.057; θ 0.3 → 0.0
- rgb_s1: detach 0.038→0.040, success 0.358→0.053, contact 0.049→0.053; θ 0.275 → 0.025
- rgb_s2: detach 0.026→0.036, success 0.347→0.043, contact 0.057→0.050; θ 0.125 → -0.1
- rgb_state_s0: detach 0.050→0.024, success 0.336→0.039, contact 0.059→0.044; θ 0.25 → -0.05
- rgb_state_s1: detach 0.070→0.049, success 0.362→0.029, contact 0.028→0.044; θ 0.15 → -0.025
- rgb_state_s2: detach 0.046→0.030, success 0.333→0.041, contact 0.075→0.041; θ 0.15 → -0.175
- rgbd_s0: detach 0.032→0.023, success 0.381→0.038, contact 0.073→0.064; θ 0.325 → 0.025
- rgbd_s1: detach 0.048→0.036, success 0.316→0.036, contact 0.022→0.060; θ 0.075 → 0.025
- rgbd_s2: detach 0.041→0.036, success 0.355→0.033, contact 0.050→0.060; θ 0.125 → -0.025
- tdmpc_s0: detach 0.027→0.023, success 0.367→0.028, contact 0.049→0.047; θ 0.075 → -0.125
- tdmpc_s1: detach 0.041→0.024, success 0.379→0.026, contact 0.040→0.049; θ 0.175 → -0.025
- tdmpc_s2: detach 0.023→0.038, success 0.366→0.023, contact 0.041→0.048; θ 0.175 → 0.05
- vjepa_s0: detach 0.038→0.029, success 0.358→0.031, contact 0.049→0.047; θ 0.175 → 0.0
- vjepa_s1: detach 0.038→0.035, success 0.351→0.028, contact 0.038→0.043; θ 0.15 → -0.05
- vjepa_s2: detach 0.036→0.046, success 0.338→0.041, contact 0.026→0.045; θ 0.175 → 0.0

Selective D&P predictor (straight picks, test trees): success AUROC 0.654, contact AUROC 0.588.

Note: GPU physics is not bit-reproducible; differences below ~0.1 safe return between single points are within run-to-run noise (see theta_sweep reproducibility check).
