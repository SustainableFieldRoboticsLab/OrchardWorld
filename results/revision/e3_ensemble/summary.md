# E3 - ensemble of 3 training seeds (calibrated)

## Offline (test split)

| model | detach AUROC | success AP | contact AUROC | ECE |
|---|---|---|---|---|
| full_s0 | 0.780 | 0.475 | 0.773 | 0.158 |
| full_s1 | 0.779 | 0.475 | 0.780 | 0.147 |
| full_s2 | 0.775 | 0.456 | 0.764 | 0.144 |
| full_s0_cal | 0.780 | 0.475 | 0.773 | 0.039 |
| full_s1_cal | 0.779 | 0.475 | 0.780 | 0.032 |
| full_s2_cal | 0.775 | 0.456 | 0.764 | 0.039 |
| ensemble | 0.788 | 0.490 | 0.782 | 0.143 |
| ensemble_cal | 0.788 | 0.491 | 0.782 | 0.038 |

Validation-selected theta of the calibrated ensemble: -0.050

## Closed loop (16 seeds, nominal)

| controller | n | harvest/pick % | contact % | fruit/robot/min | safe return |
|---|---|---|---|---|---|
| ensemble, calibrated | 16 | 53.2 [45.2, 61.2] | 36.6 [27.5, 46.4] | 0.547 [0.418, 0.684] | 0.195 [0.079, 0.318] |
| world model, calibrated (single, full_s0) | 16 | 56.4 [49.6, 63.5] | 37.4 [26.7, 48.5] | 0.559 [0.449, 0.668] | 0.194 [0.122, 0.270] |
| detect-and-pick | 16 | 33.1 [26.9, 40.8] | 61.6 [50.1, 72.0] | 0.832 [0.680, 0.992] | -0.678 [-0.973, -0.377] |

- ensemble - world model, calibrated (single, full_s0):
  - harvest_success: -3.2 [-9.4, +3.2], t=-0.97, p=0.346, Wilcoxon p=0.348, 6/16 seeds > 0
  - contact_rate: -0.8 [-5.5, +4.0], t=-0.31, p=0.761, Wilcoxon p=0.597, 7/16 seeds > 0
  - throughput: -0.012 [-0.109, +0.098], t=-0.22, p=0.832, Wilcoxon p=0.607, 6/16 seeds > 0
  - safe_return: +0.000 [-0.091, +0.096], t=+0.01, p=0.994, Wilcoxon p=0.980, 9/16 seeds > 0
- ensemble - detect-and-pick:
  - harvest_success: +20.1 [+10.3, +28.9], t=+4.09, p=0.001, Wilcoxon p=0.003, 14/16 seeds > 0
  - contact_rate: -24.9 [-31.4, -17.9], t=-7.07, p=0.000, Wilcoxon p=0.000, 1/16 seeds > 0
  - throughput: -0.285 [-0.402, -0.168], t=-4.68, p=0.000, Wilcoxon p=0.002, 2/16 seeds > 0
  - safe_return: +0.872 [+0.646, +1.096], t=+7.38, p=0.000, Wilcoxon p=0.000, 15/16 seeds > 0
