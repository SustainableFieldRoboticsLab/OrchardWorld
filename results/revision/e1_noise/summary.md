# E1 - localization-noise robustness (16 evaluation seeds, nominal condition)

(A) input noise: planner inputs only; (B) servo: per-attempt aim offset of the grasp servo.
Mean [95 % bootstrap CI over seeds].

| cell | controller | n | trunc. | harvest/pick % | contact % | fruit/robot/min | safe return |
|---|---|---|---|---|---|---|---|
| clean | wm_full_cal | 16 | 1 | 56.1 [49.0, 63.4] | 38.4 [28.1, 48.8] | 0.559 [0.461, 0.668] | 0.116 [0.006, 0.231] |
| clean | detect_and_pick | 16 | 2 | 32.8 [27.3, 38.6] | 60.9 [52.5, 69.1] | 0.883 [0.672, 1.109] | -0.532 [-0.806, -0.245] |
| servo05 | wm_full_cal | 16 | 1 | 52.5 [45.3, 60.3] | 46.0 [36.0, 56.2] | 0.531 [0.406, 0.668] | 0.108 [-0.004, 0.218] |
| servo05 | detect_and_pick | 16 | 1 | 31.0 [26.8, 35.6] | 59.7 [51.1, 68.0] | 0.836 [0.660, 1.023] | -0.551 [-0.821, -0.280] |
| servo10 | wm_full_cal | 16 | 1 | 54.4 [47.7, 60.6] | 33.6 [22.9, 44.4] | 0.535 [0.438, 0.648] | 0.175 [0.043, 0.296] |
| servo10 | detect_and_pick | 16 | 2 | 31.5 [26.7, 36.3] | 60.5 [51.7, 68.9] | 0.875 [0.664, 1.102] | -0.627 [-0.892, -0.350] |
| servo20 | wm_full_cal | 16 | 1 | 48.8 [42.1, 55.9] | 46.3 [38.6, 54.0] | 0.508 [0.406, 0.621] | 0.030 [-0.099, 0.162] |
| servo20 | detect_and_pick | 16 | 2 | 26.1 [20.3, 31.4] | 61.5 [50.4, 71.6] | 0.686 [0.508, 0.864] | -0.653 [-0.906, -0.427] |
| in05 | wm_full_cal | 16 | 2 | 55.6 [49.0, 62.3] | 41.9 [30.8, 52.9] | 0.580 [0.445, 0.766] | 0.166 [0.039, 0.310] |
| in10 | wm_full_cal | 16 | 1 | 52.5 [45.8, 58.9] | 37.0 [30.0, 44.7] | 0.473 [0.375, 0.566] | 0.102 [0.025, 0.184] |
| in20 | wm_full_cal | 16 | 1 | 58.1 [49.5, 67.1] | 41.0 [31.0, 51.4] | 0.441 [0.371, 0.516] | 0.131 [0.042, 0.213] |
| drop10 | wm_full_cal | 16 | 2 | 47.5 [40.3, 55.2] | 43.4 [30.6, 55.9] | 0.498 [0.375, 0.648] | 0.125 [0.015, 0.245] |
| drop20 | wm_full_cal | 16 | 1 | 58.1 [50.2, 67.6] | 41.8 [31.7, 51.6] | 0.496 [0.410, 0.578] | 0.150 [0.065, 0.233] |
| combo | wm_full_cal | 16 | 1 | 44.4 [34.7, 53.1] | 42.6 [33.0, 52.4] | 0.422 [0.309, 0.535] | 0.080 [0.007, 0.144] |

## Paired differences vs the clean cell of the same controller

- servo05 / wm_full_cal:
  - harvest_success: -3.6 [-11.7, +4.0], t=-0.87, p=0.396, Wilcoxon p=0.397, 5/16 seeds > 0
  - contact_rate: +7.5 [+0.2, +15.4], t=+1.88, p=0.080, Wilcoxon p=0.096, 9/16 seeds > 0
  - throughput: -0.027 [-0.098, +0.051], t=-0.70, p=0.495, Wilcoxon p=0.622, 4/16 seeds > 0
  - safe_return: -0.007 [-0.124, +0.106], t=-0.12, p=0.905, Wilcoxon p=0.980, 7/16 seeds > 0
- servo05 / detect_and_pick:
  - harvest_success: -1.8 [-5.6, +2.3], t=-0.86, p=0.401, Wilcoxon p=0.433, 7/16 seeds > 0
  - contact_rate: -1.1 [-4.8, +2.2], t=-0.62, p=0.542, Wilcoxon p=0.650, 7/16 seeds > 0
  - throughput: -0.047 [-0.180, +0.082], t=-0.67, p=0.512, Wilcoxon p=0.648, 7/16 seeds > 0
  - safe_return: -0.019 [-0.176, +0.153], t=-0.22, p=0.832, Wilcoxon p=0.782, 6/16 seeds > 0
- servo10 / wm_full_cal:
  - harvest_success: -1.7 [-8.5, +5.3], t=-0.45, p=0.656, Wilcoxon p=0.660, 8/16 seeds > 0
  - contact_rate: -4.8 [-10.2, +0.7], t=-1.68, p=0.114, Wilcoxon p=0.100, 4/16 seeds > 0
  - throughput: -0.023 [-0.105, +0.059], t=-0.54, p=0.597, Wilcoxon p=0.560, 6/16 seeds > 0
  - safe_return: +0.059 [-0.051, +0.180], t=+0.96, p=0.351, Wilcoxon p=0.495, 10/16 seeds > 0
- servo10 / detect_and_pick:
  - harvest_success: -1.3 [-6.6, +3.3], t=-0.49, p=0.633, Wilcoxon p=1.000, 8/16 seeds > 0
  - contact_rate: -0.4 [-4.3, +4.0], t=-0.20, p=0.846, Wilcoxon p=0.427, 6/16 seeds > 0
  - throughput: -0.008 [-0.188, +0.148], t=-0.09, p=0.931, Wilcoxon p=0.789, 7/16 seeds > 0
  - safe_return: -0.095 [-0.298, +0.099], t=-0.91, p=0.376, Wilcoxon p=0.433, 7/16 seeds > 0
- servo20 / wm_full_cal:
  - harvest_success: -7.4 [-17.5, +2.8], t=-1.38, p=0.188, Wilcoxon p=0.252, 7/16 seeds > 0
  - contact_rate: +7.8 [+0.5, +15.3], t=+2.02, p=0.062, Wilcoxon p=0.074, 11/16 seeds > 0
  - throughput: -0.051 [-0.164, +0.070], t=-0.82, p=0.424, Wilcoxon p=0.299, 7/16 seeds > 0
  - safe_return: -0.086 [-0.230, +0.056], t=-1.12, p=0.279, Wilcoxon p=0.404, 5/16 seeds > 0
- servo20 / detect_and_pick:
  - harvest_success: -6.8 [-14.4, -0.3], t=-1.84, p=0.085, Wilcoxon p=0.065, 4/16 seeds > 0
  - contact_rate: +0.6 [-5.0, +5.9], t=+0.21, p=0.837, Wilcoxon p=0.776, 7/16 seeds > 0
  - throughput: -0.197 [-0.355, -0.043], t=-2.37, p=0.031, Wilcoxon p=0.038, 4/16 seeds > 0
  - safe_return: -0.121 [-0.316, +0.098], t=-1.11, p=0.286, Wilcoxon p=0.193, 5/16 seeds > 0
- in05 / wm_full_cal:
  - harvest_success: -0.5 [-6.9, +5.9], t=-0.15, p=0.886, Wilcoxon p=0.875, 8/16 seeds > 0
  - contact_rate: +3.4 [-1.9, +8.8], t=+1.22, p=0.241, Wilcoxon p=0.198, 8/16 seeds > 0
  - throughput: +0.022 [-0.126, +0.207], t=+0.25, p=0.803, Wilcoxon p=1.000, 5/16 seeds > 0
  - safe_return: +0.050 [-0.059, +0.177], t=+0.80, p=0.438, Wilcoxon p=0.706, 8/16 seeds > 0
- in10 / wm_full_cal:
  - harvest_success: -3.6 [-8.7, +1.9], t=-1.28, p=0.219, Wilcoxon p=0.156, 5/16 seeds > 0
  - contact_rate: -1.5 [-8.7, +6.5], t=-0.37, p=0.719, Wilcoxon p=0.706, 6/16 seeds > 0
  - throughput: -0.086 [-0.148, -0.020], t=-2.48, p=0.025, Wilcoxon p=0.033, 3/16 seeds > 0
  - safe_return: -0.013 [-0.082, +0.068], t=-0.34, p=0.740, Wilcoxon p=0.211, 5/16 seeds > 0
- in20 / wm_full_cal:
  - harvest_success: +2.0 [-4.6, +8.1], t=+0.60, p=0.555, Wilcoxon p=0.379, 9/16 seeds > 0
  - contact_rate: +2.6 [-5.3, +10.7], t=+0.61, p=0.550, Wilcoxon p=0.820, 6/16 seeds > 0
  - throughput: -0.117 [-0.223, -0.031], t=-2.28, p=0.037, Wilcoxon p=0.016, 3/16 seeds > 0
  - safe_return: +0.016 [-0.086, +0.108], t=+0.31, p=0.763, Wilcoxon p=0.562, 11/16 seeds > 0
- drop10 / wm_full_cal:
  - harvest_success: -8.6 [-13.8, -3.7], t=-3.22, p=0.006, Wilcoxon p=0.009, 2/16 seeds > 0
  - contact_rate: +4.9 [-2.1, +13.3], t=+1.23, p=0.239, Wilcoxon p=0.510, 8/16 seeds > 0
  - throughput: -0.061 [-0.156, +0.065], t=-1.02, p=0.324, Wilcoxon p=0.046, 3/16 seeds > 0
  - safe_return: +0.009 [-0.076, +0.105], t=+0.20, p=0.845, Wilcoxon p=0.980, 9/16 seeds > 0
- drop20 / wm_full_cal:
  - harvest_success: +2.0 [-4.2, +9.5], t=+0.56, p=0.583, Wilcoxon p=0.691, 5/16 seeds > 0
  - contact_rate: +3.4 [-2.6, +9.3], t=+1.07, p=0.302, Wilcoxon p=0.256, 8/16 seeds > 0
  - throughput: -0.062 [-0.141, +0.012], t=-1.53, p=0.146, Wilcoxon p=0.163, 4/16 seeds > 0
  - safe_return: +0.035 [-0.067, +0.135], t=+0.64, p=0.529, Wilcoxon p=0.632, 9/16 seeds > 0
- combo / wm_full_cal:
  - harvest_success: -11.7 [-22.8, -1.5], t=-2.10, p=0.053, Wilcoxon p=0.053, 4/16 seeds > 0
  - contact_rate: +4.2 [-3.6, +11.7], t=+1.05, p=0.312, Wilcoxon p=0.363, 9/16 seeds > 0
  - throughput: -0.137 [-0.242, -0.047], t=-2.66, p=0.018, Wilcoxon p=0.009, 3/16 seeds > 0
  - safe_return: -0.036 [-0.168, +0.096], t=-0.51, p=0.615, Wilcoxon p=0.597, 6/16 seeds > 0

## Sanity check: clean cell vs the existing 16-seed results (expected: no difference)

- wm_full_cal harvest_success: -0.3 [-5.9, +5.3], t=-0.10, p=0.918, Wilcoxon p=0.820, 7/16 seeds > 0
- wm_full_cal safe_return: -0.079 [-0.181, +0.023], t=-1.45, p=0.167, Wilcoxon p=0.231, 6/16 seeds > 0
- detect_and_pick harvest_success: -0.3 [-4.8, +3.9], t=-0.13, p=0.896, Wilcoxon p=0.959, 7/16 seeds > 0
- detect_and_pick safe_return: +0.146 [-0.052, +0.334], t=+1.41, p=0.179, Wilcoxon p=0.211, 10/16 seeds > 0
