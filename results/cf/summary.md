# Counterfactual rollouts (checkpoint/restore) — results

- 3697 complete groups (7 branches each) from 221 tree seeds; splits {'train': 2973, 'val': 353, 'test': 371}; decision quality on 371 test groups.

## 1. Noise ceiling: pick-center repeated from the identical state

| outcome | agreement | Cohen's kappa | rate (1st) | rate (repeat) |
|---|---|---|---|---|
| success | 0.96 | 0.92 | 0.39 | 0.39 |
| detach | 0.98 | 0.92 | 0.83 | 0.83 |
| contact | 0.99 | 0.99 | 0.51 | 0.51 |
| damage | 0.97 | 0.94 | 0.32 | 0.32 |

Reward correlation between the two identical trials: 0.95; mean |difference| 0.06.

## 2. Paired effect of each action vs pick-center (same state; mean [95 % CI], tree-seed bootstrap)

| action | Δ success | Δ detach | Δ contact | Δ damage | Δ reward |
|---|---|---|---|---|---|
| pick_left | -0.216 [-0.239, -0.191] | -0.044 [-0.060, -0.027] | +0.208 [+0.187, +0.227] | +0.054 [+0.035, +0.072] | -0.320 [-0.355, -0.285] |
| pick_right | -0.224 [-0.248, -0.200] | -0.058 [-0.073, -0.044] | +0.223 [+0.202, +0.244] | +0.060 [+0.042, +0.078] | -0.338 [-0.372, -0.305] |
| repos_left | -0.386 [-0.405, -0.367] | -0.815 [-0.832, -0.798] | -0.112 [-0.140, -0.083] | -0.129 [-0.152, -0.107] | -0.248 [-0.281, -0.215] |
| repos_right | -0.386 [-0.406, -0.368] | -0.816 [-0.832, -0.798] | -0.085 [-0.117, -0.055] | -0.096 [-0.122, -0.072] | -0.289 [-0.324, -0.252] |
| pick_center (repeat) | -0.001 [-0.007, +0.006] | -0.002 [-0.007, +0.003] | +0.001 [-0.002, +0.004] | -0.002 [-0.007, +0.003] | +0.001 [-0.007, +0.010] |

(The 'repeat' row is the noise level: any action effect of similar size is not distinguishable from re-running the same action.)

## 3. Decision quality on test groups (realized counterfactual outcomes)

| policy | reward / decision | regret vs oracle | best-action rate | skip % | success / pick | contact / action |
|---|---|---|---|---|---|---|
| always skip | +0.000 [+0.000, +0.000] | +0.400 [+0.340, +0.458] | nan | 100 | nan | nan |
| always pick-center (detect-and-pick) | -0.052 [-0.156, +0.056] | +0.453 [+0.388, +0.516] | 0.49 | 0 | 0.37 | 0.49 |
| selective detect-and-pick | +0.044 [+0.004, +0.082] | +0.356 [+0.315, +0.402] | 0.60 | 73 | 0.46 | 0.33 |
| oracle (counterfactual best) | +0.400 [+0.341, +0.457] | +0.000 [+0.000, +0.000] | 1.00 | 56 | 1.00 | 0.23 |
| world model full [3 seeds] | +0.124 (+0.115..+0.134) | +0.276 (+0.266..+0.285) | 0.58 | 48 | 0.51 | 0.31 |
| world model full (calibrated) [3 seeds] | +0.117 (+0.109..+0.128) | +0.283 (+0.272..+0.292) | 0.57 | 52 | 0.53 | 0.29 |
| world model fullcf (calibrated) [3 seeds] | +0.149 (+0.135..+0.165) | +0.251 (+0.236..+0.265) | 0.69 | 67 | 0.68 | 0.17 |
| world model no_action [3 seeds] | +0.000 (-0.034..+0.040) | +0.400 (+0.360..+0.434) | 0.20 | 72 | 0.48 | 0.19 |
| world model no_action (calibrated) [3 seeds] | +0.066 (+0.025..+0.096) | +0.334 (+0.304..+0.375) | 0.46 | 49 | 0.47 | 0.29 |
| world model rgb [3 seeds] | +0.107 (+0.099..+0.114) | +0.293 (+0.286..+0.301) | 0.57 | 56 | 0.51 | 0.31 |
| world model rgb (calibrated) [3 seeds] | +0.108 (+0.100..+0.117) | +0.292 (+0.283..+0.300) | 0.59 | 55 | 0.52 | 0.30 |
| world model rgb_state [3 seeds] | +0.104 (+0.079..+0.124) | +0.296 (+0.276..+0.321) | 0.60 | 58 | 0.53 | 0.29 |
| world model rgb_state (calibrated) [3 seeds] | +0.105 (+0.088..+0.119) | +0.295 (+0.281..+0.312) | 0.60 | 55 | 0.54 | 0.30 |
| world model rgbd [3 seeds] | +0.136 (+0.127..+0.151) | +0.264 (+0.249..+0.273) | 0.59 | 49 | 0.54 | 0.28 |
| world model rgbd (calibrated) [3 seeds] | +0.135 (+0.119..+0.144) | +0.265 (+0.256..+0.281) | 0.61 | 56 | 0.57 | 0.25 |
| world model tdmpc [3 seeds] | +0.092 (+0.077..+0.102) | +0.308 (+0.298..+0.323) | 0.61 | 58 | 0.51 | 0.26 |
| world model tdmpc (calibrated) [3 seeds] | +0.090 (+0.077..+0.098) | +0.310 (+0.302..+0.323) | 0.61 | 58 | 0.52 | 0.27 |

(Baselines: mean [95 % CI]; world models: mean over training seeds (min..max).)

## 4. Within-state outcome discrimination across actions (test groups; mean over seeds)

| model | success AUROC (all actions) | success AUROC (3 picks) | contact AUROC (all) | contact AUROC (3 picks) |
|---|---|---|---|---|
| full [3 seeds] | 0.849 | 0.717 | 0.777 | 0.728 |
| fullcf [3 seeds] | 0.860 | 0.738 | 0.795 | 0.751 |
| no_action [3 seeds] | 0.563 | 0.570 | 0.707 | 0.672 |
| rgb [3 seeds] | 0.837 | 0.696 | 0.750 | 0.703 |
| rgb_state [3 seeds] | 0.854 | 0.728 | 0.758 | 0.704 |
| rgbd [3 seeds] | 0.844 | 0.709 | 0.772 | 0.750 |
| tdmpc [3 seeds] | 0.827 | 0.676 | 0.722 | 0.690 |

## 5. Prediction on the random-action test trees: full vs fullcf (trained with counterfactual data)

| run | detach AUROC | success AP | contact AUROC | ECE (uncal.) |
|---|---|---|---|---|
| full_s0 | 0.780 | 0.475 | 0.773 | 0.158 |
| full_s1 | 0.779 | 0.475 | 0.780 | 0.147 |
| full_s2 | 0.775 | 0.456 | 0.764 | 0.144 |
| fullcf_s0 | 0.790 | 0.482 | 0.784 | 0.161 |
| fullcf_s1 | 0.781 | 0.488 | 0.793 | 0.167 |
| fullcf_s2 | 0.762 | 0.453 | 0.776 | 0.121 |

## 6. Closed loop (nominal, paired seeds): world model trained with counterfactual data (calibrated)

| controller | n | harvest/pick % | contact % | fruit/robot/min | safe return |
|---|---|---|---|---|---|
| world model + counterfactual data, calibrated | 16 | 66.3 | 31.8 | 0.445 | 0.160 |
| world model, calibrated | 16 | 56.4 | 37.4 | 0.559 | 0.194 |
| detect-and-pick | 16 | 33.1 | 61.6 | 0.832 | -0.678 |
- vs world model, calibrated, harvest_success: +9.850 (t=+2.20, 10/16 positive)
- vs world model, calibrated, contact_rate: -5.620 (t=-1.49, 3/16 positive)
- vs world model, calibrated, throughput: -0.113 (t=-3.81, 1/16 positive)
- vs world model, calibrated, safe_return: -0.035 (t=-0.74, 8/16 positive)
- vs detect-and-pick, harvest_success: +33.154 (t=+5.35, 15/16 positive)
- vs detect-and-pick, contact_rate: -29.744 (t=-6.79, 0/16 positive)
- vs detect-and-pick, throughput: -0.387 (t=-10.56, 0/16 positive)
- vs detect-and-pick, safe_return: +0.837 (t=+5.39, 15/16 positive)

GPU physics is not bit-reproducible; compare effects with the noise ceiling (section 1).
