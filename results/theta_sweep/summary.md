# Skip-threshold (θ) sweep — OrchardWorld-Sim (full_s0), nominal, 8 seeds

Validation-selected θ = 0.15

| θ | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |
|---|---|---|---|---|---|---|---|---|
| −∞ (no skip) | 8 | 28.4 | 62.4 | 28.3 | 0.0 | 0.891 | -0.540 | 0 |
| -0.30 | 8 | 27.0 | 60.9 | 29.7 | 14.0 | 0.734 | -0.546 | 0 |
| -0.15 | 8 | 33.1 | 60.9 | 28.6 | 30.7 | 0.727 | -0.297 | 0 |
| -0.05 | 8 | 39.8 | 54.7 | 31.4 | 38.3 | 0.711 | -0.158 | 0 |
| +0.05 | 8 | 48.3 | 46.8 | 27.9 | 51.0 | 0.633 | 0.030 | 0 |
| +0.15 | 8 | 45.7 | 46.7 | 33.6 | 57.2 | 0.500 | -0.073 | 0 |
| +0.25 | 8 | 57.1 | 43.2 | 24.7 | 59.9 | 0.531 | 0.121 | 0 |
| +0.40 | 8 | 55.3 | 29.3 | 22.6 | 79.1 | 0.289 | 0.133 | 0 |

Reference controllers (main comparison, same seeds):

| controller | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |
|---|---|---|---|---|---|---|---|---|
| Detect-and-pick | 8 | 30.9 | 60.4 | 31.3 | 20.7 | 0.727 | -0.526 | 0 |
| Reactive retry | 8 | 29.4 | 62.7 | 28.5 | 6.6 | 0.914 | -0.623 | 0 |
| OrchardWorld-Sim (validation θ) | 8 | 45.6 | 43.2 | 20.0 | 56.0 | 0.461 | 0.030 | 0 |

Reproducibility check: re-run at the validation θ vs. the main comparison, safe-return difference per seed mean -0.103, max |diff| 0.405 (0 = bit-identical simulation).
