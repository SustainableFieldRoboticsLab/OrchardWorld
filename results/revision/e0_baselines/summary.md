# E0 - baselines with action input (same test rows as Table 2)

Test: 1767 non-skip decisions; success base rate on picks 0.207 (= chance AP).

| model | detach AUROC | success AP (x chance) | contact AUROC | damage AUROC |
|---|---|---|---|---|
| R0 Detector features, no action (paper baseline, re-fit) | 0.664 | 0.245 (1.18x) | 0.553 | 0.553 |
| A1 LR: detector features + action | 0.666 | 0.301 (1.45x) | 0.630 | 0.577 |
| A2 LR: detector features + action + feature x action | 0.668 | 0.329 (1.59x) | 0.633 | 0.583 |
| A3 LR: 54-d state + action | 0.712 | 0.305 (1.48x) | 0.686 | 0.611 |
| A4 Gradient-boosted trees: 54-d state, no action | 0.717 | 0.270 (1.31x) | 0.729 | 0.648 |
| A5 Gradient-boosted trees: 54-d state + action | 0.728 | 0.459 (2.22x) | 0.771 | 0.687 |
| world model full (mean of 3 seeds, results/offline) | 0.778 | 0.469 (2.27x) | 0.772 | 0.682 |
| world model no_action (mean of 3 seeds, results/offline) | 0.757 | 0.279 (1.35x) | 0.719 | 0.649 |
| world model rgbd (mean of 3 seeds, results/offline) | 0.756 | 0.442 (2.14x) | 0.748 | 0.667 |
| world model tdmpc (mean of 3 seeds, results/offline) | 0.701 | 0.359 (1.73x) | 0.721 | 0.627 |

R0 must reproduce the paper's detector baseline (0.664 / 0.245 / 0.553); if it does not, stop and report - the row selection differs.
