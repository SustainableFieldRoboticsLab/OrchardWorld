# OrchardWorld-Sim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23572538.svg)](https://doi.org/10.5281/zenodo.23572538)

Code, configuration, run logs and result files for the paper

> **OrchardWorld-Sim: Physics-Grounded Action-Conditioned World Models for Robotic Apple Harvesting in Digital Orchards**
> R. Sapkota, K. I. Roumeliotis

All results are from simulation (OrchardBench on NVIDIA Newton 1.3.0, MuJoCo-Warp 3.8.1, Warp 1.14.0; Franka FR3 on a Clearpath Ridgeback).

## Contents

| Path | What it is |
|---|---|
| `owsim/` | data collection, dataset building, world models, planner and baseline controllers, offline and closed-loop evaluation, counterfactual (checkpoint/restore) extension |
| `revision/` | additional experiments: action-aware baselines (E0), localization noise (E1), oracle-feature probe (E2), 3-seed ensemble (E3), 8-seed robustness (E5), within-state concordance (E6), effective sample size (ESS) |
| `orchardbench/` | the OrchardBench simulator package (`treesim`), Apache-2.0 |
| `assets/` | FR3 + hand URDF and meshes used by the simulator (see `assets/SOURCE.txt`, `assets/franka_emika_panda/LICENSE`) |
| `configs/` | `full.yaml` (paper run), `smoke.yaml` (short end-to-end test) |
| `scripts/` | stage runners (resumable) |
| `results/` | per-job and per-run result files, summaries, run logs |

Large binary files are not in this repository. They are on Zenodo: **DOI `<ZENODO-DOI>`**.

## Data on Zenodo

| Archive | Extract at repo root to get |
|---|---|
| `orchardworld_rollouts_main_and_counterfactual.tar` | `results/collect/shards/` (randomized-action rollouts, 340 tree-generation seeds) and `results/cf/shards/` (checkpoint-restored counterfactual rollouts) |
| `orchardworld_dataset_arrays.tar` | `results/dataset/*.npy` (training arrays built from the rollouts; rebuildable with `python -m owsim.dataset`) |
| `orchardworld_model_checkpoints.tar` | `results/runs/<variant>_s<seed>/best.pt` (trained world models: full, rgbd, rgb, rgb_state, no_action, tdmpc, vjepa; seeds 0-2) |

```bash
tar -xf <archive>.tar        # from the repository root
shasum -a 256 -c SHA256SUMS.txt
```

## Setup

Requires Linux, NVIDIA GPUs (paper: 4x H100 NVL, driver 570, CUDA 12.x), `uv` and Python 3.12.

```bash
bash setup.sh
source env.sh
python -m owsim.check
```

`requirements.txt` pins the physics stack (Newton 1.3.0, MuJoCo 3.8.1, MuJoCo-Warp 3.8.1, Warp 1.14.0 CUDA-12 build) and PyTorch.

## Reproducing the experiments

All runners are resumable: finished stages are skipped, so after an interruption run the same command again. Run them inside `tmux` or `nohup`.

```bash
bash scripts/run_all.sh full            # collect -> dataset -> train -> offline -> closed loop (about 9-10 h on 4 GPUs)
bash scripts/run_phase2.sh full         # TD-MPC2-style model and frozen V-JEPA 2 features
bash scripts/run_extras.sh              # Platt calibration, selective detect-and-pick baseline
bash scripts/run_theta_sweep.sh         # skip-threshold sweep (8 seeds)
bash scripts/run_seeds16.sh             # 16 unseen seeds
bash scripts/run_counterfactual.sh full # checkpoint/restore counterfactual extension
bash revision/run_revision.sh all       # E0, ESS, E6, E2, E3, E1, E5
bash scripts/run_all.sh smoke           # quick end-to-end test, writes results_smoke/
```

To skip data generation, extract the Zenodo archives and start from the `train` stage.
GPU physics is not bit-reproducible: expect small run-to-run differences in closed-loop metrics.
`revision/e1_noise.py` refuses to run if `owsim/decision_picker.py` differs from the audited version (MD5 check).

## Where the paper's numbers are

| Paper | Files |
|---|---|
| Table 2-3 (dataset, terminal events) | computed from the rollout shards (Zenodo); run metadata in `results/collect/`, `results/stages/` |
| Table 4 (offline prediction) | `results/offline/offline_metrics.json`, `results/runs/*/stats.json`, `results/revision/e0_baselines/`, `e2_oracle/`, `e3_ensemble/` |
| Table 5-6 (closed loop, matched throughput) | `results/report/summary.md` (8-seed summary), `results/control/jobs/`, `results/extras/`, `results/seeds16/` (`summary.md`, `stats.json`), `results/theta_sweep/` |
| Table 7-8 (ablation, robustness) | `results/control/jobs/`, `results/revision/e5_robustness/` |
| Table 9 (localization noise) | `results/revision/e1_noise/` |
| Table 10, Fig. 8 (counterfactual) | `results/cf/` (`analysis.json`, `summary.md`), `results/revision/e6_within_state/`, `results/revision/ess/` |

Each closed-loop job is one JSON file with its metrics; `summary.md` and `stats.json` in each folder aggregate them (bootstrap intervals over tree seeds, paired tests, Holm correction).

## Notes

- The scripts that typeset the paper's tables and draw its figures are not part of this repository. The numbers they used are in the per-job and per-run JSON files and the `summary.md` / `stats.json` files under `results/`.
- Checkpoints of the counterfactual-data model (`fullcf_s0-2`, Section III-F) are not in the Zenodo archive yet; retrain with `python -m owsim.counterfactual train`.
- Everything reported is simulation-only; the code does not control a physical robot.
- Not included here: raw RGB-D rollouts, dataset arrays and checkpoints (Zenodo), the E2 oracle-probe dataset and models (`python -m revision.e2_oracle build` recreates them), and the counterfactual dataset arrays (`python -m owsim.counterfactual dataset` recreates them from `results/cf/shards/`).
- No LICENSE file has been chosen for the code in `owsim/` and `revision/` yet; add one before publishing. `orchardbench/` is Apache-2.0.
