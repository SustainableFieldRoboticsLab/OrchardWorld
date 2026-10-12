# OrchardWorld

Physics-grounded, action-conditioned world models for apple-harvesting robots in digital orchards.

This repository contains the code, configurations, run logs and generated results of the study
*OrchardWorld: Physics-Grounded Action-Conditioned World Models for Apple-Harvesting Robots in
Digital Orchards* (Dr. R. Sapkota, Dr. K. I. Roumeliotis). All results are from simulation only.

## What the code does

1. **Rollout generation.** OrchardBench's autonomous apple picker (NVIDIA Newton + MuJoCo-Warp)
   gets a decision point in front of every target fruit. A randomized behaviour policy chooses one of
   six actions: `pick-center`, `pick-left`, `pick-right`, `reposition-left`, `reposition-right`,
   `skip`. The physically simulated outcome is logged: detachment, harvest success, branch contact,
   damage, target visibility and execution time. 16 simulator processes (4 per GPU) each step 16
   domain-randomized trees in one batched solve.
2. **World model.** A 15.4 M-parameter latent-dynamics model (CNN tokenizers + Transformer encoder,
   FiLM action-conditioned transition, outcome heads, future-depth decoder). It is trained on the
   interventional dataset, split by tree seed. Variants: ablations (RGB, RGB-D, RGB + state, no
   action), a TD-MPC2-style model, and one with frozen V-JEPA 2 features.
3. **Safety-aware action selection.** Each candidate action is scored by its predicted
   safety-weighted reward `r = harvest - 0.25 contact - damage - 0.01 tau`, and the fruit is skipped
   below a threshold θ. θ is selected off-policy (self-normalized IPS) on validation trees, with
   optional Platt calibration.
4. **Closed-loop evaluation** on unseen procedural trees against these controllers:
   - detect-and-pick;
   - reactive retry;
   - a selective detect-and-pick control;
   - ablations;
   - robustness shifts;
   - a foliage sweep;
   - a skip-threshold sweep;
   - a 16-seed extension.

## Requirements

- Linux x86_64, NVIDIA GPU(s) with a driver supporting CUDA 12.8 or newer. The study used 4x NVIDIA
  H100 NVL (driver 570); the smoke test runs on a single GPU.
- Python 3.12, [uv](https://docs.astral.sh/uv/), tmux.
- Pinned stack: Newton 1.3.0, MuJoCo 3.8.1, MuJoCo-Warp 3.8.1, Warp 1.14.0 (CUDA 12 wheel),
  PyTorch 2.13.0 (CUDA 12.9 build, selected through `UV_TORCH_BACKEND=cu129`).
- Phase 2 downloads `facebook/vjepa2-vitl-fpc64-256` from Hugging Face (about 1.3 GB). No access
  token is needed.

## Setup

```bash
git clone https://github.com/SustainableFieldRoboticsLab/OrchardWorld.git
cd OrchardWorld
bash setup.sh
```

`env.sh` keeps all caches (uv, Hugging Face, Warp kernels) under `.cache_shared/` inside the
repository. To use another location, set `SHARED`.

## Running the experiments

Every stage is resumable: after an interruption, run the same command again. Long runs should be
started inside tmux, so they keep running after you log out.

| Step | Command | Time on 4x H100 |
|---|---|---|
| Smoke test (toy scale, writes `results_smoke/`) | `bash scripts/start_tmux.sh smoke` | ~30 min |
| Phase 1: collect, dataset, train, offline evaluation, closed-loop control | `bash scripts/start_tmux.sh full` | ~8 h |
| Phase 2: TD-MPC2-style and V-JEPA 2 models | `bash scripts/start_tmux.sh phase2` | ~1.1 h |
| Calibration and selective detect-and-pick baseline | `bash scripts/start_tmux.sh extras` | ~1 h |
| Skip-threshold sweep | `tmux new -d -s orchard_theta "bash scripts/run_theta_sweep.sh"` | ~1 h |
| 16-seed trade-off extension | `tmux new -d -s orchard_s16 "bash scripts/run_seeds16.sh"` | ~2 h |

`start_tmux.sh` opens a session `orchard` with three windows: `run`, `status` and `gpu`.

- Attach with `tmux attach -t orchard`.
- Detach with `Ctrl-b d`.
- Show progress at any time with `bash scripts/status.sh full --once`.

Configurations are in `configs/full.yaml` and `configs/smoke.yaml`. Tree seeds `100000+` are used
for data collection (seed mod 10: 0 = test, 1 = validation, otherwise training), and seeds
`900000+` for closed-loop evaluation. Models are trained with seeds 0, 1 and 2.

The whole study took about 12.5 h of wall-clock time on four H100 GPUs (at most 50 GPU-hours).

Summaries can be regenerated from the job files at any time:

```bash
source env.sh
python -m owsim.theta_sweep report
python -m owsim.extras report
python -m owsim.seeds16 report
```

## Repository layout

```
owsim/        experiment code
  simenv.py, decision_picker.py, rollout.py   batched simulation and the decision layer
  collect.py, dataset.py                      randomized-action data collection and dataset
  model.py, train.py, vjepa.py, features.py   world models and training
  eval_offline.py, eval_control.py, policies.py  offline metrics, closed loop and controllers
  extras.py, theta_sweep.py, seeds16.py       calibration, selective baseline, θ sweep, 16 seeds
  launch.py, check.py, status.py, get_model.py   multi-GPU launcher and utilities
scripts/      pipeline and tmux entry points
configs/      full and smoke configurations
orchardbench/ OrchardBench simulator (treesim), unmodified, Apache-2.0
assets/       Franka FR3 description from newton-assets, Apache-2.0
results/      results of the study (see below)
```

## Results included

| Path | Content |
|---|---|
| `results/summary.md` | offline prediction and main closed-loop comparison |
| `results/dataset/table.csv` | all 21,943 logged decisions (action, behaviour propensity, outcomes, tree seed, split, conditions); images are not included |
| `results/dataset/stats.json` | dataset statistics |
| `results/offline/` | offline metrics and test-set predictions of every model and seed |
| `results/control/jobs/` | per-episode metrics of all closed-loop jobs (main comparison, ablations, robustness, foliage sweep, phase 2) |
| `results/control/qualitative/` | decision records with observations from the main-comparison episodes of the two world-model planners |
| `results/extras/` | calibration fits, selective-baseline fit, closed-loop jobs and summary |
| `results/theta_sweep/` | θ-sweep jobs and summary |
| `results/seeds16/` | 16-seed jobs, paired statistics (`stats.json`) and summary |
| `results/runs/<variant>_s<seed>/` | training history, normalization, selected θ and calibration of every trained model |
| `results/logs/` | run logs |
| `results/stages/timing.json` | wall-clock time per pipeline stage |

The raw image dataset (~4 GB) and the model checkpoints (~60 MB each) are not stored in this
repository; both are regenerated by the pipeline.

GPU physics is not bit-reproducible. Re-running a closed-loop episode on the same seed gives
statistically equivalent results, not identical ones: in our runs, the per-episode safe return
varied by about ±0.1.

## Licenses

`orchardbench/` (OrchardBench, H. Munn) and `assets/franka_emika_panda/` (newton-assets) are
distributed under the Apache License 2.0. Their license files are kept in place.
