from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np

from .simenv import Batch
from .decision_picker import DecisionPicker, N_ACTIONS

SCALARS = ["env", "target_uid", "dec_idx", "action", "behavior_p", "skipped", "detach",
           "success", "contact", "damage", "visible_after", "tau", "target_disp",
           "wood_disp", "pull_max", "arm_wood_frames", "base_wood_frames",
           "arm_apple_frames", "branches_broken", "collateral_drops", "fail_code",
           "reward", "start_frame", "end_frame"]
PRIV = ["occlusion", "in_fov", "gt_z", "stem_strength"]
OPTIONAL = []

_IK = None


def _shared_ik():
    global _IK
    if _IK is None:
        from treesim.picker import ArmIK
        from treesim import robot as _robot
        _IK = ArmIK(list(_robot._ARM_HOME.values()))
    return _IK


def run_batch(spec: dict, policy, log=print, deadline: float | None = None,
              keep_obs: int | None = None, picker_factory=None):
    t_build = time.time()
    B = Batch(spec["seed"], spec["num_envs"], spec["foliage"], spec["apples"],
              light_dir=spec["light_dir"], shadows=spec["shadows"],
              device=spec.get("device", "cuda"))
    build_s = time.time() - t_build
    records = []
    rng = np.random.default_rng(spec["seed"] + 17)
    ik = _shared_ik()
    if picker_factory is not None:
        pickers = [picker_factory(B, e, records.append, ik, rng) for e in range(B.num_envs)]
    else:
        pickers = [DecisionPicker(B, e, policy, records.append, ik, rng,
                                  max_decisions=spec.get("max_decisions", 4))
                   for e in range(B.num_envs)]
    status = "ok"
    t0 = time.time()
    frames = int(spec["frames"])
    hb = spec.get("heartbeat")
    tm = dict(physics=0.0, camera=0.0, probe=0.0, control=0.0)
    clk = time.perf_counter
    try:
        for f in range(frames):
            t1 = clk()
            B.step()
            t2 = clk()
            B.cam.update(B.sim.state_0)
            t3 = clk()
            active = any(p.act is not None and p.act.get("end_frame") is None for p in pickers)
            if active and f % 2 == 0:
                B.probe(True, True)
            else:
                B.last_contacts, B.pull_np = None, None
            t4 = clk()
            for p in pickers:
                p.update()
            t5 = clk()
            tm["physics"] += t2 - t1
            tm["camera"] += t3 - t2
            tm["probe"] += t4 - t3
            tm["control"] += t5 - t4
            if f % 120 == 119:
                if not B.finite():
                    status = "nan"
                    log(f"[rollout] seed {spec['seed']}: non-finite state at frame {f}, stopping batch")
                    break
                if hb is not None:
                    hb(dict(frame=f + 1, frames=frames, decisions=len(records),
                            fps=(f + 1) / max(time.time() - t0, 1e-6)))
            if deadline is not None and time.time() > deadline:
                status = "deadline"
                break
            if all(p.done for p in pickers):
                status = "all_done"
                break
    except Exception:
        status = "error"
        log("[rollout] exception:\n" + traceback.format_exc())
    wall = time.time() - t0
    sim_frames = B.frame
    summary = dict(
        spec={k: v for k, v in spec.items() if k != "heartbeat"}, status=status,
        build_s=round(build_s, 2), wall_s=round(wall, 2), sim_frames=sim_frames,
        sim_s=sim_frames / 60.0, fps=sim_frames / max(wall, 1e-6),
        env_steps=sim_frames * B.num_envs, n_records=len(records),
        per_env=[dict(targets=p.n_targets, decisions=p.n_decisions,
                      approach_fail=p.n_approach_fail, phantom=p.n_phantom,
                      branches_broken=B.broken_env(p.env),
                      apples_detached=int(B.detached_env(p.env).sum()))
                 for p in pickers],
        bodies=int(B.tm.model.body_count), envs=B.num_envs,
        ms_per_frame={k: round(1e3 * v / max(sim_frames, 1), 2) for k, v in tm.items()})
    if hasattr(policy, "latency_ms") and policy.latency_ms:
        summary["policy_latency_ms"] = float(np.median(policy.latency_ms))
    del B
    if keep_obs is not None:
        for i, r in enumerate(records):
            if i >= keep_obs:
                r["obs_pre"] = None
                r["obs_post"] = None
    return records, summary


def save_records(path: str, records: list, summary: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(records)
    out = {}
    for k in SCALARS:
        out[k] = np.asarray([r[k] for r in records], dtype=np.float32 if k in (
            "behavior_p", "tau", "target_disp", "wood_disp", "pull_max", "reward") else np.int32)
    for k in OPTIONAL:
        if records and all(k in r for r in records):
            out[k] = np.asarray([r[k] for r in records], dtype=np.int32)
    sc = np.full((n, N_ACTIONS), np.nan, dtype=np.float32)
    for i, r in enumerate(records):
        if r.get("scores") is not None:
            sc[i] = r["scores"]
    out["scores"] = sc
    with_obs = [i for i, r in enumerate(records) if r["obs_pre"] is not None]
    out["obs_index"] = np.asarray(with_obs, dtype=np.int32)
    for tag in ("pre", "post"):
        obs = [records[i]["obs_" + tag] for i in with_obs]
        if not obs:
            continue
        out[f"{tag}_g_rgb"] = np.stack([o["g_rgb"] for o in obs])
        out[f"{tag}_g_dep"] = np.stack([o["g_dep"] for o in obs])
        out[f"{tag}_c_rgb"] = np.stack([o["c_rgb"] for o in obs])
        out[f"{tag}_c_dep"] = np.stack([o["c_dep"] for o in obs])
        out[f"{tag}_state"] = np.stack([o["state"] for o in obs]).astype(np.float32)
        out[f"{tag}_det"] = np.asarray([o["det"] for o in obs], dtype=np.int8)
        out[f"{tag}_uv"] = np.asarray([o["uv"] for o in obs], dtype=np.float32)
        out[f"{tag}_pose_ok"] = np.asarray([o["pose_ok"] for o in obs], dtype=np.int8)
        for k in PRIV:
            out[f"{tag}_{k}"] = np.asarray([o["priv"][k] for o in obs], dtype=np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
