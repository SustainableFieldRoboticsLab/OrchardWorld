from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from .common import load_config, split_of, rdir, logger, write_json, device_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=None, help="unix time to stop")
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["collect"]
    wid, nw = args.worker_id, args.n_workers
    log = logger(f"collect w{wid}")
    deadline = args.deadline or (time.time() + 3600.0 * float(cc["hours"]))
    out = rdir("collect", "shards")
    hb_path = os.path.join(rdir("collect", "heartbeat"), f"w{wid:02d}.json")

    from .rollout import run_batch, save_records
    from .policies import RandomPolicy

    rng = np.random.default_rng(10007 + wid)
    policy = RandomPolicy(rng, probs=cc.get("action_probs", (0.2, 0.2, 0.2, 0.15, 0.15, 0.1)))
    tot = dict(worker=wid, batches=0, decisions=0, env_steps=0, sim_s=0.0, wall_s=0.0,
               errors=0, started=time.time(), deadline=deadline, state="running")
    est = None
    k = -1
    done_batches = 0
    while done_batches < args.max_batches:
        k += 1
        seed = int(cc["seed_base"]) + k * nw + wid
        shard = os.path.join(out, f"b{seed}.npz")
        if os.path.exists(shard):
            continue
        remaining = deadline - time.time()
        if remaining < 90 or (est is not None and remaining < 0.35 * est):
            break
        fol = float(rng.choice(cc["foliage"]))
        app = int(rng.choice(cc["apples"]))
        az = rng.uniform(0, 2 * math.pi)
        el = math.radians(rng.uniform(35, 80))
        light = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
        spec = dict(seed=seed, split=split_of(seed), num_envs=int(cc["num_envs"]),
                    foliage=fol, apples=app, light_dir=light, light_el=math.degrees(el),
                    shadows=False, frames=int(cc["frames"]), device=device_name(),
                    max_decisions=int(cc.get("max_decisions", 4)))

        def hb(d, _seed=seed):
            write_json(hb_path, dict(tot, current=dict(seed=_seed, **d), t=time.time()))

        spec["heartbeat"] = hb
        log(f"batch seed={seed} split={spec['split']} foliage={fol} apples={app} "
            f"envs={spec['num_envs']} frames={spec['frames']}")
        t0 = time.time()
        try:
            records, summary = run_batch(spec, policy, log=log, deadline=deadline)
        except Exception as ex:
            import traceback
            log("batch failed:\n" + traceback.format_exc())
            tot["errors"] += 1
            if tot["errors"] > 20 and tot["batches"] == 0:
                raise SystemExit(f"too many failures: {ex}")
            continue
        save_records(shard, records, summary)
        dt = time.time() - t0
        est = dt if est is None else 0.7 * est + 0.3 * dt
        done_batches += 1
        tot["batches"] += 1
        tot["decisions"] += len(records)
        tot["env_steps"] += summary["env_steps"]
        tot["sim_s"] += summary["sim_s"]
        tot["wall_s"] += dt
        write_json(hb_path, dict(tot, last=dict(seed=seed, status=summary["status"],
                                                decisions=len(records), fps=summary["fps"],
                                                build_s=summary["build_s"]), t=time.time()))
        log(f"  -> {len(records)} decisions, status={summary['status']}, "
            f"{summary['fps']:.1f} batch-fps, build {summary['build_s']:.0f}s, wall {dt:.0f}s, "
            f"ms/frame {summary['ms_per_frame']}")
    tot["state"] = "finished"
    write_json(hb_path, dict(tot, t=time.time()))
    log(f"finished: {tot['batches']} batches, {tot['decisions']} decisions")


if __name__ == "__main__":
    main()
