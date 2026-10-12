from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from .common import RESULTS, load_config, rdir, logger, write_json, device_name, split_of
from .decision_picker import PICKS, REPOS, SKIP

CONTROLLERS_MAIN = ["detect_and_pick", "reactive_retry", "wm_full", "wm_rgbd",
                    "wm_rgb_state", "wm_no_action", "wm_full_noskip"]
CONTROLLERS_CORE = ["detect_and_pick", "reactive_retry", "wm_full"]
NOMINAL = dict(foliage=0.8, apples=35, light="nominal")
SHIFTS = {
    "nominal": dict(),
    "foliage_low": dict(foliage=0.3),
    "foliage_high": dict(foliage=1.3),
    "fruit_sparse": dict(apples=20),
    "fruit_dense": dict(apples=55),
    "light_altered": dict(light="altered"),
    "trees_seen": dict(seen=True),
}
SWEEP_FOLIAGE = [0.2, 0.6, 1.0, 1.4]
CONTROLLERS_P2_MAIN = ["wm_tdmpc", "wm_vjepa"]
CONTROLLERS_P2_CORE = ["wm_vjepa"]


def seen_seeds(cfg, n):
    base = int(cfg["collect"]["seed_base"])
    out, s = [], base
    while len(out) < n:
        if split_of(s) == "train":
            out.append(s)
        s += 1
    return out


def build_jobs(cfg, phase=1):
    ev = cfg["eval"]
    main_ctrls = CONTROLLERS_MAIN if phase == 1 else CONTROLLERS_P2_MAIN
    core_ctrls = CONTROLLERS_CORE if phase == 1 else CONTROLLERS_P2_CORE
    base = int(ev["seed_base"])
    nm, ns = int(ev["main_seeds"]), int(ev["sweep_seeds"])
    jobs = []

    def job(group, cond, ctrl, seed, **kw):
        c = dict(NOMINAL)
        c.update(kw)
        jid = f"{group}__{cond}__{ctrl}__{seed}"
        jobs.append(dict(id=jid, group=group, cond=cond, controller=ctrl, seed=int(seed), **c))

    for i in range(nm):
        for ctrl in main_ctrls:
            job("main", "nominal", ctrl, base + i)
    seen = seen_seeds(cfg, ns)
    for i in range(ns):
        for cond, kw in SHIFTS.items():
            kw = dict(kw)
            seed = seen[i] if kw.pop("seen", False) else base + 200 + i
            for ctrl in core_ctrls:
                job("shift", cond, ctrl, seed, **kw)
    for i in range(ns):
        for f in SWEEP_FOLIAGE:
            for ctrl in core_ctrls:
                job("foliage", f"f{f:.1f}", ctrl, base + 100 + i, foliage=f)
    return jobs


def light_for(seed, kind):
    rng = np.random.default_rng(int(seed) * 7 + 3)
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(8, 18) if kind == "altered" else rng.uniform(35, 80))
    return (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)), math.degrees(el)


def job_metrics(records, summary):
    envs = summary["envs"]
    minutes = max(summary["sim_s"] / 60.0, 1e-6)
    act = np.array([r["action"] for r in records], dtype=int) if records else np.zeros(0, int)
    get = lambda k: np.array([float(r[k]) for r in records]) if records else np.zeros(0)
    pick = np.isin(act, PICKS)
    execd = act != SKIP
    succ, det, con, dmg, rew = get("success"), get("detach"), get("contact"), get("damage"), get("reward")
    harvested = float(succ[pick].sum()) if len(act) else 0.0
    m = dict(
        envs=envs, sim_minutes=minutes, decisions=int(len(act)),
        targets=int(sum(e["targets"] for e in summary["per_env"])),
        pick_attempts=int(pick.sum()), executed=int(execd.sum()),
        harvested=harvested, detached=float(det[pick].sum()) if len(act) else 0.0,
        contacts=float(con[execd].sum()) if len(act) else 0.0,
        damages=float(dmg[execd].sum()) if len(act) else 0.0,
        skips=int((act == SKIP).sum()), repositions=int(np.isin(act, REPOS).sum()),
        return_sum=float(rew.sum()) if len(act) else 0.0,
        branches_broken_total=int(sum(e["branches_broken"] for e in summary["per_env"])),
    )
    pa, ex = max(m["pick_attempts"], 1), max(m["executed"], 1)
    m.update(
        harvest_success=100.0 * harvested / pa if m["pick_attempts"] else float("nan"),
        detachment=100.0 * m["detached"] / pa if m["pick_attempts"] else float("nan"),
        contact_rate=100.0 * m["contacts"] / ex if m["executed"] else float("nan"),
        damage_rate=100.0 * m["damages"] / ex if m["executed"] else float("nan"),
        actions_per_fruit=(m["executed"] / harvested) if harvested > 0 else float("nan"),
        throughput=harvested / (envs * minutes),
        safe_return=m["return_sum"] / (envs * minutes),
        skip_rate=100.0 * m["skips"] / max(m["decisions"], 1),
        repos_rate=100.0 * m["repositions"] / max(m["decisions"], 1),
        fruit_per_robot=harvested / envs,
    )
    return m


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--run-token", default="local")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev = cfg["eval"]
    jobs = build_jobs(cfg, args.phase)
    if args.list:
        from collections import Counter
        print(len(jobs), "jobs:", dict(Counter(j["group"] for j in jobs)))
        return
    log = logger(f"eval w{args.worker_id}")
    deadline = args.deadline or (time.time() + 3600.0 * float(ev["hours"]))
    jdir = rdir("control", "jobs")
    cdir = rdir("control", f"claims_{args.run_token}")
    hb_path = os.path.join(rdir("control", "heartbeat"), f"w{args.worker_id:02d}.json")
    from .rollout import run_batch, save_records
    from .policies import make_policy
    pol_cache = {}
    n_done = 0
    for jb in jobs:
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res):
            continue
        if time.time() > deadline - 60:
            log("deadline reached; leaving remaining jobs")
            break
        if not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        ctrl = jb["controller"]
        try:
            if ctrl not in pol_cache:
                pol_cache[ctrl] = make_policy(ctrl, runs_root=os.path.join(RESULTS, "runs"),
                                              device="cuda" if device_name() == "cuda" else "cpu")
            policy = pol_cache[ctrl]
            if hasattr(policy, "latency_ms"):
                policy.latency_ms = []
        except Exception as ex:
            log(f"cannot build controller {ctrl}: {ex}")
            write_json(res, dict(job=jb, error=f"controller unavailable: {ex}"))
            continue
        light, el = light_for(jb["seed"], jb["light"])
        spec = dict(seed=jb["seed"], num_envs=int(ev["num_envs"]), foliage=jb["foliage"],
                    apples=jb["apples"], light_dir=light, light_el=el,
                    shadows=jb["light"] == "altered", frames=int(ev["frames"]),
                    device=device_name(), max_decisions=int(cfg["collect"].get("max_decisions", 4)))

        def hb(d, _jid=jb["id"]):
            write_json(hb_path, dict(worker=args.worker_id, job=_jid, done=n_done, t=time.time(), **d))

        spec["heartbeat"] = hb
        log(f"job {jb['id']}")
        keep = (int(ev.get("keep_obs", 16))
                if (ctrl in ("wm_full", "wm_vjepa") and jb["group"] == "main") else 0)
        try:
            records, summary = run_batch(spec, policy, log=log, deadline=None, keep_obs=keep)
        except Exception:
            import traceback
            log("job failed:\n" + traceback.format_exc())
            write_json(res, dict(job=jb, error="exception"))
            continue
        m = job_metrics(records, summary)
        if keep:
            save_records(os.path.join(rdir("control", "qualitative"), jb["id"] + ".npz"),
                         records, summary)
        write_json(res, dict(job=jb, metrics=m, summary=summary))
        n_done += 1
        log(f"  -> harvest {m['harvest_success']:.1f}% contact {m['contact_rate']:.1f}% "
            f"skip {m['skip_rate']:.1f}% return {m['safe_return']:.3f} ({summary['wall_s']:.0f}s)")
    write_json(hb_path, dict(worker=args.worker_id, state="finished", done=n_done, t=time.time()))


if __name__ == "__main__":
    main()
