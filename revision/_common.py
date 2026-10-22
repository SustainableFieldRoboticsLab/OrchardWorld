from __future__ import annotations

import os
import sys
import time

import numpy as np

from owsim.common import RESULTS, rdir, logger, write_json, read_json, device_name, load_config

EVAL_SEEDS = list(range(900000, 900016))


def rev_dir(*parts):
    return rdir("revision", *parts)


def nominal_spec(cfg, seed, foliage=0.8, apples=35, light="nominal", frames=None, num_envs=None):
    from owsim.eval_control import light_for
    ev = cfg["eval"]
    light_dir, el = light_for(seed, light)
    return dict(seed=int(seed), num_envs=int(num_envs or ev["num_envs"]), foliage=float(foliage),
                apples=int(apples), light_dir=light_dir, light_el=el, shadows=(light == "altered"),
                frames=int(frames or ev["frames"]), device=device_name(),
                max_decisions=int(cfg["collect"].get("max_decisions", 4)))


def run_jobs(exp, jobs, run_one, worker_id, run_token):
    from owsim.eval_control import claim, job_metrics
    log = logger(f"{exp} w{worker_id}")
    jdir = rev_dir(exp, "jobs")
    cdir = rev_dir(exp, f"claims_{run_token}")
    for jb in jobs:
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res) or not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        log(f"job {jb['id']}")
        t0 = time.time()
        try:
            records, summary = run_one(jb)
        except Exception:
            import traceback
            log("job failed:\n" + traceback.format_exc())
            write_json(res, dict(job=jb, error="exception"))
            continue
        m = job_metrics(records, summary)
        write_json(res, dict(job=jb, metrics=m, summary=summary))
        log(f"  -> harvest {m['harvest_success']:.1f}% contact {m['contact_rate']:.1f}% "
            f"skip {m['skip_rate']:.1f}% fpm {m['throughput']:.3f} return {m['safe_return']:.3f} "
            f"status {summary['status']} ({time.time() - t0:.0f}s)")


def launch_workers(module, exp, n_jobs_todo, config="full", extra=()):
    from owsim.launch import gpus, spawn, wait_all
    cfg = load_config(config)
    log = logger(f"{exp} launch")
    G = gpus()
    n = min(len(G) * int(cfg["eval"]["workers_per_gpu"]), max(n_jobs_todo, 0))
    token = time.strftime("%Y%m%d_%H%M%S")
    log(f"{n_jobs_todo} jobs to do; {n} workers on GPUs {G}")
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-m", module, "worker", "--config", config,
               "--worker-id", str(i), "--run-token", token, *extra]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rev_dir("logs"), f"{exp}_w{i:02d}.log")))
        time.sleep(2.0)
    wait_all(procs, log)


def todo(exp, jobs):
    jdir = rev_dir(exp, "jobs")
    return [j for j in jobs if not os.path.exists(os.path.join(jdir, j["id"] + ".json"))]


def load_jobs(exp, prefix=""):
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "revision", exp, "jobs", prefix + "*.json"))):
        d = read_json(f)
        if d and "metrics" in d:
            out[d["job"]["id"]] = d
    return out


def fmt_ci(m, d=1):
    return f"{m[0]:.{d}f} [{m[1]:.{d}f}, {m[2]:.{d}f}]"


def fmt_paired(r, d=1):
    if r is None:
        return "n/a"
    return (f"{r['mean']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}], t={r['t']:+.2f}, p={r['p']:.3f}, "
            f"Wilcoxon p={r['p_wilcoxon']:.3f}, {r['pos']}/{r['n']} seeds > 0")


def per_seed_metrics(jobs_by_id, match):
    out = {}
    for d in jobs_by_id.values():
        if match(d["job"]):
            out[d["job"]["seed"]] = dict(d["metrics"], _status=d["summary"].get("status"))
    return out


def common_args(ap):
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--run-token", default="local")
    return ap


def rng_for(*keys):
    import zlib
    return np.random.default_rng([zlib.crc32(str(k).encode()) for k in keys])
