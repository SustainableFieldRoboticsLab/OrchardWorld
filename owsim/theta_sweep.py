from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from .common import RESULTS, load_config, rdir, logger, write_json, read_json, device_name

THETAS = [-0.3, -0.15, -0.05, 0.05, 0.15, 0.25, 0.4]
VARIANT_RUN = "full_s0"


def build_jobs(cfg):
    from .eval_control import NOMINAL
    ev = cfg["eval"]
    base, nm = int(ev["seed_base"]), int(ev["main_seeds"])
    jobs = []
    for i in range(nm):
        for th in THETAS:
            c = dict(NOMINAL)
            jobs.append(dict(id=f"theta__{th:+.2f}__{base + i}", theta=th, seed=base + i, **c))
    return jobs


def worker(args):
    from .eval_control import light_for, job_metrics, claim
    from .policies import WorldModelPolicy
    from .rollout import run_batch
    cfg = load_config(args.config)
    ev = cfg["eval"]
    log = logger(f"theta w{args.worker_id}")
    jdir = rdir("theta_sweep", "jobs")
    cdir = rdir("theta_sweep", f"claims_{args.run_token}")
    dev = "cuda" if device_name() == "cuda" else "cpu"
    policy = WorldModelPolicy(os.path.join(RESULTS, "runs", VARIANT_RUN), device=dev,
                              allow_skip=True, name="wm_full")
    theta_val = policy.theta
    for jb in build_jobs(cfg):
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res):
            continue
        if args.deadline and time.time() > args.deadline - 60:
            log("deadline reached; leaving remaining jobs")
            break
        if not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        policy.theta = float(jb["theta"])
        policy.latency_ms = []
        light, el = light_for(jb["seed"], jb["light"])
        spec = dict(seed=jb["seed"], num_envs=int(ev["num_envs"]), foliage=jb["foliage"],
                    apples=jb["apples"], light_dir=light, light_el=el, shadows=False,
                    frames=int(ev["frames"]), device=device_name(),
                    max_decisions=int(cfg["collect"].get("max_decisions", 4)))
        log(f"job {jb['id']}")
        try:
            records, summary = run_batch(spec, policy, log=log, deadline=None, keep_obs=0)
        except Exception:
            import traceback
            log("job failed:\n" + traceback.format_exc())
            write_json(res, dict(job=jb, error="exception"))
            continue
        m = job_metrics(records, summary)
        write_json(res, dict(job=jb, metrics=m, summary=summary, theta_validation=theta_val))
        log(f"  -> theta {jb['theta']:+.2f} harvest {m['harvest_success']:.1f}% contact "
            f"{m['contact_rate']:.1f}% skip {m['skip_rate']:.1f}% fruit/robot/min "
            f"{m['throughput']:.3f} return {m['safe_return']:.3f} ({summary['wall_s']:.0f}s)")


def launch(args):
    from .launch import gpus, spawn, wait_all
    cfg = load_config(args.config)
    log = logger("theta launch")
    G = gpus()
    n = len(G) * int(cfg["eval"]["workers_per_gpu"])
    deadline = time.time() + 3600.0 * args.hours
    token = time.strftime("%Y%m%d_%H%M%S")
    todo = [j for j in build_jobs(cfg)
            if not os.path.exists(os.path.join(rdir("theta_sweep", "jobs"), j["id"] + ".json"))]
    log(f"{len(todo)} theta-sweep jobs to do; {n} workers on GPUs {G}; "
        f"deadline {time.ctime(deadline)}")
    procs = []
    for i in range(min(n, len(todo))):
        cmd = [sys.executable, "-m", "owsim.theta_sweep", "worker", "--config", args.config,
               "--worker-id", str(i), "--run-token", token, "--deadline", str(deadline)]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rdir("logs"), f"theta_w{i:02d}.log")))
        time.sleep(2.0)
    wait_all(procs, log)
    report(args)


def _load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = read_json(f)
        if d and "metrics" in d:
            out.append(d)
    return out


def _agg(rows):
    k = ["harvest_success", "contact_rate", "damage_rate", "skip_rate", "throughput", "safe_return"]
    r = {q: float(np.nanmean([d["metrics"][q] for d in rows])) if rows else float("nan") for q in k}
    r["throughput_sd"] = float(np.nanstd([d["metrics"]["throughput"] for d in rows])) if rows else float("nan")
    r["safe_return_sd"] = float(np.nanstd([d["metrics"]["safe_return"] for d in rows])) if rows else float("nan")
    r["n"] = len(rows)
    r["truncated"] = sum(d["summary"].get("status") not in ("ok", "all_done", "deadline") for d in rows)
    return r


def report(args):
    log = logger("theta report")
    out = rdir("theta_sweep")
    jobs = os.path.join(RESULTS, "control", "jobs")
    pts = []
    for th in THETAS:
        rows = [d for d in _load(os.path.join(out, "jobs", "theta__*.json"))
                if abs(d["job"]["theta"] - th) < 1e-9]
        pts.append((th, _agg(rows)))
    noskip = _agg(_load(os.path.join(jobs, "main__nominal__wm_full_noskip__*.json")))
    ref = {name: _agg(_load(os.path.join(jobs, f"main__nominal__{c}__*.json")))
           for c, name in [("detect_and_pick", "Detect-and-pick"),
                           ("reactive_retry", "Reactive retry"),
                           ("wm_full", "OrchardWorld-Sim (validation θ)")]}
    theta_val = None
    for d in _load(os.path.join(out, "jobs", "theta__*.json")):
        theta_val = d.get("theta_validation")
        break

    S = ["# Skip-threshold (θ) sweep — OrchardWorld-Sim (full_s0), nominal, 8 seeds", "",
         f"Validation-selected θ = {theta_val}" if theta_val is not None else "", "",
         "| θ | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |",
         "|---|---|---|---|---|---|---|---|---|"]
    row = lambda name, a: (f"| {name} | {a['n']} | {a['harvest_success']:.1f} | {a['contact_rate']:.1f} | "
                           f"{a['damage_rate']:.1f} | {a['skip_rate']:.1f} | {a['throughput']:.3f} | "
                           f"{a['safe_return']:.3f} | {a['truncated']} |")
    S.append(row("−∞ (no skip)", noskip))
    S += [row(f"{th:+.2f}", a) for th, a in pts]
    S += ["", "Reference controllers (main comparison, same seeds):", "",
          "| controller | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |",
          "|---|---|---|---|---|---|---|---|---|"] + [row(n, a) for n, a in ref.items()]
    rep = [d for d in _load(os.path.join(out, "jobs", "theta__*.json"))
           if theta_val is not None and abs(d["job"]["theta"] - theta_val) < 1e-9]
    if rep:
        main = {d["job"]["seed"]: d["metrics"]["safe_return"]
                for d in _load(os.path.join(jobs, "main__nominal__wm_full__*.json"))}
        diffs = [d["metrics"]["safe_return"] - main[d["job"]["seed"]] for d in rep if d["job"]["seed"] in main]
        if diffs:
            S += ["", f"Reproducibility check: re-run at the validation θ vs. the main comparison, "
                      f"safe-return difference per seed mean {np.mean(diffs):+.3f}, "
                      f"max |diff| {np.max(np.abs(diffs)):.3f} (0 = bit-identical simulation)."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(S) + "\n")
    log(f"wrote {out}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["launch", "worker", "report", "list"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--run-token", default="local")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--hours", type=float, default=1000.0)
    args = ap.parse_args()
    if args.mode == "list":
        J = build_jobs(load_config(args.config))
        print(len(J), "jobs; thetas", THETAS, "; first", J[0]["id"], "last", J[-1]["id"])
    else:
        {"launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
