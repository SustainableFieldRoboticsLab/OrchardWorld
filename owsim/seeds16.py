from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from .common import RESULTS, load_config, rdir, logger, write_json, read_json, device_name

NEW_SEEDS = list(range(900008, 900016))
OLD_SEEDS = list(range(900000, 900008))
N_BOOT = 10000


def _rs_thetas():
    P = read_json(os.path.join(RESULTS, "extras", "reactive_select.json"))
    if P is None:
        raise SystemExit("results/extras/reactive_select.json missing - run phase 3 extras first")
    return [float(t) for t in P["sweep_thetas"]], float(P["theta"])


def configs():
    from .theta_sweep import THETAS
    rs, _ = _rs_thetas()
    C = [(f"wm__{th:+.2f}", "wm_full", th) for th in THETAS]
    C.append(("wm__noskip", "wm_full_noskip", None))
    C += [(f"rs__{th:+.3f}", "reactive_select", th) for th in rs]
    C += [("detect_and_pick", "detect_and_pick", None), ("reactive_retry", "reactive_retry", None),
          ("wm_cal", "wm_full_cal", None)]
    return C


def old_pattern(key):
    J, T, E = (os.path.join(RESULTS, d, "jobs") for d in ("control", "theta_sweep", "extras"))
    if key.startswith("wm__") and key != "wm__noskip":
        return os.path.join(T, f"theta__{key[4:]}__*.json")
    if key == "wm__noskip":
        return os.path.join(J, "main__nominal__wm_full_noskip__*.json")
    if key.startswith("rs__"):
        return os.path.join(E, f"extras__reactive_select__{key[4:]}__*.json")
    if key == "wm_cal":
        return os.path.join(E, "extras__wm_full_cal__val__*.json")
    return os.path.join(J, f"main__nominal__{key}__*.json")


def build_jobs():
    from .eval_control import NOMINAL
    jobs = []
    for s in NEW_SEEDS:
        for key, ctrl, th in configs():
            jobs.append(dict(id=f"s16__{key}__{s}", key=key, controller=ctrl, theta=th, seed=s, **NOMINAL))
    return jobs


def worker(args):
    from .eval_control import light_for, job_metrics, claim
    from .policies import make_policy
    from .rollout import run_batch
    cfg = load_config(args.config)
    ev = cfg["eval"]
    log = logger(f"s16 w{args.worker_id}")
    jdir = rdir("seeds16", "jobs")
    cdir = rdir("seeds16", f"claims_{args.run_token}")
    dev = "cuda" if device_name() == "cuda" else "cpu"
    cache, default_theta = {}, {}
    for jb in build_jobs():
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res):
            continue
        if not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        c = jb["controller"]
        if c not in cache:
            cache[c] = make_policy(c, runs_root=os.path.join(RESULTS, "runs"), device=dev)
            default_theta[c] = getattr(cache[c], "theta", None)
        pol = cache[c]
        if hasattr(pol, "theta"):
            pol.theta = float(jb["theta"]) if jb["theta"] is not None else default_theta[c]
        if hasattr(pol, "latency_ms"):
            pol.latency_ms = []
        light, el = light_for(jb["seed"], jb["light"])
        spec = dict(seed=jb["seed"], num_envs=int(ev["num_envs"]), foliage=jb["foliage"],
                    apples=jb["apples"], light_dir=light, light_el=el, shadows=False,
                    frames=int(ev["frames"]), device=device_name(),
                    max_decisions=int(cfg["collect"].get("max_decisions", 4)))
        log(f"job {jb['id']}")
        try:
            records, summary = run_batch(spec, pol, log=log, deadline=None, keep_obs=0)
        except Exception:
            import traceback
            log("job failed:\n" + traceback.format_exc())
            write_json(res, dict(job=jb, error="exception"))
            continue
        m = job_metrics(records, summary)
        write_json(res, dict(job=dict(jb, theta_used=getattr(pol, "theta", None)), metrics=m, summary=summary))
        log(f"  -> harvest {m['harvest_success']:.1f}% contact {m['contact_rate']:.1f}% "
            f"skip {m['skip_rate']:.1f}% fruit/robot/min {m['throughput']:.3f} "
            f"return {m['safe_return']:.3f} ({summary['wall_s']:.0f}s)")


def launch(args):
    from .launch import gpus, spawn, wait_all
    cfg = load_config(args.config)
    log = logger("s16 launch")
    G = gpus()
    n = len(G) * int(cfg["eval"]["workers_per_gpu"])
    token = time.strftime("%Y%m%d_%H%M%S")
    todo = [j for j in build_jobs() if not os.path.exists(os.path.join(rdir("seeds16", "jobs"), j["id"] + ".json"))]
    log(f"{len(todo)} jobs to do on seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}; {n} workers on GPUs {G}; no time limit")
    procs = []
    for i in range(min(n, len(todo))):
        cmd = [sys.executable, "-m", "owsim.seeds16", "worker", "--config", args.config,
               "--worker-id", str(i), "--run-token", token]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rdir("logs"), f"s16_w{i:02d}.log")))
        time.sleep(2.0)
    wait_all(procs, log)
    report(args)


KEYS = ["harvest_success", "contact_rate", "damage_rate", "skip_rate", "throughput", "safe_return"]


def per_seed(key):
    out = {}
    for f in sorted(glob.glob(old_pattern(key))):
        d = read_json(f)
        if d and "metrics" in d and d["job"]["seed"] in OLD_SEEDS:
            out[d["job"]["seed"]] = dict(d["metrics"], _status=d["summary"].get("status"))
    for f in sorted(glob.glob(os.path.join(RESULTS, "seeds16", "jobs", f"s16__{key}__*.json"))):
        d = read_json(f)
        if d and "metrics" in d:
            out[d["job"]["seed"]] = dict(d["metrics"], _status=d["summary"].get("status"))
    return out


def boot_ci(x, rng):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    bs = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(A, B, key, rng):
    s = sorted(set(A) & set(B))
    d = np.array([A[x][key] - B[x][key] for x in s if A[x][key] == A[x][key] and B[x][key] == B[x][key]])
    if len(d) < 3:
        return None
    from scipy import stats
    t, p = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    m, lo, hi = boot_ci(d, rng)
    return dict(n=len(d), mean=m, lo=lo, hi=hi, t=float(t), p=float(p), p_wilcoxon=float(w),
                pos=int((d > 0).sum()))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def report(args):
    from .theta_sweep import THETAS
    log = logger("s16 report")
    out = rdir("seeds16")
    rng = np.random.default_rng(0)
    rs_th, rs_val = _rs_thetas()
    keys = [k for k, _, _ in configs()]
    D = {k: per_seed(k) for k in keys}
    A = {}
    for k in keys:
        a = {"n": len(D[k]), "truncated": sum(v["_status"] not in ("ok", "all_done", "deadline") for v in D[k].values())}
        for q in KEYS:
            a[q] = boot_ci([v[q] for v in D[k].values()], rng)
        A[k] = a

    wm_keys = ["wm__noskip"] + [f"wm__{th:+.2f}" for th in THETAS]
    rs_keys = [f"rs__{th:+.3f}" for th in rs_th]

    pairs = []
    for rk in rs_keys:
        wk = min(wm_keys, key=lambda w: abs(A[w]["throughput"][0] - A[rk]["throughput"][0]))
        pairs.append((rk, wk))
    tests = {}
    for q in ("harvest_success", "contact_rate", "safe_return"):
        res = [paired(D[wk], D[rk], q, rng) for rk, wk in pairs]
        ps = [r["p"] if r else 1.0 for r in res]
        adj = holm(np.array(ps))
        tests[q] = [(rk, wk, r, a) for (rk, wk), r, a in zip(pairs, res, adj)]
    vs = {}
    for name, (ka, kb) in {
            "world model (val θ +0.15) − detect-and-pick": ("wm__+0.15", "detect_and_pick"),
            "world model (val θ +0.15) − reactive retry": ("wm__+0.15", "reactive_retry"),
            "world model (val θ +0.15) − selective D&P (val θ)": ("wm__+0.15", f"rs__{rs_val:+.3f}"),
            "selective D&P (val θ) − detect-and-pick": (f"rs__{rs_val:+.3f}", "detect_and_pick"),
            "calibrated − uncalibrated world model (val θ each)": ("wm_cal", "wm__+0.15")}.items():
        vs[name] = {q: paired(D[ka], D[kb], q, rng) for q in ("harvest_success", "contact_rate", "safe_return", "throughput")}
    main_run = {}
    for f in glob.glob(os.path.join(RESULTS, "control", "jobs", "main__nominal__wm_full__*.json")):
        d = read_json(f)
        if d and "metrics" in d:
            main_run[d["job"]["seed"]] = d["metrics"]
    rep = paired({s: v for s, v in D["wm__+0.15"].items() if s in OLD_SEEDS}, main_run, "safe_return", rng)

    f3 = lambda m: f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]"
    f1 = lambda m: f"{m[0]:.1f} [{m[1]:.1f}, {m[2]:.1f}]"
    S = ["# Trade-off curves over 16 seeds (8 main + 8 new unseen-tree seeds)", "",
         "Mean [95 % bootstrap CI over seeds]. n = seeds with a result; truncated = episodes stopped "
         "by a simulator blow-up (kept, as in all earlier tables).", "",
         "| config | n | trunc. | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in ["detect_and_pick", "reactive_retry"] + rs_keys + wm_keys + ["wm_cal"]:
        a = A[k]
        S.append(f"| {k} | {a['n']} | {a['truncated']} | {f1(a['harvest_success'])} | {f1(a['contact_rate'])} | "
                 f"{f1(a['damage_rate'])} | {f1(a['skip_rate'])} | {f3(a['throughput'])} | {f3(a['safe_return'])} |")
    fmt_t = lambda r, d=1: ("n/a" if r is None else
                            f"{r['mean']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}], t={r['t']:+.2f}, "
                            f"p={r['p']:.3f}, Wilcoxon p={r['p_wilcoxon']:.3f}, {r['pos']}/{r['n']} seeds > 0")
    S += ["", "## Matched throughput: world model − selective D&P (paired over seeds)", "",
          "Pairs fixed by rule (nearest 16-seed mean throughput), not by outcome. "
          "p_holm = Holm-adjusted over the 5 pairs, per metric.", ""]
    for q, d in [("harvest_success", 1), ("contact_rate", 1), ("safe_return", 3)]:
        S.append(f"**{q}**")
        S.append("")
        for rk, wk, r, a in tests[q]:
            S.append(f"- {wk} ({A[wk]['throughput'][0]:.3f} fpm) vs {rk} ({A[rk]['throughput'][0]:.3f} fpm): "
                     f"{fmt_t(r, d)}, p_holm={a:.3f}")
        S.append("")
    S += ["## Other paired comparisons (16 seeds)", ""]
    for name, R in vs.items():
        S.append(f"- {name}:")
        for q, d in [("harvest_success", 1), ("contact_rate", 1), ("safe_return", 3), ("throughput", 3)]:
            S.append(f"  - {q}: {fmt_t(R[q], d)}")
    if rep:
        S += ["", f"Reproducibility (8 main seeds run twice at θ = +0.15): safe-return difference "
                  f"{rep['mean']:+.3f} [{rep['lo']:+.3f}, {rep['hi']:+.3f}] (t={rep['t']:+.2f}, p={rep['p']:.3f})."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(S) + "\n")
    write_json(os.path.join(out, "stats.json"), dict(agg=A, tests={q: [(rk, wk, r, float(a)) for rk, wk, r, a in v]
                                                                  for q, v in tests.items()}, paired=vs))
    log(f"wrote {out}/summary.md, stats.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["launch", "worker", "report", "list"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--run-token", default="local")
    args = ap.parse_args()
    if args.mode == "list":
        J = build_jobs()
        print(len(J), "jobs;", len(configs()), "configs:", [k for k, _, _ in configs()])
    else:
        {"launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
