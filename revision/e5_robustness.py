from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from owsim.common import RESULTS, load_config, logger, write_json, read_json, device_name
from revision._common import (rev_dir, nominal_spec, run_jobs, launch_workers, todo, load_jobs,
                              fmt_ci, common_args)

EXP = "e5_robustness"
NEW = range(4, 8)
CTRLS = ["detect_and_pick", "reactive_retry", "wm_full", "wm_vjepa"]


def build_jobs(cfg, with_foliage=False):
    from owsim.eval_control import SHIFTS, SWEEP_FOLIAGE, NOMINAL, seen_seeds
    base = int(cfg["eval"]["seed_base"])
    seen = seen_seeds(cfg, max(NEW) + 1)
    jobs = []
    for i in NEW:
        for cond, kw in SHIFTS.items():
            kw = dict(kw)
            seed = seen[i] if kw.pop("seen", False) else base + 200 + i
            c = dict(NOMINAL)
            c.update(kw)
            for ctrl in CTRLS:
                jobs.append(dict(id=f"shift__{cond}__{ctrl}__{seed}", group="shift", cond=cond,
                                 controller=ctrl, seed=int(seed), **c))
        if with_foliage:
            for f in SWEEP_FOLIAGE:
                c = dict(NOMINAL)
                c["foliage"] = f
                for ctrl in CTRLS:
                    jobs.append(dict(id=f"foliage__f{f:.1f}__{ctrl}__{base + 100 + i}", group="foliage",
                                     cond=f"f{f:.1f}", controller=ctrl, seed=base + 100 + i, **c))
    return jobs


def worker(args):
    from owsim.policies import make_policy
    from owsim.rollout import run_batch
    cfg = load_config(args.config)
    cache = {}
    log = logger(f"e5 w{args.worker_id}")

    def run_one(jb):
        c = jb["controller"]
        if c not in cache:
            cache[c] = make_policy(c, runs_root=os.path.join(RESULTS, "runs"),
                                   device="cuda" if device_name() == "cuda" else "cpu")
        pol = cache[c]
        if hasattr(pol, "latency_ms"):
            pol.latency_ms = []
        spec = nominal_spec(cfg, jb["seed"], jb["foliage"], jb["apples"], light=jb["light"])
        return run_batch(spec, pol, log=log, deadline=None, keep_obs=0)

    run_jobs(EXP, build_jobs(cfg, args.with_foliage), run_one, args.worker_id, args.run_token)


def launch(args):
    cfg = load_config(args.config)
    extra = ("--with-foliage",) if args.with_foliage else ()
    launch_workers("revision.e5_robustness", EXP, len(todo(EXP, build_jobs(cfg, args.with_foliage))),
                   args.config, extra)
    report(args)


def report(args):
    from owsim.seeds16 import boot_ci
    log = logger("e5 report")
    rng = np.random.default_rng(0)
    out = rev_dir(EXP)
    rows = {}
    for f in glob.glob(os.path.join(RESULTS, "control", "jobs", "*.json")):
        d = read_json(f)
        if d and "metrics" in d and d["job"].get("group") in ("shift", "foliage"):
            rows[d["job"]["id"]] = d
    old_ids = set(rows)
    rows.update(load_jobs(EXP))
    cells = {}
    for jid, d in rows.items():
        jb = d["job"]
        cells.setdefault((jb["group"], jb["cond"], jb["controller"]), []).append(
            (jb["seed"], d["metrics"], d["summary"].get("status"), jid in old_ids))
    Q = [("harvest_success", "HS", 1), ("contact_rate", "CR", 1), ("safe_return", "SR", 2)]
    L = ["# E5 - robustness cells with up to 8 seeds (4 from the paper + 4 new)", "",
         "Mean [95 % bootstrap CI over seeds]; trunc. = episodes stopped by a non-finite state.", "",
         "| group | condition | controller | n (old+new) | trunc. | HS | CR | SR |", "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for key in sorted(cells):
        v = cells[key]
        a = {q: boot_ci([m[q] for _, m, _, _ in v], rng) for q, _, _ in Q}
        n_old = sum(o for *_, o in v)
        tr = sum(s not in ("ok", "all_done", "deadline") for _, _, s, _ in v)
        agg["|".join(key)] = dict(a, n=len(v), n_old=n_old, truncated=tr)
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(v)} ({n_old}+{len(v) - n_old}) | {tr} | "
                 f"{fmt_ci(a['harvest_success'])} | {fmt_ci(a['contact_rate'])} | {fmt_ci(a['safe_return'], 2)} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    write_json(os.path.join(out, "stats.json"), agg)

    log(f"wrote {out}/summary.md, stats.json")


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("mode", choices=["list", "launch", "worker", "report"])
    ap.add_argument("--with-foliage", action="store_true")
    args = ap.parse_args()
    if args.mode == "list":
        J = build_jobs(load_config(args.config), args.with_foliage)
        print(len(J), "jobs;", len(todo(EXP, J)), "still to do")
        return
    {"launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
