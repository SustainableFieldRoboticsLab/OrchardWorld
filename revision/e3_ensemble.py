from __future__ import annotations

import argparse
import os
import time

import numpy as np

from owsim.common import RESULTS, load_config, logger, write_json, read_json, device_name
from revision._common import (EVAL_SEEDS, rev_dir, nominal_spec, run_jobs, launch_workers, todo,
                              load_jobs, per_seed_metrics, fmt_ci, fmt_paired, common_args)

EXP = "e3_ensemble"
RUNS = ["full_s0", "full_s1", "full_s2"]
GRID = np.round(np.arange(-0.3, 0.41, 0.025), 3)


def fit(args):
    import torch
    from owsim.dataset import load
    from owsim.eval_offline import predict, outcome_metrics, ips_value
    from owsim.extras import _splits, _finite_rows
    from owsim.model import load_run
    log = logger("e3 fit")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load()
    arrs = dict(arrs)
    arrs["__action"] = table["action"].values.astype(np.int64)
    tr, va, te, va_all = (_finite_rows(arrs, x) for x in _splits(table))
    res = dict(members={})
    for cal in (False, True):
        Pt, St, Sv = [], [], []
        for r in RUNS:
            model, meta = load_run(os.path.join(RESULTS, "runs", r), dev, calibrated=cal)
            P, _, S, _ = predict(model, meta, arrs, te, dev)
            _, _, Svv, _ = predict(model, meta, arrs, va_all, dev)
            Pt.append(P)
            St.append(S)
            Sv.append(Svv)
            m = outcome_metrics(table.iloc[te], P)
            res["members"][r + ("_cal" if cal else "")] = dict(
                detach_auroc=m["detach"]["auroc"], success_ap=m["success"]["ap"],
                contact_auroc=m["contact"]["auroc"], ece_mean=m["ece_mean"])
            del model
        m = outcome_metrics(table.iloc[te], np.mean(Pt, 0))
        Sv6 = np.concatenate([np.mean(Sv, 0), np.full((len(va_all), 1), -np.inf)], 1)
        vals = [ips_value(table.iloc[va_all], Sv6, th) for th in GRID]
        k = int(np.nanargmax(vals))
        key = "ensemble_cal" if cal else "ensemble"
        res[key] = dict(detach_auroc=m["detach"]["auroc"], success_ap=m["success"]["ap"],
                        contact_auroc=m["contact"]["auroc"], damage_auroc=m["damage"]["auroc"],
                        ece_mean=m["ece_mean"], theta=float(GRID[k]), ips_value=float(vals[k]),
                        ips_curve=dict(theta=GRID.tolist(), value=vals))
        log(f"{key}: detAUC {m['detach']['auroc']:.3f} succAP {m['success']['ap']:.3f} "
            f"contactAUC {m['contact']['auroc']:.3f} ECE {m['ece_mean']:.3f} theta {GRID[k]:+.3f}")
    write_json(os.path.join(rev_dir(EXP), "fit.json"), res)


class EnsemblePolicy:

    def __init__(self, device, theta):
        from owsim.model import load_run
        self.members = [load_run(os.path.join(RESULTS, "runs", r), device, calibrated=True) for r in RUNS]
        self.device = device
        self.theta = float(theta)
        self.name = "wm_ens3_cal"
        self.latency_ms = []

    def decide(self, picker, obs):
        from owsim.model import score_observation
        from owsim.decision_picker import N_ACTIONS, PICKS, REPOS, SKIP
        t0 = time.perf_counter()
        scores = np.mean([score_observation(m, meta, obs, self.device) for m, meta in self.members], 0)
        self.latency_ms.append(1e3 * (time.perf_counter() - t0))
        best = max(list(PICKS) + list(REPOS), key=lambda a: scores[a])
        if scores[best] < self.theta:
            best = SKIP
        sc = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        sc[:len(scores)] = scores
        return best, dict(p=1.0, scores=sc)


def build_jobs():
    return [dict(id=f"e3__wm_ens3_cal__{s}", controller="wm_ens3_cal", seed=s, foliage=0.8, apples=35)
            for s in EVAL_SEEDS]


def worker(args):
    from owsim.rollout import run_batch
    cfg = load_config(args.config)
    F = read_json(os.path.join(rev_dir(EXP), "fit.json"))
    if F is None:
        raise SystemExit("run 'python -m revision.e3_ensemble fit' first")
    pol = EnsemblePolicy("cuda" if device_name() == "cuda" else "cpu", F["ensemble_cal"]["theta"])
    log = logger(f"e3 w{args.worker_id}")

    def run_one(jb):
        pol.latency_ms = []
        spec = nominal_spec(cfg, jb["seed"], jb["foliage"], jb["apples"])
        records, summary = run_batch(spec, pol, log=log, deadline=None, keep_obs=0)
        summary["theta_used"] = pol.theta
        return records, summary

    run_jobs(EXP, build_jobs(), run_one, args.worker_id, args.run_token)


def launch(args):
    if read_json(os.path.join(rev_dir(EXP), "fit.json")) is None:
        fit(args)
    launch_workers("revision.e3_ensemble", EXP, len(todo(EXP, build_jobs())), args.config)
    report(args)


def report(args):
    from owsim.seeds16 import boot_ci, paired, per_seed
    log = logger("e3 report")
    rng = np.random.default_rng(0)
    out = rev_dir(EXP)
    F = read_json(os.path.join(out, "fit.json"), {})
    E = per_seed_metrics(load_jobs(EXP), lambda jb: True)
    refs = {"world model, calibrated (single, full_s0)": per_seed("wm_cal"),
            "detect-and-pick": per_seed("detect_and_pick")}
    Q = [("harvest_success", 1), ("contact_rate", 1), ("throughput", 3), ("safe_return", 3)]
    L = ["# E3 - ensemble of 3 training seeds (calibrated)", ""]
    if F:
        L += ["## Offline (test split)", "", "| model | detach AUROC | success AP | contact AUROC | ECE |",
              "|---|---|---|---|---|"]
        for k, d in list(F["members"].items()) + [("ensemble", F.get("ensemble")), ("ensemble_cal", F.get("ensemble_cal"))]:
            if d:
                L.append(f"| {k} | {d['detach_auroc']:.3f} | {d['success_ap']:.3f} | {d['contact_auroc']:.3f} | "
                         f"{d['ece_mean']:.3f} |")
        L += ["", f"Validation-selected theta of the calibrated ensemble: {F['ensemble_cal']['theta']:+.3f}", ""]
    L += ["## Closed loop (16 seeds, nominal)", "", "| controller | n | harvest/pick % | contact % | fruit/robot/min | safe return |",
          "|---|---|---|---|---|---|"]
    for name, d in [("ensemble, calibrated", E)] + list(refs.items()):
        if d:
            a = {q: boot_ci([v[q] for v in d.values()], rng) for q, _ in Q}
            L.append(f"| {name} | {len(d)} | {fmt_ci(a['harvest_success'])} | {fmt_ci(a['contact_rate'])} | "
                     f"{fmt_ci(a['throughput'], 3)} | {fmt_ci(a['safe_return'], 3)} |")
    L.append("")
    tests = {}
    for name, ref in refs.items():
        L.append(f"- ensemble - {name}:")
        for q, d in Q:
            r = paired(E, ref, q, rng)
            tests[f"{name}|{q}"] = r
            L.append(f"  - {q}: {fmt_paired(r, d)}")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    write_json(os.path.join(out, "stats.json"), tests)
    log(f"wrote {out}/summary.md")


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("mode", choices=["fit", "launch", "worker", "report"])
    args = ap.parse_args()
    {"fit": fit, "launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
