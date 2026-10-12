from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from .common import RESULTS, load_config, rdir, logger, write_json, read_json, device_name

DET_COLS = ["tgt_bx", "tgt_by", "tgt_bz", "tgt_cx", "tgt_cy", "tgt_cz", "det_found",
            "det_radius", "det_log_npix", "det_rms_cm", "tgt_u", "tgt_v", "tgt_dist",
            "tgt_height", "n_dets"]
SWEEP_Q = [0.2, 0.4, 0.6, 0.8]
HEAD_PICK_ONLY = [True, True, False, False, False]


def _splits(table):
    clean = (table["batch_status"].fillna("nan") != "nan").values
    usable = (table["skipped"] == 0).values & clean
    sp = table["split"].values
    return (np.nonzero(usable & (sp == "train"))[0], np.nonzero(usable & (sp == "val"))[0],
            np.nonzero(usable & (sp == "test"))[0], np.nonzero(clean & (sp == "val"))[0])


def _finite_rows(arrs, idx):
    ok = np.isfinite(np.asarray(arrs["pre_state"][idx])).all(1)
    return idx[ok]


def fit_calibration(table, arrs, tr, va, te, va_all, dev, log, run_dirs=None):
    import torch
    from sklearn.linear_model import LogisticRegression
    from .eval_offline import predict, binary_metrics, ips_value
    from .decision_picker import PICKS
    from .model import load_run, BIN_HEADS
    pick = table["action"].isin(PICKS).values
    Y = np.stack([table[k].values for k in BIN_HEADS], 1).astype(int)
    grid = np.round(np.arange(-0.3, 0.41, 0.025), 3)
    out = {}
    for run_dir in (run_dirs or sorted(glob.glob(os.path.join(RESULTS, "runs", "*_s*")))):
        name = os.path.basename(run_dir)
        if not os.path.exists(os.path.join(run_dir, "best.pt")):
            continue
        try:
            model, meta = load_run(run_dir, dev)
        except Exception as ex:
            log(f"skip {name}: {ex}")
            continue
        Pv, _, _, _ = predict(model, meta, arrs, va, dev)
        Pt, _, _, _ = predict(model, meta, arrs, te, dev)
        a, b = np.ones(5), np.zeros(5)
        before, after = {}, {}
        for k, h in enumerate(BIN_HEADS):
            mv = pick[va] if HEAD_PICK_ONLY[k] else np.ones(len(va), bool)
            mt = pick[te] if HEAD_PICK_ONLY[k] else np.ones(len(te), bool)
            z = np.log(np.clip(Pv[mv, k], 1e-5, 1 - 1e-5) / (1 - np.clip(Pv[mv, k], 1e-5, 1 - 1e-5)))
            y = Y[va][mv, k]
            if len(np.unique(y)) == 2:
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
                a[k], b[k] = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            zt = np.log(np.clip(Pt[mt, k], 1e-5, 1 - 1e-5) / (1 - np.clip(Pt[mt, k], 1e-5, 1 - 1e-5)))
            before[h] = binary_metrics(Y[te][mt, k], Pt[mt, k])
            after[h] = binary_metrics(Y[te][mt, k], 1 / (1 + np.exp(-(a[k] * zt + b[k]))))
        write_json(os.path.join(run_dir, "calib.json"),
                   dict(a=a.tolist(), b=b.tolist(), heads=BIN_HEADS, fitted_on="validation trees",
                        method="per-head Platt scaling on logit of the (ensemble-mean) probability"))
        model.calib = (torch.tensor(a, dtype=torch.float32, device=dev),
                       torch.tensor(b, dtype=torch.float32, device=dev))
        _, _, Sv, _ = predict(model, meta, arrs, va_all, dev)
        Sv6 = np.concatenate([Sv, np.full((len(Sv), 1), -np.inf)], 1)
        vals = [ips_value(table.iloc[va_all], Sv6, th) for th in grid]
        kbest = int(np.nanargmax(vals))
        write_json(os.path.join(run_dir, "policy_cal.json"),
                   dict(theta=float(grid[kbest]), ips_value=float(vals[kbest]),
                        ips_curve=dict(theta=grid.tolist(), value=vals), calibrated=True))
        out[name] = dict(variant=meta["variant"], a=a.tolist(), b=b.tolist(),
                         theta_cal=float(grid[kbest]),
                         theta_uncal=float(meta.get("policy", {}).get("theta", float("nan"))),
                         before=before, after=after)
        log(f"{name}: ECE success {before['success']['ece']:.3f} -> {after['success']['ece']:.3f}, "
            f"contact {before['contact']['ece']:.3f} -> {after['contact']['ece']:.3f}; "
            f"theta {out[name]['theta_uncal']} -> {out[name]['theta_cal']}")
        if name == "full_s0":
            np.savez_compressed(os.path.join(rdir("extras"), "calib_preds_full_s0.npz"),
                                idx=te, P=Pt, a=a, b=b)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def fit_reactive_select(table, arrs, tr, va, te, va_all, log):
    from sklearn.linear_model import LogisticRegression
    from .decision_picker import STATE_NAMES
    from .eval_offline import ips_value, binary_metrics
    from .policies import ReactiveSelect
    cols = [STATE_NAMES.index(c) for c in DET_COLS]
    trp = tr[table["action"].values[tr] == 0]
    trp = _finite_rows(arrs, trp)
    X = np.asarray(arrs["pre_state"][trp])[:, cols].astype(float)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    Xs = (X - mu) / sd
    heads, test_m = {}, {}
    tep = te[table["action"].values[te] == 0]
    tep = _finite_rows(arrs, tep)
    Xt = (np.asarray(arrs["pre_state"][tep])[:, cols] - mu) / sd
    for h in ("success", "contact", "damage", "detach"):
        y = table[h].values[trp].astype(int)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
        heads[h] = dict(w=lr.coef_[0].tolist(), b=float(lr.intercept_[0]))
        test_m[h] = binary_metrics(table[h].values[tep].astype(int), lr.predict_proba(Xt)[:, 1])
    P = dict(cols=cols, col_names=DET_COLS, mu=mu.tolist(), sd=sd.tolist(), heads=heads,
             tau_pick_center=float(table["tau"].values[trp].mean()), theta=0.0,
             n_train=int(len(trp)), test_metrics_pick_center=test_m)
    path = os.path.join(rdir("extras"), "reactive_select.json")
    write_json(path, P)
    pol = ReactiveSelect(path)
    vv = _finite_rows(arrs, va_all)
    sc = np.array([pol.score(s) for s in np.asarray(arrs["pre_state"][vv])])
    S6 = np.full((len(vv), 6), -np.inf)
    S6[:, 0] = sc
    grid = np.round(np.arange(-0.6, 0.41, 0.025), 3)
    vals = [ips_value(table.iloc[vv], S6, th) for th in grid]
    kb = int(np.nanargmax(vals))
    first = table["dec_idx"].values[vv] == 0
    qs = [float(np.round(np.quantile(sc[first], q), 3)) for q in SWEEP_Q]
    P.update(theta=float(grid[kb]), ips_value=float(vals[kb]),
             ips_curve=dict(theta=grid.tolist(), value=vals),
             sweep_thetas=sorted(set([float(grid[kb])] + qs)),
             val_score_quantiles=dict(zip([str(q) for q in SWEEP_Q], qs)))
    write_json(path, P)
    log(f"reactive_select: {len(trp)} straight-pick train rows; test success AUROC "
        f"{test_m['success']['auroc']:.3f}; theta_val {P['theta']}; sweep {P['sweep_thetas']}")
    return P


def fit(args):
    import torch
    from .dataset import load
    log = logger("extras fit")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load()
    arrs = dict(arrs)
    arrs["__action"] = table["action"].values.astype(np.int64)
    tr, va, te, va_all = _splits(table)
    tr, va, te, va_all = (_finite_rows(arrs, x) for x in (tr, va, te, va_all))
    log(f"rows: train {len(tr)} val {len(va)} test {len(te)} (val incl. skip {len(va_all)})")
    cal = fit_calibration(table, arrs, tr, va, te, va_all, dev, log)
    rs = fit_reactive_select(table, arrs, tr, va, te, va_all, log)
    write_json(os.path.join(rdir("extras"), "fit.json"), dict(calibration=cal, reactive_select=dict(
        theta=rs["theta"], sweep_thetas=rs["sweep_thetas"], test=rs["test_metrics_pick_center"]),
        t=time.time()))


def build_jobs(cfg):
    from .eval_control import NOMINAL
    ev = cfg["eval"]
    base, nm = int(ev["seed_base"]), int(ev["main_seeds"])
    P = read_json(os.path.join(RESULTS, "extras", "reactive_select.json"))
    if P is None:
        raise SystemExit("run `python -m owsim.extras fit` first")
    jobs = []
    for i in range(nm):
        s = base + i
        jobs.append(dict(id=f"extras__wm_full_cal__val__{s}", controller="wm_full_cal",
                         theta=None, seed=s, **NOMINAL))
        for th in P["sweep_thetas"]:
            jobs.append(dict(id=f"extras__reactive_select__{th:+.3f}__{s}",
                             controller="reactive_select", theta=float(th), seed=s, **NOMINAL))
    return jobs


def worker(args):
    from .eval_control import light_for, job_metrics, claim
    from .policies import make_policy
    from .rollout import run_batch
    cfg = load_config(args.config)
    ev = cfg["eval"]
    log = logger(f"extras w{args.worker_id}")
    jdir = rdir("extras", "jobs")
    cdir = rdir("extras", f"claims_{args.run_token}")
    dev = "cuda" if device_name() == "cuda" else "cpu"
    cache = {}
    for jb in build_jobs(cfg):
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res):
            continue
        if args.deadline and time.time() > args.deadline - 60:
            log("deadline reached; leaving remaining jobs")
            break
        if not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        c = jb["controller"]
        if c not in cache:
            cache[c] = make_policy(c, runs_root=os.path.join(RESULTS, "runs"), device=dev)
        pol = cache[c]
        if jb["theta"] is not None:
            pol.theta = float(jb["theta"])
        pol.latency_ms = []
        light, el = light_for(jb["seed"], jb["light"])
        spec = dict(seed=jb["seed"], num_envs=int(ev["num_envs"]), foliage=jb["foliage"],
                    apples=jb["apples"], light_dir=light, light_el=el, shadows=False,
                    frames=int(ev["frames"]), device=device_name(),
                    max_decisions=int(cfg["collect"].get("max_decisions", 4)))
        log(f"job {jb['id']} (theta {pol.theta:+.3f})")
        try:
            records, summary = run_batch(spec, pol, log=log, deadline=None, keep_obs=0)
        except Exception:
            import traceback
            log("job failed:\n" + traceback.format_exc())
            write_json(res, dict(job=jb, error="exception"))
            continue
        m = job_metrics(records, summary)
        write_json(res, dict(job=dict(jb, theta_used=float(pol.theta)), metrics=m, summary=summary))
        log(f"  -> harvest {m['harvest_success']:.1f}% contact {m['contact_rate']:.1f}% "
            f"skip {m['skip_rate']:.1f}% fruit/robot/min {m['throughput']:.3f} "
            f"return {m['safe_return']:.3f} ({summary['wall_s']:.0f}s)")


def launch(args):
    from .launch import gpus, spawn, wait_all
    cfg = load_config(args.config)
    log = logger("extras launch")
    G = gpus()
    n = len(G) * int(cfg["eval"]["workers_per_gpu"])
    deadline = time.time() + 3600.0 * args.hours
    token = time.strftime("%Y%m%d_%H%M%S")
    todo = [j for j in build_jobs(cfg)
            if not os.path.exists(os.path.join(rdir("extras", "jobs"), j["id"] + ".json"))]
    log(f"{len(todo)} extras jobs to do; {n} workers on GPUs {G}; deadline {time.ctime(deadline)}")
    procs = []
    for i in range(min(n, len(todo))):
        cmd = [sys.executable, "-m", "owsim.extras", "worker", "--config", args.config,
               "--worker-id", str(i), "--run-token", token, "--deadline", str(deadline)]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rdir("logs"), f"extras_w{i:02d}.log")))
        time.sleep(2.0)
    wait_all(procs, log)
    report(args)


def _load(pattern):
    return [d for d in (read_json(f) for f in sorted(glob.glob(pattern))) if d and "metrics" in d]


def report(args):
    from .theta_sweep import _agg, THETAS
    log = logger("extras report")
    out = rdir("extras")
    J = os.path.join(RESULTS, "control", "jobs")
    T = os.path.join(RESULTS, "theta_sweep", "jobs")
    E = os.path.join(out, "jobs")
    P = read_json(os.path.join(out, "reactive_select.json"), {})
    FIT = read_json(os.path.join(out, "fit.json"), {})

    ref = {c: _agg(_load(os.path.join(J, f"main__nominal__{c}__*.json")))
           for c in ["detect_and_pick", "reactive_retry", "wm_full", "wm_full_noskip"]}
    wm_curve = [(float("-inf"), ref["wm_full_noskip"])]
    for th in THETAS:
        rows = [d for d in _load(os.path.join(T, "theta__*.json")) if abs(d["job"]["theta"] - th) < 1e-9]
        if rows:
            wm_curve.append((th, _agg(rows)))
    rs_curve = []
    for th in P.get("sweep_thetas", []):
        rows = _load(os.path.join(E, f"extras__reactive_select__{th:+.3f}__*.json"))
        if rows:
            rs_curve.append((th, _agg(rows)))
    cal = _agg(_load(os.path.join(E, "extras__wm_full_cal__val__*.json")))

    C = FIT.get("calibration", {})

    def paired(pa, pb, key):
        A = {d["job"]["seed"]: d["metrics"][key] for d in _load(pa)}
        B = {d["job"]["seed"]: d["metrics"][key] for d in _load(pb)}
        s = sorted(set(A) & set(B))
        dl = np.array([A[x] - B[x] for x in s if A[x] == A[x] and B[x] == B[x]])
        if len(dl) < 2:
            return "n/a"
        t = dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)) + 1e-12)
        return f"{dl.mean():+.3f} (s.d. {dl.std(ddof=1):.3f}, t={t:+.2f}, {int((dl > 0).sum())}/{len(dl)} seeds positive)"

    rsv = f"{P.get('theta', 0):+.3f}"
    S = ["# Phase 3 extras: calibration + selectivity control", "",
         "## Trade-off points (nominal, 8 seeds; mean over seeds)", "",
         "| controller | θ | n | harvest/pick % | contact % | damage % | skip % | fruit/robot/min | safe return | truncated |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    row = lambda n, th, a: (f"| {n} | {th} | {a['n']} | {a['harvest_success']:.1f} | {a['contact_rate']:.1f} | "
                            f"{a['damage_rate']:.1f} | {a['skip_rate']:.1f} | {a['throughput']:.3f} | "
                            f"{a['safe_return']:.3f} | {a['truncated']} |")
    S.append(row("detect-and-pick", "-", ref["detect_and_pick"]))
    S.append(row("reactive retry", "-", ref["reactive_retry"]))
    S += [row("selective D&P", f"{th:+.3f}" + (" (val)" if abs(th - P.get('theta', 99)) < 1e-9 else ""), a)
          for th, a in rs_curve]
    S += [row("world model", "−∞" if not np.isfinite(th) else f"{th:+.2f}", a) for th, a in wm_curve]
    S.append(row("world model (main run, val θ)", "val", ref["wm_full"]))
    if cal["n"]:
        S.append(row("world model calibrated", "val (cal)", cal))
    pa_rs = os.path.join(E, f"extras__reactive_select__{rsv}__*.json")
    S += ["", "## Paired comparisons (per seed)", "",
          f"- world model (val θ) − selective D&P (val θ), safe return: "
          f"{paired(os.path.join(J, 'main__nominal__wm_full__*.json'), pa_rs, 'safe_return')}",
          f"- same, harvest success per attempt: "
          f"{paired(os.path.join(J, 'main__nominal__wm_full__*.json'), pa_rs, 'harvest_success')}",
          f"- same, contact rate: "
          f"{paired(os.path.join(J, 'main__nominal__wm_full__*.json'), pa_rs, 'contact_rate')}",
          f"- selective D&P (val θ) − detect-and-pick, safe return: "
          f"{paired(pa_rs, os.path.join(J, 'main__nominal__detect_and_pick__*.json'), 'safe_return')}",
          f"- calibrated − uncalibrated world model, safe return: "
          f"{paired(os.path.join(E, 'extras__wm_full_cal__val__*.json'), os.path.join(J, 'main__nominal__wm_full__*.json'), 'safe_return')}",
          "", "## Matched throughput (world-model θ point closest in fruit/robot/min)", ""]
    for th, a in rs_curve:
        if not wm_curve:
            break
        j = int(np.argmin([abs(b["throughput"] - a["throughput"]) for _, b in wm_curve]))
        tw, b = wm_curve[j]
        S.append(f"- selective D&P θ={th:+.3f} ({a['throughput']:.3f} fpm): HS {a['harvest_success']:.1f} %, "
                 f"CR {a['contact_rate']:.1f} %, SR {a['safe_return']:.3f}  vs  world model θ="
                 f"{'−∞' if not np.isfinite(tw) else f'{tw:+.2f}'} ({b['throughput']:.3f} fpm): "
                 f"HS {b['harvest_success']:.1f} %, CR {b['contact_rate']:.1f} %, SR {b['safe_return']:.3f}")
    if C:
        S += ["", "## Calibration (test trees, ECE before → after, per run)", ""]
        for name, r in sorted(C.items()):
            S.append(f"- {name}: detach {r['before']['detach']['ece']:.3f}→{r['after']['detach']['ece']:.3f}, "
                     f"success {r['before']['success']['ece']:.3f}→{r['after']['success']['ece']:.3f}, "
                     f"contact {r['before']['contact']['ece']:.3f}→{r['after']['contact']['ece']:.3f}; "
                     f"θ {r['theta_uncal']} → {r['theta_cal']}")
    rt = FIT.get("reactive_select", {}).get("test", {})
    if rt:
        S += ["", f"Selective D&P predictor (straight picks, test trees): success AUROC "
                  f"{rt['success']['auroc']:.3f}, contact AUROC {rt['contact']['auroc']:.3f}."]
    S += ["", "Note: GPU physics is not bit-reproducible; differences below ~0.1 safe return between "
              "single points are within run-to-run noise (see theta_sweep reproducibility check)."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(S) + "\n")
    log(f"wrote {out}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "launch", "worker", "report", "list"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--run-token", default="local")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--hours", type=float, default=1000.0)
    args = ap.parse_args()
    if args.mode == "list":
        J = build_jobs(load_config(args.config))
        print(len(J), "jobs; first", J[0]["id"], "last", J[-1]["id"])
    else:
        {"fit": fit, "launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
