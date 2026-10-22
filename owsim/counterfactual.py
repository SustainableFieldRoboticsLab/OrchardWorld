from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

from .common import RESULTS, load_config, rdir, logger, write_json, read_json, device_name, split_of
from .decision_picker import DecisionPicker, SKIP, ACTIONS

BRANCHES = [SKIP, 0, 1, 2, 3, 4, 0]
CENTER1, CENTER2 = 1, 6


class _Fixed:
    def __init__(self):
        self.a = 0

    def decide(self, picker, obs):
        return self.a, dict(p=1.0)


class CounterfactualPicker(DecisionPicker):

    _snap_exclude = ("cf", "idx", "branches", "_sink_inner", "n_groups", "cf_policy")

    def __init__(self, batch, env, sink, ik, rng, idx, branches=BRANCHES):
        self.cf_policy = _Fixed()
        super().__init__(batch, env, self.cf_policy, sink, ik, rng, max_decisions=1)
        self.idx = idx
        self.branches = list(branches)
        self.cf = None
        self.n_groups = 0
        self._sink_inner = sink
        self.sink = self._tagged_sink

    def _tagged_sink(self, rec):
        if self.cf is not None:
            rec["cf_group"] = int(self.cf["gid"])
            rec["cf_branch"] = int(self.cf["k"])
        else:
            rec["cf_group"], rec["cf_branch"] = -1, -1
        self._sink_inner(rec)

    def _decide(self, obs):
        from .snapshot import EnvSnapshot
        if self.cf is None:
            tg = self.tgt
            if tg is None or tg["n_dec"] >= 1 or bool(self.apples.detached[tg["gt_glob"]]):
                self._end_target()
                return
            self.n_groups += 1
            self.cf = dict(gid=self.n_groups, k=0, obs=obs,
                           snap=EnvSnapshot(self.B, self.idx, self.env, self))
        self._run_branch()

    def _run_branch(self):
        self.policy = self.cf_policy
        self.cf_policy.a = self.branches[self.cf["k"]]
        DecisionPicker._decide(self, self.cf["obs"])

    def _finish_action(self, obs_post, skipped=False):
        DecisionPicker._finish_action(self, obs_post, skipped)
        cf = self.cf
        if cf is None:
            return
        cf["k"] += 1
        cf["snap"].restore(self.B, self.idx, self)
        if cf["k"] < len(self.branches):
            self._run_branch()
        else:
            self.cf = None
            self._end_target()


def worker(args):
    from .rollout import run_batch, save_records
    from .snapshot import EnvIndex
    cfg = load_config(args.config)
    cc = dict(cfg["collect"])
    cc.update(cfg.get("counterfactual", {}).get("collect", {}))
    wid, nw = args.worker_id, args.n_workers
    log = logger(f"cf w{wid}")
    out = rdir("cf", "shards")
    hb_path = os.path.join(rdir("cf", "heartbeat"), f"w{wid:02d}.json")
    rng = np.random.default_rng(20011 + wid)
    tot = dict(worker=wid, batches=0, decisions=0, groups=0, started=time.time(), state="running")
    est, k = None, -1
    while True:
        k += 1
        seed = int(cc["seed_base"]) + k * nw + wid
        shard = os.path.join(out, f"b{seed}.npz")
        if os.path.exists(shard):
            continue
        remaining = args.deadline - time.time()
        if remaining < 90 or (est is not None and remaining < 0.35 * est):
            break
        fol = float(rng.choice(cc["foliage"]))
        app = int(rng.choice(cc["apples"]))
        az, el = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(35, 80))
        spec = dict(seed=seed, split=split_of(seed), num_envs=int(cc["num_envs"]), foliage=fol,
                    apples=app, light_dir=(-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az),
                                           -math.sin(el)),
                    light_el=math.degrees(el), shadows=False, frames=int(cc["frames"]),
                    device=device_name(), max_decisions=1, counterfactual=True)

        def hb(d, _seed=seed):
            write_json(hb_path, dict(tot, current=dict(seed=_seed, **d), t=time.time()))

        spec["heartbeat"] = hb
        holder = {}

        def factory(B, e, sink, ik, rng_, _h=holder):
            if "idx" not in _h:
                _h["idx"] = EnvIndex(B)
            return CounterfactualPicker(B, e, sink, ik, rng_, _h["idx"])

        log(f"batch seed={seed} split={spec['split']} foliage={fol} apples={app}")
        t0 = time.time()
        try:
            records, summary = run_batch(spec, None, log=log, deadline=args.deadline,
                                         picker_factory=factory)
        except Exception:
            import traceback
            log("batch failed:\n" + traceback.format_exc())
            continue

        groups = {}
        for r in records:
            groups.setdefault((r["env"], r["cf_group"]), []).append(r)
        records = [r for g in groups.values() if len(g) == len(BRANCHES) for r in g]
        save_records(shard, records, summary)
        dt = time.time() - t0
        est = dt if est is None else 0.7 * est + 0.3 * dt
        tot["batches"] += 1
        tot["decisions"] += len(records)
        tot["groups"] += len(records) // len(BRANCHES)
        write_json(hb_path, dict(tot, t=time.time()))
        log(f"  -> {len(records) // len(BRANCHES)} complete groups ({len(records)} rows), "
            f"status={summary['status']}, {summary['fps']:.1f} fps, wall {dt:.0f}s")
    tot["state"] = "finished"
    write_json(hb_path, dict(tot, t=time.time()))


def collect(args):
    from .launch import gpus, spawn, wait_all
    cfg = load_config(args.config)
    cc = dict(cfg["collect"])
    cc.update(cfg.get("counterfactual", {}).get("collect", {}))
    log = logger("cf collect")
    G = gpus()
    n = len(G) * int(cc["workers_per_gpu"])
    st_path = os.path.join(rdir("cf"), "deadline.json")
    st = read_json(st_path)
    if st and time.time() < st["deadline"]:
        deadline = st["deadline"]
    elif st:
        log("counterfactual collection deadline already passed (delete results/cf/deadline.json to collect more)")
        return
    else:
        deadline = time.time() + 3600.0 * float(cc["hours"])
        write_json(st_path, dict(deadline=deadline, started=time.time(), workers=n, gpus=G))
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-m", "owsim.counterfactual", "worker", "--config", args.config,
               "--worker-id", str(i), "--n-workers", str(n), "--deadline", str(deadline)]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rdir("logs"), f"cf_w{i:02d}.log")))
        time.sleep(2.0)
    log(f"{n} counterfactual workers on GPUs {G}; deadline {time.ctime(deadline)}")
    wait_all(procs, log)
    if not glob.glob(os.path.join(RESULTS, "cf", "shards", "b*.npz")):
        raise SystemExit("no counterfactual data collected - see results/logs/cf_w*.log")


def dataset(args):
    from .dataset import build
    log = logger("cf dataset")
    cf_sh = os.path.join(RESULTS, "cf", "shards")
    build(out_dir=rdir("cf", "dataset"), shard_dirs=[cf_sh])
    build(out_dir=rdir("dataset_cf"), shard_dirs=[os.path.join(RESULTS, "collect", "shards"), cf_sh])
    log("built results/cf/dataset (counterfactual only) and results/dataset_cf (random + counterfactual)")


def train(args):
    from .launch import gpus, spawn
    cfg = load_config(args.config)
    tc = cfg["train"]
    log = logger("cf train")
    seeds = cfg.get("counterfactual", {}).get("train_seeds", tc["seeds"])
    todo = [s for s in seeds if not os.path.exists(os.path.join(RESULTS, "runs", f"fullcf_s{s}", "meta.json"))]
    G = gpus()
    os.environ["OW_DATASET_DIR"] = os.path.join(RESULTS, "dataset_cf")
    procs = []
    for i, s in enumerate(todo):
        g = G[i % len(G)]
        cmd = [sys.executable, "-m", "owsim.train", "--config", args.config, "--variant", "fullcf",
               "--seed", str(s), "--device", "cpu" if g == "cpu" else "cuda"]
        procs.append(spawn(cmd, g, os.path.join(rdir("logs"), f"train_fullcf_s{s}.log")))
    for p in procs:
        p.wait()
    os.environ.pop("OW_DATASET_DIR", None)
    if not os.path.exists(os.path.join(RESULTS, "runs", "fullcf_s0", "meta.json")):
        raise SystemExit("fullcf_s0 did not train - see results/logs/train_fullcf_s0.log")

    import torch
    from .dataset import load
    from .extras import fit_calibration, _splits, _finite_rows
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load(os.path.join(RESULTS, "dataset"))
    arrs = dict(arrs)
    arrs["__action"] = table["action"].values.astype(np.int64)
    tr, va, te, va_all = (_finite_rows(arrs, x) for x in _splits(table))
    runs = sorted(glob.glob(os.path.join(RESULTS, "runs", "fullcf_s*")))
    cal = fit_calibration(table, arrs, tr, va, te, va_all, dev, log, run_dirs=runs)
    write_json(os.path.join(rdir("cf"), "calibration_fullcf.json"), cal)
    log(f"trained + calibrated {len(runs)} fullcf runs")


def _cluster_boot(values, clusters, rng, n=2000):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return [float("nan")] * 3
    u = np.unique(clusters)
    idx = [np.nonzero(clusters == c)[0] for c in u]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        bs.append(np.mean(np.concatenate([values[idx[i]] for i in pick])))
    return [float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _groups(table):
    G = {}
    for i, (sh, e, g, k) in enumerate(zip(table["shard"], table["env"], table["cf_group"], table["cf_branch"])):
        if g >= 0:
            G.setdefault((sh, e, g), {})[int(k)] = i
    return {key: v for key, v in G.items() if len(v) == len(BRANCHES)}


def _score_models(table, arrs, rows, dev, log):
    import torch
    from .model import load_run, prep_images, plan_scores
    out = {}
    runs = sorted(glob.glob(os.path.join(RESULTS, "runs", "*_s*")))
    for run_dir in runs:
        name = os.path.basename(run_dir)
        variant = name.rsplit("_s", 1)[0]
        if variant == "vjepa" or not os.path.exists(os.path.join(run_dir, "best.pt")):
            continue
        for cal in (False, True):
            if cal and not os.path.exists(os.path.join(run_dir, "calib.json")):
                continue
            if not cal and not os.path.exists(os.path.join(run_dir, "policy.json")):
                continue
            try:
                model, meta = load_run(run_dir, dev, calibrated=cal)
            except Exception as ex:
                log(f"skip {name}: {ex}")
                continue
            P, S = [], []
            with torch.no_grad():
                for i0 in range(0, len(rows), 256):
                    j = rows[i0:i0 + 256]
                    T = lambda k, dt=None: torch.as_tensor(np.array(arrs[k][j]), device=dev, dtype=dt)
                    g, c = prep_images(model, T("pre_g_rgb"), T("pre_g_dep", torch.float32),
                                       T("pre_c_rgb"), T("pre_c_dep", torch.float32), T("pre_uv", torch.float32))
                    s = (T("pre_state", torch.float32) - meta["state_mean_t"]) / meta["state_std_t"]
                    h = model.encode(g, c, s)
                    pa = []
                    for a in range(5):
                        h1 = model.step(h, torch.full((len(j),), a, device=dev, dtype=torch.long))
                        lb, _ = model.heads(h1)
                        pa.append(torch.sigmoid(lb.float()).cpu().numpy())
                    P.append(np.stack(pa, 1))
                    S.append(plan_scores(model, h).float().cpu().numpy())
            out[name + ("_cal" if cal else "")] = dict(
                P=np.concatenate(P), S=np.concatenate(S), variant=variant, calibrated=cal,
                theta=float(meta.get("policy", {}).get("theta", 0.0)))
            del model
    return out


def analyze(args):
    import torch
    from sklearn.metrics import roc_auc_score
    from .dataset import load
    from .policies import ReactiveSelect
    log = logger("cf analyze")
    rng = np.random.default_rng(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load(os.path.join(RESULTS, "cf", "dataset"))
    G = _groups(table)
    keys = sorted(G)
    log(f"{len(keys)} complete counterfactual groups ({len(table)} rows)")
    seeds = np.array([table["seed"].values[G[k][0]] for k in keys])
    split = np.array([table["split"].values[G[k][0]] for k in keys])
    Y = {}
    for col in ("success", "detach", "contact", "damage", "reward", "tau", "visible_after"):
        Y[col] = np.array([[float(table[col].values[G[k][b]]) for b in range(len(BRANCHES))] for k in keys])
    res = dict(n_groups=len(keys), n_seeds=int(len(np.unique(seeds))),
               per_split={s: int((split == s).sum()) for s in ("train", "val", "test")})


    rel = {}
    for col in ("success", "detach", "contact", "damage"):
        a, b = Y[col][:, CENTER1], Y[col][:, CENTER2]
        agree = float(np.mean(a == b))
        pa, pb = a.mean(), b.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        rel[col] = dict(agreement=agree, kappa=float((agree - pe) / max(1 - pe, 1e-9)),
                        rate_first=float(pa), rate_repeat=float(pb))
    ra, rb = Y["reward"][:, CENTER1], Y["reward"][:, CENTER2]
    rel["reward_corr"] = float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")
    rel["reward_abs_diff_mean"] = float(np.mean(np.abs(ra - rb)))
    res["noise_ceiling"] = rel


    eff = {}
    for b, a in enumerate(BRANCHES):
        if b in (0, CENTER1):
            continue
        name = ACTIONS[a] + (" (repeat)" if b == CENTER2 else "")
        eff[name] = {col: _cluster_boot(Y[col][:, b] - Y[col][:, CENTER1], seeds, rng)
                     for col in ("success", "detach", "contact", "damage", "reward")}
    res["paired_effects_vs_center"] = eff
    res["rates_per_branch"] = {f"{b}:{ACTIONS[a]}": {col: float(Y[col][:, b].mean())
                                                    for col in ("success", "detach", "contact", "damage", "reward")}
                               for b, a in enumerate(BRANCHES)}


    te = np.nonzero(split == "test")[0]
    if len(te) < 20:
        log(f"only {len(te)} test groups; using all groups for decision quality (smoke run?)")
        te = np.arange(len(keys))
    R = np.concatenate([np.zeros((len(keys), 1)), Y["reward"][:, [CENTER1, 2, 3, 4, 5]]], 1)
    oracle = R.max(1)
    best_nonskip = R[:, 1:].argmax(1)
    rows = np.array([G[k][0] for k in keys])
    scores = _score_models(table, arrs, rows, dev, log)
    pol = {}

    def summarize(choice, name):
        ch = np.asarray(choice)
        r = np.where(ch < 0, 0.0, R[np.arange(len(R)), ch + 1])
        sub = te
        picks = sub[(ch[sub] >= 0) & (ch[sub] <= 2)]
        execd = sub[ch[sub] >= 0]
        succ = np.array([Y["success"][i, [CENTER1, 2, 3, 4, 5][ch[i]]] for i in picks]) if len(picks) else np.zeros(0)
        cont = np.array([Y["contact"][i, [CENTER1, 2, 3, 4, 5][ch[i]]] for i in execd]) if len(execd) else np.zeros(0)
        pol[name] = dict(
            reward=_cluster_boot(r[sub], seeds[sub], rng),
            regret=_cluster_boot(oracle[sub] - r[sub], seeds[sub], rng),
            best_action_rate=float(np.mean(ch[sub][ch[sub] >= 0] == best_nonskip[sub][ch[sub] >= 0]))
            if (ch[sub] >= 0).any() else float("nan"),
            skip_rate=float(np.mean(ch[sub] < 0)),
            success_per_pick=float(succ.mean()) if len(succ) else float("nan"),
            contact_per_action=float(cont.mean()) if len(cont) else float("nan"),
            action_share={ACTIONS[a]: float(np.mean(ch[sub] == a)) for a in range(5)})

    summarize(np.full(len(keys), -1), "always skip")
    summarize(np.zeros(len(keys), int), "always pick-center (detect-and-pick)")
    rs_path = os.path.join(RESULTS, "extras", "reactive_select.json")
    if os.path.exists(rs_path):
        rsel = ReactiveSelect(rs_path)
        st = np.asarray(arrs["pre_state"][rows])
        sc = np.array([rsel.score(s) for s in st])
        summarize(np.where(sc >= rsel.theta, 0, -1), "selective detect-and-pick")
    best = np.where(oracle > 0, R[:, 1:].argmax(1), -1)
    summarize(best, "oracle (counterfactual best)")
    disc = {}
    for name, d in scores.items():
        S = d["S"]
        a_best = S.argmax(1)
        summarize(np.where(S.max(1) >= d["theta"], a_best, -1), f"world model {name}")

        P = d["P"][te]
        dd = {}
        for oi, col in ((1, "success"), (2, "contact"), (0, "detach")):
            y = Y[col][te][:, [CENTER1, 2, 3, 4, 5]].ravel()
            p = P[:, :, oi].ravel()
            dd[col + "_auroc_all"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")

            yp = Y[col][te][:, [CENTER1, 2, 3]].ravel()
            pp = P[:, :3, oi].ravel()
            dd[col + "_auroc_picks"] = float(roc_auc_score(yp, pp)) if len(np.unique(yp)) == 2 else float("nan")
        disc[name] = dd
    res["decision_quality_test"] = pol
    res["counterfactual_discrimination_test"] = disc
    res["n_test_groups"] = int(len(te))


    try:
        from .eval_offline import predict, outcome_metrics
        from .model import load_run
        rt, ra_ = load(os.path.join(RESULTS, "dataset"))
        ra_ = dict(ra_)
        ra_["__action"] = rt["action"].values.astype(np.int64)
        clean = (rt["batch_status"].fillna("nan") != "nan").values & (rt["skipped"] == 0).values
        tidx = np.nonzero(clean & (rt["split"] == "test").values)[0]
        tidx = tidx[np.isfinite(np.asarray(ra_["pre_state"][tidx])).all(1)]
        off = {}
        for run_dir in sorted(glob.glob(os.path.join(RESULTS, "runs", "full*_s*"))):
            name = os.path.basename(run_dir)
            model, meta = load_run(run_dir, dev)
            P, C, S, dmae = predict(model, meta, ra_, tidx, dev)
            m = outcome_metrics(rt.iloc[tidx], P, C, dmae)
            off[name] = dict(detach_auroc=m["detach"]["auroc"], success_ap=m["success"]["ap"],
                             contact_auroc=m["contact"]["auroc"], ece_mean=m["ece_mean"])
        res["offline_random_test"] = off
    except Exception as ex:
        log(f"offline comparison skipped: {ex}")
    write_json(os.path.join(rdir("cf"), "analysis.json"), res)
    log("wrote results/cf/analysis.json")


def control_jobs(cfg):
    from .eval_control import NOMINAL
    base = int(cfg["eval"]["seed_base"])
    n = int(cfg.get("counterfactual", {}).get("eval_seeds", 16))
    return [dict(id=f"cf__wm_fullcf_cal__{base + i}", controller="wm_fullcf_cal", seed=base + i, **NOMINAL)
            for i in range(n)]


def control_worker(args):
    from .eval_control import light_for, job_metrics, claim
    from .policies import make_policy
    from .rollout import run_batch
    cfg = load_config(args.config)
    ev = cfg["eval"]
    log = logger(f"cf ctrl w{args.worker_id}")
    jdir = rdir("cf", "jobs")
    cdir = rdir("cf", f"claims_{args.run_token}")
    pol = None
    for jb in control_jobs(cfg):
        res = os.path.join(jdir, jb["id"] + ".json")
        if os.path.exists(res) or not claim(os.path.join(cdir, jb["id"] + ".claim")):
            continue
        if pol is None:
            pol = make_policy("wm_fullcf_cal", runs_root=os.path.join(RESULTS, "runs"),
                              device="cuda" if device_name() == "cuda" else "cpu")
        pol.latency_ms = []
        light, el = light_for(jb["seed"], jb["light"])
        spec = dict(seed=jb["seed"], num_envs=int(ev["num_envs"]), foliage=jb["foliage"], apples=jb["apples"],
                    light_dir=light, light_el=el, shadows=False, frames=int(ev["frames"]), device=device_name(),
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
        write_json(res, dict(job=jb, metrics=m, summary=summary))
        log(f"  -> harvest {m['harvest_success']:.1f}% contact {m['contact_rate']:.1f}% "
            f"fruit/robot/min {m['throughput']:.3f} return {m['safe_return']:.3f}")


def control(args):
    from .launch import gpus, spawn, wait_all
    cfg = load_config(args.config)
    log = logger("cf control")
    G = gpus()
    todo = [j for j in control_jobs(cfg) if not os.path.exists(os.path.join(RESULTS, "cf", "jobs", j["id"] + ".json"))]
    n = min(len(G) * int(cfg["eval"]["workers_per_gpu"]), len(todo))
    token = time.strftime("%Y%m%d_%H%M%S")
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-m", "owsim.counterfactual", "control_worker", "--config", args.config,
               "--worker-id", str(i), "--run-token", token]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rdir("logs"), f"cf_ctrl_w{i:02d}.log")))
        time.sleep(2.0)
    log(f"{len(todo)} closed-loop jobs, {n} workers")
    wait_all(procs, log)


def _load_jobs(*patterns):
    out = {}
    for p in patterns:
        for f in glob.glob(p):
            d = read_json(f)
            if d and "metrics" in d:
                out[d["job"]["seed"]] = d["metrics"]
    return out


def report(args):
    log = logger("cf report")
    out = rdir("cf")
    A = read_json(os.path.join(out, "analysis.json"), {})
    L = ["# Counterfactual rollouts (checkpoint/restore) — results", ""]
    if A:
        L += [f"- {A['n_groups']} complete groups (7 branches each) from {A['n_seeds']} tree seeds; "
              f"splits {A['per_split']}; decision quality on {A['n_test_groups']} test groups.", "",
              "## 1. Noise ceiling: pick-center repeated from the identical state", "",
              "| outcome | agreement | Cohen's kappa | rate (1st) | rate (repeat) |", "|---|---|---|---|---|"]
        for col in ("success", "detach", "contact", "damage"):
            r = A["noise_ceiling"][col]
            L.append(f"| {col} | {r['agreement']:.2f} | {r['kappa']:.2f} | {r['rate_first']:.2f} | {r['rate_repeat']:.2f} |")
        L += ["", f"Reward correlation between the two identical trials: {A['noise_ceiling']['reward_corr']:.2f}; "
                  f"mean |difference| {A['noise_ceiling']['reward_abs_diff_mean']:.2f}.", "",
              "## 2. Paired effect of each action vs pick-center (same state; mean [95 % CI], tree-seed bootstrap)", "",
              "| action | Δ success | Δ detach | Δ contact | Δ damage | Δ reward |", "|---|---|---|---|---|---|"]
        f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
        for a, d in A["paired_effects_vs_center"].items():
            L.append(f"| {a} | {f(d['success'])} | {f(d['detach'])} | {f(d['contact'])} | {f(d['damage'])} | {f(d['reward'])} |")
        L += ["", "(The 'repeat' row is the noise level: any action effect of similar size is not distinguishable "
                  "from re-running the same action.)", "",
              "## 3. Decision quality on test groups (realized counterfactual outcomes)", "",
              "| policy | reward / decision | regret vs oracle | best-action rate | skip % | success / pick | contact / action |",
              "|---|---|---|---|---|---|---|"]

        import re
        agg = {}
        for name, d in A["decision_quality_test"].items():
            m = re.match(r"world model (.+)_s\d+(_cal)?$", name)
            if not m:
                L.append(f"| {name} | {f(d['reward'])} | {f(d['regret'])} | {d['best_action_rate']:.2f} | "
                         f"{100 * d['skip_rate']:.0f} | {d['success_per_pick']:.2f} | {d['contact_per_action']:.2f} |")
            else:
                agg.setdefault(f"world model {m.group(1)}{' (calibrated)' if m.group(2) else ''}", []).append(d)
        rng_ = lambda v: f"{np.nanmean(v):+.3f} ({np.nanmin(v):+.3f}..{np.nanmax(v):+.3f})"
        for name, ds in agg.items():
            g = lambda q, i=0: [x[q][i] if isinstance(x[q], list) else x[q] for x in ds]
            L.append(f"| {name} [{len(ds)} seeds] | {rng_(g('reward'))} | {rng_(g('regret'))} | "
                     f"{np.nanmean(g('best_action_rate')):.2f} | {100 * np.nanmean(g('skip_rate')):.0f} | "
                     f"{np.nanmean(g('success_per_pick')):.2f} | {np.nanmean(g('contact_per_action')):.2f} |")
        L += ["", "(Baselines: mean [95 % CI]; world models: mean over training seeds (min..max).)", "",
              "## 4. Within-state outcome discrimination across actions (test groups; mean over seeds)", "",
              "| model | success AUROC (all actions) | success AUROC (3 picks) | contact AUROC (all) | contact AUROC (3 picks) |",
              "|---|---|---|---|---|"]
        dagg = {}
        for name, d in A.get("counterfactual_discrimination_test", {}).items():
            m = re.match(r"((.+)_s\d+)(_cal)?$", name)
            if m:
                dagg.setdefault(m.group(2), {}).setdefault(m.group(1), d)
        dagg = {k: list(v.values()) for k, v in dagg.items()}
        for name, ds in dagg.items():
            g = lambda q: np.nanmean([x[q] for x in ds])
            L.append(f"| {name} [{len(ds)} seeds] | {g('success_auroc_all'):.3f} | {g('success_auroc_picks'):.3f} | "
                     f"{g('contact_auroc_all'):.3f} | {g('contact_auroc_picks'):.3f} |")
        if A.get("offline_random_test"):
            L += ["", "## 5. Prediction on the random-action test trees: full vs fullcf (trained with counterfactual data)", "",
                  "| run | detach AUROC | success AP | contact AUROC | ECE (uncal.) |", "|---|---|---|---|---|"]
            for name, d in A["offline_random_test"].items():
                L.append(f"| {name} | {d['detach_auroc']:.3f} | {d['success_ap']:.3f} | {d['contact_auroc']:.3f} | {d['ece_mean']:.3f} |")

    J = os.path.join(RESULTS, "control", "jobs")
    E = os.path.join(RESULTS, "extras", "jobs")
    S16 = os.path.join(RESULTS, "seeds16", "jobs")
    cfj = _load_jobs(os.path.join(out, "jobs", "cf__*.json"))
    if cfj:
        refs = {"world model, calibrated": _load_jobs(os.path.join(E, "extras__wm_full_cal__val__*.json"),
                                                      os.path.join(S16, "s16__wm_cal__*.json")),
                "detect-and-pick": _load_jobs(os.path.join(J, "main__nominal__detect_and_pick__*.json"),
                                              os.path.join(S16, "s16__detect_and_pick__*.json"))}
        L += ["", "## 6. Closed loop (nominal, paired seeds): world model trained with counterfactual data (calibrated)", "",
              "| controller | n | harvest/pick % | contact % | fruit/robot/min | safe return |", "|---|---|---|---|---|---|"]
        mean = lambda d, q: float(np.nanmean([v[q] for v in d.values()]))
        for name, d in [("world model + counterfactual data, calibrated", cfj)] + list(refs.items()):
            if d:
                L.append(f"| {name} | {len(d)} | {mean(d, 'harvest_success'):.1f} | {mean(d, 'contact_rate'):.1f} | "
                         f"{mean(d, 'throughput'):.3f} | {mean(d, 'safe_return'):.3f} |")
        for name, d in refs.items():
            s = sorted(set(cfj) & set(d))
            if len(s) >= 2:
                for q in ("harvest_success", "contact_rate", "throughput", "safe_return"):
                    dl = np.array([cfj[x][q] - d[x][q] for x in s])
                    dl = dl[np.isfinite(dl)]
                    if len(dl) >= 2:
                        t = dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)) + 1e-12)
                        L.append(f"- vs {name}, {q}: {dl.mean():+.3f} (t={t:+.2f}, {int((dl > 0).sum())}/{len(dl)} positive)")
    L += ["", "GPU physics is not bit-reproducible; compare effects with the noise ceiling (section 1)."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "worker", "dataset", "train", "analyze", "control",
                                     "control_worker", "report"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--run-token", default="local")
    args = ap.parse_args()
    fn = {"collect": collect, "worker": worker, "dataset": dataset, "train": train, "analyze": analyze,
          "control": control, "control_worker": control_worker, "report": report}[args.mode]
    fn(args)


if __name__ == "__main__":
    main()
