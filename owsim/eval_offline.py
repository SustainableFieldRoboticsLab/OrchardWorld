from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from .common import RESULTS, rdir, logger, write_json
from .dataset import load
from .decision_picker import PICKS, SKIP, STATE_NAMES
from .model import (load_run, BIN_HEADS, plan_scores, future_depth_target, prep_images)
from .simenv import DEPTH_MAX
from .train import labels_from_table


def ece_score(y, p, bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = dict(n=int(len(y)), pos_rate=float(y.mean()) if len(y) else float("nan"))
    if len(np.unique(y)) == 2:
        out.update(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))
    else:
        out.update(auroc=float("nan"), ap=float("nan"))
    out.update(brier=float(brier_score_loss(y, p)) if len(y) else float("nan"),
               ece=ece_score(y, p))
    return out


def outcome_metrics(table_sub, P, C=None, dep_mae_cm=None):
    pick = table_sub["action"].isin(PICKS).values
    m = dict(
        detach=binary_metrics(table_sub["detach"].values[pick], P[pick, 0]),
        success=binary_metrics(table_sub["success"].values[pick], P[pick, 1]),
        contact=binary_metrics(table_sub["contact"].values, P[:, 2]),
        damage=binary_metrics(table_sub["damage"].values, P[:, 3]),
        visible=binary_metrics(table_sub["visible_after"].values, P[:, 4]),
    )
    m["ece_mean"] = float(np.nanmean([m[k]["ece"] for k in ("detach", "success", "contact")]))
    if C is not None:
        m["tau_mae_s"] = float(np.nanmean(np.abs(C[:, 0] * 10 - table_sub["tau"].values)))
    if dep_mae_cm is not None:
        m["future_depth_mae_cm"] = float(dep_mae_cm)
    return m


@torch.no_grad()
def predict(model, meta, arrs, idx, dev, bs=512):
    mean, std = meta["state_mean_t"], meta["state_std_t"]
    P, C, S, dep_err, dep_n = [], [], [], 0.0, 0
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        T = lambda k, dt=None: torch.as_tensor(np.array(arrs[k][j]), device=dev, dtype=dt)
        g, c = prep_images(model, T("pre_g_rgb"), T("pre_g_dep", torch.float32),
                           T("pre_c_rgb"), T("pre_c_dep", torch.float32),
                           T("pre_uv", torch.float32))
        s = (T("pre_state", torch.float32) - mean) / std
        a = T("__action", torch.long) if "__action" in arrs else None
        vg = vc = None
        if model.use_vjepa:
            vg, vc = T("vj_pre_g", torch.float16), T("vj_pre_c", torch.float16)
        h = model.encode(g, c, s, vg, vc)
        h1 = model.step(h, a)
        lb, lc = model.heads(h1)
        P.append(torch.sigmoid(lb).float().cpu().numpy())
        C.append(lc.float().cpu().numpy())
        S.append(plan_scores(model, h).float().cpu().numpy())
        tgt = future_depth_target(T("post_g_dep", torch.float32))
        pred = model.future_depth(h1).float().clamp(0, 1)
        valid = tgt < 0.995
        dep_err += float((pred - tgt).abs()[valid].sum()) * DEPTH_MAX * 100.0
        dep_n += int(valid.sum())
    return (np.concatenate(P), np.concatenate(C), np.concatenate(S),
            dep_err / max(dep_n, 1))


@torch.no_grad()
def latency_ms(model, meta, arrs, dev, reps=100, vj=None):
    from .model import score_observation
    i = 0
    obs = dict(g_rgb=arrs["pre_g_rgb"][i], g_dep=np.asarray(arrs["pre_g_dep"][i], np.float32),
               c_rgb=arrs["pre_c_rgb"][i], c_dep=np.asarray(arrs["pre_c_dep"][i], np.float32),
               uv=arrs["pre_uv"][i], state=arrs["pre_state"][i])
    for _ in range(10):
        score_observation(model, meta, obs, dev, vj)
    ts = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        score_observation(model, meta, obs, dev, vj)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts))


def ips_value(table_sub, scores, theta, allow_skip=True):
    best = np.argmax(scores, axis=1)
    if allow_skip:
        best = np.where(scores.max(1) < theta, SKIP, best)
    a = table_sub["action"].values
    mu = table_sub["behavior_p"].values
    r = table_sub["reward"].values
    w = (best == a) / np.maximum(mu, 1e-6)
    return float(np.sum(w * r) / max(np.sum(w), 1e-6))


def reactive_baseline(table, arrs, tr, te, log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cols = [STATE_NAMES.index(c) for c in
            ["tgt_bx", "tgt_by", "tgt_bz", "tgt_cx", "tgt_cy", "tgt_cz", "det_found",
             "det_radius", "det_log_npix", "det_rms_cm", "tgt_u", "tgt_v", "tgt_dist",
             "tgt_height", "n_dets"]]
    Xtr = np.asarray(arrs["pre_state"][tr])[:, cols]
    Xte = np.asarray(arrs["pre_state"][te])[:, cols]
    ybtr, _ = labels_from_table(table.iloc[tr])
    P = np.zeros((len(te), len(BIN_HEADS)), dtype=np.float32)
    t0 = time.perf_counter()
    for k in range(len(BIN_HEADS)):
        y = ybtr[:, k]
        if len(np.unique(y)) < 2:
            P[:, k] = y.mean()
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(Xtr, y)
        P[:, k] = clf.predict_proba(Xte)[:, 1]
    rt = 1e3 * (time.perf_counter() - t0) / max(len(te), 1)
    log(f"reactive baseline fitted on {len(tr)} rows")
    return P, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = logger("eval_offline")
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out = rdir("offline")
    table, arrs = load()
    arrs = dict(arrs)
    arrs["__action"] = table["action"].values.astype(np.int64)
    usable = (table["skipped"] == 0).values & (table["batch_status"].fillna("nan") != "nan").values
    split = table["split"].values
    te = np.nonzero(usable & (split == "test"))[0]
    va_all = np.nonzero((split == "val") & (table["batch_status"].fillna("nan") != "nan").values)[0]
    tr = np.nonzero(usable & (split == "train"))[0]
    if len(te) < 20:
        log("WARNING: very few test rows; using validation rows as test (smoke run?)")
        te = np.nonzero(usable & (split != "train"))[0]
        if len(te) < 5:
            te = tr
    if len(va_all) < 10:
        va_all = np.nonzero(split != "test")[0]
    log(f"test rows {len(te)}  val rows (incl. skip) {len(va_all)}")
    summary = {}

    Pb, rt = reactive_baseline(table, arrs, tr, te, log)
    mb = outcome_metrics(table.iloc[te], Pb)
    mb["runtime_ms"] = rt
    summary["reactive"] = dict(runs={"reactive": mb})
    np.savez_compressed(os.path.join(out, "preds_reactive.npz"), idx=te, P=Pb)

    for run_dir in sorted(glob.glob(os.path.join(RESULTS, "runs", "*_s*"))):
        name = os.path.basename(run_dir)
        if not os.path.exists(os.path.join(run_dir, "best.pt")):
            continue
        try:
            model, meta = load_run(run_dir, dev)
        except Exception as ex:
            log(f"skipping {name}: cannot load ({type(ex).__name__}: {str(ex)[:120]})")
            continue
        variant = meta["variant"]
        P, C, S, dmae = predict(model, meta, arrs, te, dev)
        m = outcome_metrics(table.iloc[te], P, C, dmae)
        vj = None
        if model.use_vjepa:
            from .vjepa import VJEPAEncoder
            vj = VJEPAEncoder(device=str(dev))
        m["runtime_ms"] = latency_ms(model, meta, arrs, dev, vj=vj)
        del vj
        m["params_M"] = meta["params"] / 1e6

        Pv, Cv, Sv, _ = predict(model, meta, arrs, va_all, dev)
        Sv6 = np.concatenate([Sv, np.full((len(Sv), 1), -np.inf)], 1)
        grid = np.round(np.arange(-0.3, 0.41, 0.025), 3)
        vals = [ips_value(table.iloc[va_all], Sv6, th) for th in grid]
        k = int(np.nanargmax(vals))
        pol = dict(theta=float(grid[k]), ips_value=float(vals[k]),
                   ips_curve=dict(theta=grid.tolist(), value=vals),
                   ips_always_pick_center=float(np.mean(
                       table.iloc[va_all]["reward"].values[table.iloc[va_all]["action"].values == 0]))
                   if (table.iloc[va_all]["action"].values == 0).any() else float("nan"))
        write_json(os.path.join(run_dir, "policy.json"), pol)
        m["theta"] = pol["theta"]
        summary.setdefault(variant, dict(runs={}))["runs"][name] = m
        np.savez_compressed(os.path.join(out, f"preds_{name}.npz"), idx=te, P=P, C=C, S=S)
        log(f"{name}: detAUC {m['detach']['auroc']:.3f} succAP {m['success']['ap']:.3f} "
            f"contactAUC {m['contact']['auroc']:.3f} ECE {m['ece_mean']:.3f} "
            f"depthMAE {dmae:.1f}cm  {m['runtime_ms']:.1f}ms  theta {pol['theta']}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    write_json(os.path.join(out, "offline_metrics.json"), summary)
    log(f"wrote {os.path.join(out, 'offline_metrics.json')}")


if __name__ == "__main__":
    main()
