from __future__ import annotations

import os
import time

import numpy as np

from owsim.common import RESULTS, logger, write_json, read_json
from revision._common import rev_dir

N_PICK_REPOS = 5


def _onehot(a):
    X = np.zeros((len(a), N_PICK_REPOS), dtype=np.float32)
    X[np.arange(len(a)), a] = 1.0
    return X


def _fit_predict(kind, Xtr, Ytr, Xte, log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    P = np.zeros((len(Xte), Ytr.shape[1]), dtype=np.float32)
    for k in range(Ytr.shape[1]):
        y = Ytr[:, k]
        if len(np.unique(y)) < 2:
            P[:, k] = y.mean()
            continue
        if kind == "lr":
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(Xtr, y)
        else:

            pos = float(y.mean())
            sw = np.where(y > 0.5, 0.5 / pos, 0.5 / (1.0 - pos))
            clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 validation_fraction=0.1, random_state=0)
            clf.fit(Xtr, y, sample_weight=sw)
        P[:, k] = clf.predict_proba(Xte)[:, 1]
    return P


def main():
    from owsim.dataset import load
    from owsim.decision_picker import STATE_NAMES
    from owsim.eval_offline import outcome_metrics
    from owsim.train import labels_from_table
    from owsim.extras import DET_COLS
    log = logger("e0")
    out = rev_dir("e0_baselines")
    table, arrs = load()

    usable = (table["skipped"] == 0).values & (table["batch_status"].fillna("nan") != "nan").values
    split = table["split"].values
    tr = np.nonzero(usable & (split == "train"))[0]
    te = np.nonzero(usable & (split == "test"))[0]
    S = np.asarray(arrs["pre_state"], dtype=np.float64)
    if not np.isfinite(S[np.concatenate([tr, te])]).all():
        raise SystemExit("non-finite pre_state rows in train/test - unexpected; stop and check")
    det = [STATE_NAMES.index(c) for c in DET_COLS]
    act = table["action"].values.astype(int)
    yb, _ = labels_from_table(table)
    log(f"train {len(tr)} test {len(te)} rows")

    def feats(name, idx):
        D, F, A = S[idx][:, det], S[idx], _onehot(act[idx])
        if name == "R0":
            return D
        if name == "A1":
            return np.concatenate([D, A], 1)
        if name == "A2":
            inter = (D[:, :, None] * A[:, None, :]).reshape(len(idx), -1)
            return np.concatenate([D, A, inter], 1)
        if name == "A3":
            return np.concatenate([F, A], 1)
        if name == "A4":
            return F
        if name == "A5":
            return np.concatenate([F, A], 1)
        raise ValueError(name)

    models = {"R0": ("lr", "Detector features, no action (paper baseline, re-fit)"),
              "A1": ("lr", "LR: detector features + action"),
              "A2": ("lr", "LR: detector features + action + feature x action"),
              "A3": ("lr", "LR: 54-d state + action"),
              "A4": ("hgb", "Gradient-boosted trees: 54-d state, no action"),
              "A5": ("hgb", "Gradient-boosted trees: 54-d state + action")}
    res = {}
    for name, (kind, label) in models.items():
        t0 = time.time()
        P = _fit_predict(kind, feats(name, tr), yb[tr], feats(name, te), log)
        m = outcome_metrics(table.iloc[te], P)
        m["label"], m["fit_s"] = label, time.time() - t0
        res[name] = m
        np.savez_compressed(os.path.join(out, f"preds_{name}.npz"), idx=te, P=P)
        log(f"{name}: detAUC {m['detach']['auroc']:.3f} succAP {m['success']['ap']:.3f} "
            f"contactAUC {m['contact']['auroc']:.3f} damageAUC {m['damage']['auroc']:.3f} ({m['fit_s']:.0f}s)")

    ref = read_json(os.path.join(RESULTS, "offline", "offline_metrics.json"), {})
    wm = {}
    for variant in ("full", "no_action", "rgbd", "tdmpc"):
        runs = ref.get(variant, {}).get("runs", {})
        if runs:
            g = lambda h, q: [r[h][q] for r in runs.values()]
            wm[variant] = {h + "_" + q: (float(np.mean(g(h, q))), float(np.std(g(h, q))))
                           for h, q in (("detach", "auroc"), ("success", "ap"), ("contact", "auroc"),
                                        ("damage", "auroc"))}
    write_json(os.path.join(out, "metrics.json"), dict(baselines=res, world_model_reference=wm,
                                                       n_train=int(len(tr)), n_test=int(len(te))))
    base = res["R0"]["success"]["pos_rate"]
    L = ["# E0 - baselines with action input (same test rows as Table 2)", "",
         f"Test: {len(te)} non-skip decisions; success base rate on picks {base:.3f} (= chance AP).", "",
         "| model | detach AUROC | success AP (x chance) | contact AUROC | damage AUROC |",
         "|---|---|---|---|---|"]
    for name, m in res.items():
        L.append(f"| {name} {m['label']} | {m['detach']['auroc']:.3f} | {m['success']['ap']:.3f} "
                 f"({m['success']['ap'] / base:.2f}x) | {m['contact']['auroc']:.3f} | {m['damage']['auroc']:.3f} |")
    for v, d in wm.items():
        L.append(f"| world model {v} (mean of 3 seeds, results/offline) | {d['detach_auroc'][0]:.3f} | "
                 f"{d['success_ap'][0]:.3f} ({d['success_ap'][0] / base:.2f}x) | {d['contact_auroc'][0]:.3f} | "
                 f"{d['damage_auroc'][0]:.3f} |")
    L += ["", "R0 must reproduce the paper's detector baseline (0.664 / 0.245 / 0.553); if it does not, "
              "stop and report - the row selection differs."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md")


if __name__ == "__main__":
    main()
