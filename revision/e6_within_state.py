from __future__ import annotations

import os

import numpy as np

from owsim.common import RESULTS, logger, write_json
from revision._common import rev_dir

PICK_BRANCHES = [1, 2, 3]


def concordance(P, Y, higher_is_positive=True):
    out = np.full(len(P), np.nan)
    for g, (p, y) in enumerate(zip(P, Y)):
        c, m = 0.0, 0
        for i in range(3):
            for j in range(3):
                if y[i] == 1 and y[j] == 0:
                    m += 1
                    d = (p[i] - p[j]) if higher_is_positive else (p[j] - p[i])
                    c += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        if m:
            out[g] = c / m
    return out


def main():
    import torch
    from owsim.counterfactual import _groups, _score_models, _cluster_boot, CENTER1
    from owsim.dataset import load
    log = logger("e6")
    out = rev_dir("e6_within_state")
    rng = np.random.default_rng(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load(os.path.join(RESULTS, "cf", "dataset"))
    G = _groups(table)
    keys = sorted(G)
    seeds = np.array([table["seed"].values[G[k][0]] for k in keys])
    split = np.array([table["split"].values[G[k][0]] for k in keys])
    te = np.nonzero(split == "test")[0]
    assert PICK_BRANCHES[0] == CENTER1
    Y = {col: np.array([[float(table[col].values[G[k][b]]) for b in PICK_BRANCHES] for k in keys])
         for col in ("success", "contact")}
    rows = np.array([G[k][0] for k in keys])
    scores = _score_models(table, arrs, rows, dev, log)


    per_run = {}
    for name, d in sorted(scores.items(), key=lambda kv: kv[1]["calibrated"]):
        per_run.setdefault(name[:-4] if d["calibrated"] else name, d)
    res = {}
    for name, d in per_run.items():
        P = d["P"][te]
        r = {}
        for col, oi, pos in (("success", 1, True), ("contact", 2, True)):
            cg = concordance(P[:, :3, oi], Y[col][te], higher_is_positive=pos)
            ok = np.isfinite(cg)
            r[col] = dict(mean_ci=_cluster_boot(cg[ok], seeds[te][ok], rng), n_groups=int(ok.sum()))
        res[name] = r
        log(f"{name}: success {r['success']['mean_ci'][0]:.3f} (n={r['success']['n_groups']}), "
            f"contact {r['contact']['mean_ci'][0]:.3f}")
    write_json(os.path.join(out, "within_state.json"), res)
    agg = {}
    for name, r in res.items():
        v = name.rsplit("_s", 1)[0]
        agg.setdefault(v, []).append(r)
    L = ["# E6 - within-state concordance among the three picks (counterfactual test groups)", "",
         "Per-group fraction of correctly ordered (positive, negative) action pairs; 0.5 = chance "
         "(the no-action model must give 0.5). Mean over training seeds; CI of the first seed in brackets.", "",
         "| model | success concordance | groups | contact concordance | groups |", "|---|---|---|---|---|"]
    for v, rs in agg.items():
        s = np.mean([x["success"]["mean_ci"][0] for x in rs])
        c = np.mean([x["contact"]["mean_ci"][0] for x in rs])
        ci_s, ci_c = rs[0]["success"]["mean_ci"], rs[0]["contact"]["mean_ci"]
        L.append(f"| {v} [{len(rs)} seeds] | {s:.3f} [{ci_s[1]:.3f}, {ci_s[2]:.3f}] | {rs[0]['success']['n_groups']} | "
                 f"{c:.3f} [{ci_c[1]:.3f}, {ci_c[2]:.3f}] | {rs[0]['contact']['n_groups']} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md")


if __name__ == "__main__":
    main()
