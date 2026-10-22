from __future__ import annotations

import os

import numpy as np

from owsim.common import RESULTS, logger, write_json, read_json
from revision._common import rev_dir

GRID = np.round(np.arange(-0.3, 0.41, 0.025), 3)


def snips_ess(table_sub, scores6, theta):
    from owsim.decision_picker import SKIP
    best = np.argmax(scores6, axis=1)
    best = np.where(scores6.max(1) < theta, SKIP, best)
    a = table_sub["action"].values
    mu = table_sub["behavior_p"].values
    r = table_sub["reward"].values
    w = (best == a) / np.maximum(mu, 1e-6)
    sw, sw2 = float(w.sum()), float((w ** 2).sum())
    return dict(value=float(np.sum(w * r) / max(sw, 1e-6)), ess=(sw * sw / sw2) if sw2 > 0 else 0.0,
                n_matched=int((w > 0).sum()), n_rows=int(len(a)),
                skip_share=float(np.mean(best == SKIP)))


def main():
    import torch
    from owsim.dataset import load
    from owsim.eval_offline import predict
    from owsim.extras import _splits, _finite_rows
    from owsim.model import load_run
    from owsim.policies import ReactiveSelect
    log = logger("ess")
    out = rev_dir("ess")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, arrs = load()
    arrs = dict(arrs)
    arrs["__action"] = table["action"].values.astype(np.int64)
    _, _, _, va_all = _splits(table)
    va_all = _finite_rows(arrs, va_all)
    tv = table.iloc[va_all]
    log(f"validation rows (incl. skip): {len(va_all)}")
    res = {}
    for s in (0, 1, 2):
        run = os.path.join(RESULTS, "runs", f"full_s{s}")
        for cal in (False, True):
            pol = read_json(os.path.join(run, "policy_cal.json" if cal else "policy.json"))
            if pol is None:
                continue
            model, meta = load_run(run, dev, calibrated=cal)
            _, _, Sv, _ = predict(model, meta, arrs, va_all, dev)
            S6 = np.concatenate([Sv, np.full((len(Sv), 1), -np.inf)], 1)
            curve = {float(t): snips_ess(tv, S6, t) for t in GRID}
            th = float(pol["theta"])
            key = f"full_s{s}" + ("_cal" if cal else "")
            res[key] = dict(theta=th, at_theta=snips_ess(tv, S6, th), curve=curve)
            log(f"{key}: theta {th:+.3f}  ESS {res[key]['at_theta']['ess']:.0f} "
                f"(matched {res[key]['at_theta']['n_matched']} of {len(va_all)})")
            del model

    rs_path = os.path.join(RESULTS, "extras", "reactive_select.json")
    if os.path.exists(rs_path):
        rsel = ReactiveSelect(rs_path)
        sc = np.array([rsel.score(x) for x in np.asarray(arrs["pre_state"][va_all])])
        S6 = np.full((len(sc), 6), -np.inf)
        S6[:, 0] = sc
        res["selective_detect_and_pick"] = dict(theta=rsel.theta, at_theta=snips_ess(tv, S6, rsel.theta),
                                                curve={float(t): snips_ess(tv, S6, t) for t in GRID})
    write_json(os.path.join(out, "ess.json"), res)
    L = ["# SNIPS effective sample size at the validation-selected threshold", "",
         f"Validation rows (including skips): {len(va_all)}.", "",
         "| policy | theta | SNIPS value | ESS | matched rows | skip share |", "|---|---|---|---|---|---|"]
    for k, d in res.items():
        a = d["at_theta"]
        L.append(f"| {k} | {d['theta']:+.3f} | {a['value']:+.3f} | {a['ess']:.0f} | {a['n_matched']} | "
                 f"{100 * a['skip_share']:.0f}% |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md")


if __name__ == "__main__":
    main()
