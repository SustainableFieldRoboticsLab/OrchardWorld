from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from owsim.common import RESULTS, logger, write_json
from revision._common import rev_dir

EXP = "e2_oracle"
EXTRA = ["stem_strength", "occlusion"]
STATE_DIM_ORACLE = 54 + len(EXTRA)


def ds_dir():
    return rev_dir(EXP, "dataset")


def root_dir():
    return rev_dir(EXP, "root")


def build(args):
    log = logger("e2 build")
    src = os.path.join(RESULTS, "dataset")
    dst = ds_dir()
    for f in glob.glob(os.path.join(src, "*")):
        name = os.path.basename(f)
        if name in ("pre_state.npy", "post_state.npy"):
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(f), link)
    for tag in ("pre", "post"):
        st = np.load(os.path.join(src, f"{tag}_state.npy"))
        if st.shape[1] != 54:
            raise SystemExit(f"{tag}_state has {st.shape[1]} columns, expected 54")
        cols = [np.load(os.path.join(src, f"{tag}_{k}.npy")).astype(np.float32)[:, None] for k in EXTRA]
        ext = np.concatenate([st.astype(np.float32)] + cols, 1)
        np.save(os.path.join(dst, f"{tag}_state.npy"), ext)
        log(f"{tag}_state: {st.shape} -> {ext.shape}")
    log(f"dataset view in {dst}")


def train(args):
    from owsim.launch import gpus, spawn
    log = logger("e2 train")
    if not os.path.exists(os.path.join(ds_dir(), "pre_state.npy")):
        raise SystemExit("run 'python -m revision.e2_oracle build' first")
    os.environ["OW_RESULTS"] = root_dir()
    os.environ["OW_DATASET_DIR"] = ds_dir()
    G = gpus()
    procs = []
    for i, s in enumerate((0, 1, 2)):
        if os.path.exists(os.path.join(root_dir(), "runs", f"full_s{s}", "meta.json")):
            log(f"full_s{s} already trained")
            continue
        cmd = [sys.executable, "-m", "revision.e2_oracle", "train_one", "--seed", str(s)]
        procs.append(spawn(cmd, G[i % len(G)], os.path.join(rev_dir("logs"), f"e2_train_s{s}.log")))
    for p in procs:
        p.wait()
    bad = [p.args for p in procs if p.returncode != 0]
    if bad:
        raise SystemExit(f"{len(bad)} training run(s) failed - see results/revision/logs/e2_train_s*.log")
    log("done")


def train_one(args):


    if not os.environ.get("OW_DATASET_DIR", "").endswith(os.path.join(EXP, "dataset")):
        raise SystemExit("train_one must be started by 'python -m revision.e2_oracle train'")
    import owsim.model as M
    M.STATE_DIM = STATE_DIM_ORACLE
    from owsim import train as T
    sys.argv = ["owsim.train", "--config", args.config, "--variant", "full", "--seed", str(args.seed),
                "--device", "cuda"]
    T.main()


def _load(run_dir, dev, state_dim):
    import owsim.model as M
    old = M.STATE_DIM
    M.STATE_DIM = state_dim
    try:
        return M.load_run(run_dir, dev)
    finally:
        M.STATE_DIM = old


def evaluate(args):
    import torch
    from owsim.dataset import load
    from owsim.eval_offline import predict, outcome_metrics
    log = logger("e2 eval")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for label, ddir, rdir_, dim in (("standard", os.path.join(RESULTS, "dataset"), os.path.join(RESULTS, "runs"), 54),
                                    ("oracle", ds_dir(), os.path.join(root_dir(), "runs"), STATE_DIM_ORACLE)):
        table, arrs = load(ddir)
        arrs = dict(arrs)
        arrs["__action"] = table["action"].values.astype(np.int64)
        usable = (table["skipped"] == 0).values & (table["batch_status"].fillna("nan") != "nan").values
        te = np.nonzero(usable & (table["split"] == "test").values)[0]
        for s in (0, 1, 2):
            run = os.path.join(rdir_, f"full_s{s}")
            if not os.path.exists(os.path.join(run, "best.pt")):
                log(f"missing {run}")
                continue
            model, meta = _load(run, dev, dim)
            P, C, S, dmae = predict(model, meta, arrs, te, dev)
            m = outcome_metrics(table.iloc[te], P, C, dmae)
            res.setdefault(label, {})[f"full_s{s}"] = dict(
                detach_auroc=m["detach"]["auroc"], success_auroc=m["success"]["auroc"],
                success_ap=m["success"]["ap"], contact_auroc=m["contact"]["auroc"],
                damage_auroc=m["damage"]["auroc"], best_epoch=meta.get("best_epoch"))
            log(f"{label} full_s{s}: {res[label][f'full_s{s}']}")
            del model
    out = rev_dir(EXP)
    write_json(os.path.join(out, "metrics.json"), res)
    keys = ["detach_auroc", "success_auroc", "success_ap", "contact_auroc", "damage_auroc"]
    L = ["# E2 - oracle-feature probe (privileged stem strength + GT occlusion), test split", "",
         "Mean +- s.d. over 3 training seeds; same test rows as Table 2.", "",
         "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    agg = {}
    for label, runs in res.items():
        agg[label] = {k: (np.mean([r[k] for r in runs.values()]), np.std([r[k] for r in runs.values()], ddof=1))
                      for k in keys}
        L.append(f"| {label} | " + " | ".join(f"{agg[label][k][0]:.3f} +- {agg[label][k][1]:.3f}" for k in keys) + " |")
    if len(agg) == 2:
        L.append("| delta (oracle - standard) | " +
                 " | ".join(f"{agg['oracle'][k][0] - agg['standard'][k][0]:+.3f}" for k in keys) + " |")
    L += ["", "The 'standard' rows must match Table 2 (0.778 / 0.469 / 0.772); otherwise stop and report."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "train", "train_one", "eval"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"build": build, "train": train, "train_one": train_one, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
