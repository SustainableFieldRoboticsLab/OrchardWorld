from __future__ import annotations

import glob
import json
import os

import numpy as np

from .common import RESULTS, rdir, logger, write_json, split_of
from .rollout import SCALARS, PRIV, OPTIONAL

IMG_KEYS = ["g_rgb", "g_dep", "c_rgb", "c_dep", "state", "det", "uv", "pose_ok"] + PRIV


def build(out_dir=None, shard_dirs=None):
    import pandas as pd
    log = logger("dataset")
    shard_dirs = shard_dirs or [os.path.join(RESULTS, "collect", "shards")]
    files = sorted(f for sd in shard_dirs for f in glob.glob(os.path.join(sd, "b*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        raise SystemExit(f"no shards in {shard_dirs}")
    out_dir = out_dir or rdir("dataset")

    rows = []
    shapes = {}
    total = 0
    good = []
    for si, f in enumerate(files):
        try:
            d = np.load(f)
            n = len(d["action"])
            meta = json.load(open(f.replace(".npz", ".json")))
        except Exception as ex:
            log(f"skip unreadable shard {f}: {ex}")
            continue
        if n == 0 or "pre_g_rgb" not in d.files:
            continue
        if len(d["obs_index"]) != n:
            log(f"skip {f}: images missing for some records")
            continue
        spec = meta["spec"]
        tab = {k: d[k] for k in SCALARS}
        for k in OPTIONAL:
            tab[k] = d[k] if k in d.files else np.full(n, -1, dtype=np.int32)
        tab.update({f"pre_{k}": d[f"pre_{k}"] for k in PRIV})
        tab["pre_det"] = d["pre_det"]
        tab["pre_pose_ok"] = d["pre_pose_ok"]
        df = pd.DataFrame(tab)
        df["shard"] = len(good)
        df["seed"] = spec["seed"]
        df["split"] = spec.get("split") or split_of(spec["seed"])
        df["foliage"] = spec["foliage"]
        df["apples"] = spec["apples"]
        df["light_el"] = spec.get("light_el", np.nan)
        df["batch_status"] = meta["status"]
        rows.append(df)
        for tag in ("pre", "post"):
            for k in IMG_KEYS:
                a = d[f"{tag}_{k}"]
                shapes[f"{tag}_{k}"] = (a.shape[1:], a.dtype)
        good.append((f, n, meta))
        total += n
    table = pd.concat(rows, ignore_index=True)

    mm = {k: np.lib.format.open_memmap(os.path.join(out_dir, f"{k}.npy"), mode="w+",
                                       dtype=dt, shape=(total,) + sh)
          for k, (sh, dt) in shapes.items()}
    o = 0
    for f, n, _ in good:
        d = np.load(f)
        for k in mm:
            mm[k][o:o + n] = d[k]
        o += n
    for k in mm:
        mm[k].flush()

    key = list(zip(table["shard"], table["env"], table["target_uid"], table["dec_idx"]))
    pos = {k: i for i, k in enumerate(key)}
    table["next_idx"] = [pos.get((s, e, t, di + 1), -1) for (s, e, t, di) in key]
    table.to_csv(os.path.join(out_dir, "table.csv"), index=False)
    metas = [m for _, _, m in good]
    stats = dict(
        n_decisions=int(total), n_batches=len(good),
        n_trees=int(sum(m["envs"] for m in metas)),
        n_seeds=int(table["seed"].nunique()),
        per_split={sp: int((table["split"] == sp).sum()) for sp in ("train", "val", "test")},
        per_action={int(a): int((table["action"] == a).sum()) for a in sorted(table["action"].unique())},
        env_steps=int(sum(m["env_steps"] for m in metas)),
        sim_hours=float(sum(m["sim_s"] * m["envs"] for m in metas) / 3600.0),
        wall_hours_sum=float(sum(m["wall_s"] + m["build_s"] for m in metas) / 3600.0),
    )
    write_json(os.path.join(out_dir, "stats.json"), stats)
    log(f"dataset: {total} decisions from {len(good)} batches -> {out_dir}")
    log(json.dumps(stats))
    return stats


def load(ds_dir=None, keys=None):
    import pandas as pd
    ds_dir = ds_dir or os.environ.get("OW_DATASET_DIR") or os.path.join(RESULTS, "dataset")
    table = pd.read_csv(os.path.join(ds_dir, "table.csv"))
    arrs = {}
    for f in glob.glob(os.path.join(ds_dir, "*.npy")):
        k = os.path.basename(f)[:-4]
        if keys is None or k in keys:
            arrs[k] = np.load(f, mmap_mode="r")
    return table, arrs


if __name__ == "__main__":
    build()
