from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .common import RESULTS, logger, write_json

KEYS = [("pre", "g"), ("pre", "c"), ("post", "g"), ("post", "c")]


def part_range(n, k, nparts):
    a = (n * k) // nparts
    b = (n * (k + 1)) // nparts
    return a, b


def run_part(k, nparts, device, bs=256):
    import torch
    from .vjepa import VJEPAEncoder, FEAT_DIM, CROP_GRID
    log = logger(f"features p{k}")
    ds = os.path.join(RESULTS, "dataset")
    out = os.path.join(ds, "vjepa_parts")
    os.makedirs(out, exist_ok=True)
    done_flag = os.path.join(out, f"part{k}.done")
    if os.path.exists(done_flag):
        log("already done")
        return
    arr = {f"{t}_{x}": np.load(os.path.join(ds, f"{t}_{x}_rgb.npy"), mmap_mode="r")
           for t in ("pre", "post") for x in ("g", "c")}
    n = len(arr["pre_g"])
    a, b = part_range(n, k, nparts)
    enc = VJEPAEncoder(device=device)
    res = {f"{t}_{x}": np.lib.format.open_memmap(
        os.path.join(out, f"part{k}_{t}_{x}.npy"), mode="w+", dtype=np.float16,
        shape=(b - a, FEAT_DIM) if x == "g" else (b - a, CROP_GRID * CROP_GRID, FEAT_DIM))
        for t, x in KEYS}
    t0 = time.time()
    for i in range(a, b, bs):
        j = min(i + bs, b)
        for t in ("pre", "post"):
            gv, cg = enc.encode(np.asarray(arr[f"{t}_g"][i:j]), np.asarray(arr[f"{t}_c"][i:j]))
            res[f"{t}_g"][i - a:j - a] = gv.cpu().numpy()
            res[f"{t}_c"][i - a:j - a] = cg.cpu().numpy()
        if (i - a) // bs % 20 == 0:
            el = time.time() - t0
            log(f"rows {j - a}/{b - a}  ({(j - a) / max(el, 1e-6):.0f} rows/s)")
    for m in res.values():
        m.flush()
    open(done_flag, "w").write(str(time.time() - t0))
    log(f"done {b - a} rows in {time.time() - t0:.0f}s")
    del torch


def merge(nparts):
    log = logger("features merge")
    ds = os.path.join(RESULTS, "dataset")
    out = os.path.join(ds, "vjepa_parts")
    missing = [k for k in range(nparts) if not os.path.exists(os.path.join(out, f"part{k}.done"))]
    if missing:
        raise SystemExit(f"feature parts not finished: {missing} (see results/logs/features_*.log)")
    for t, x in KEYS:
        parts = [np.load(os.path.join(out, f"part{k}_{t}_{x}.npy"), mmap_mode="r") for k in range(nparts)]
        n = sum(len(p) for p in parts)
        dst = np.lib.format.open_memmap(os.path.join(ds, f"vj_{t}_{x}.npy"), mode="w+",
                                        dtype=np.float16, shape=(n,) + parts[0].shape[1:])
        o = 0
        for p in parts:
            dst[o:o + len(p)] = p
            o += len(p)
        dst.flush()
        log(f"vj_{t}_{x}.npy {dst.shape}")
    write_json(os.path.join(ds, "vjepa_features.json"),
               dict(model=os.environ.get("OW_VJEPA_ID", "facebook/vjepa2-vitl-fpc64-256"),
                    rows=int(n), nparts=nparts, t=time.time()))
    import shutil
    shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.merge:
        merge(args.nparts)
    else:
        run_part(args.part, args.nparts, args.device)


if __name__ == "__main__":
    main()
