from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from .common import RESULTS, rdir, write_json

OK, BAD, WARN = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rep = {}
    fails = 0

    def line(tag, status, msg):
        print(f"[{status}] {tag:14s} {msg}", flush=True)

    line("python", OK, sys.version.split()[0])
    try:
        import torch
        cu = torch.cuda.is_available()
        rep["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda, gpus=torch.cuda.device_count())
        line("torch", OK if cu else BAD,
             f"{torch.__version__} cuda={torch.version.cuda} available={cu} gpus={torch.cuda.device_count()}")
        fails += 0 if cu else 1
        if cu:
            x = torch.randn(512, 512, device="cuda")
            float((x @ x).sum())
    except Exception as ex:
        line("torch", BAD, repr(ex))
        fails += 1
    try:
        import warp as wp
        try:
            wp.config.log_level = wp.LOG_WARNING
        except Exception:
            pass
        wp.init()
        n = wp.get_cuda_device_count()
        ver = wp.__version__
        try:
            from warp._src.context import runtime as rt
        except Exception:
            rt = getattr(getattr(wp, "context", None), "runtime", None)
        tk = getattr(rt, "toolkit_version", None)
        drv = getattr(rt, "driver_version", None)
        rep["warp"] = dict(version=ver, cuda_devices=n, toolkit=str(tk), driver=str(drv))
        good = n > 0 and (tk is None or tuple(tk)[0] <= 12)
        line("warp", OK if good else BAD,
             f"{ver} cuda_devices={n} toolkit={tk} driver={drv}"
             + ("" if good else "  <- need the +cu12 Warp wheel (see README, Troubleshooting)"))
        fails += 0 if good else 1
    except Exception as ex:
        line("warp", BAD, repr(ex))
        fails += 1
    try:
        import newton
        import mujoco_warp
        rep["newton"] = newton.__version__
        line("newton", OK, f"{newton.__version__} (mujoco_warp {getattr(mujoco_warp, '__version__', '?')})")
    except Exception as ex:
        line("newton", BAD, repr(ex))
        fails += 1
    du = shutil.disk_usage(os.path.dirname(RESULTS) if not os.path.exists(RESULTS) else RESULTS)
    free_gb = du.free / 1e9
    rep["disk_free_gb"] = free_gb
    line("disk", OK if free_gb > 30 else WARN,
         f"{free_gb:.0f} GB free at {RESULTS} (the full run needs ~10-20 GB)")
    if args.quick or fails:
        print("\nRESULT:", "all good" if not fails else f"{fails} problem(s) - fix before running",
              flush=True)
        sys.exit(1 if fails else 0)

    import numpy as np
    from .rollout import run_batch
    from .policies import RandomPolicy
    from .simenv import _FRANKA
    line("robot asset", OK if os.path.exists(_FRANKA) else WARN, _FRANKA)
    spec = dict(seed=123457, num_envs=2, foliage=0.6, apples=35, light_dir=(-0.4, 0.3, -1.0),
                shadows=False, frames=600, device="cuda", max_decisions=4)
    t0 = time.time()
    try:
        recs, summ = run_batch(spec, RandomPolicy(np.random.default_rng(0)), log=print)
        rep["sim"] = dict(status=summ["status"], fps=summ["fps"], build_s=summ["build_s"],
                          decisions=len(recs), wall=time.time() - t0)
        good = summ["status"] in ("ok", "all_done")
        line("simulation", OK if good else BAD,
             f"status={summ['status']} {summ['fps']:.1f} batch-fps, build {summ['build_s']:.0f}s, "
             f"{len(recs)} decisions in {summ['sim_s']:.0f}s sim (first run compiles kernels: slow)")
        fails += 0 if good else 1
    except Exception:
        import traceback
        traceback.print_exc()
        line("simulation", BAD, "exception (see above)")
        fails += 1
    write_json(os.path.join(rdir(), "check.json"), rep)
    print("\nRESULT:", "all good - ready to run" if not fails else f"{fails} problem(s)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
