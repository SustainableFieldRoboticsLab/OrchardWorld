from __future__ import annotations

import glob
import os
import subprocess
import time

from .common import RESULTS, read_json, load_config

STAGES = ["check", "collect", "dataset", "train", "offline", "control"]
STAGES2 = ["p2_model", "p2_features", "p2_train", "p2_offline", "p2_control"]


def hm(sec):
    sec = max(int(sec), 0)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def main():
    now = time.time()
    print(f"=== OrchardWorld-Sim status  {time.strftime('%Y-%m-%d %H:%M:%S')}  results: {RESULTS}")
    sdir = os.path.join(RESULTS, "stages")
    cur = ""
    try:
        cur = open(os.path.join(sdir, "current.txt")).read().strip()
    except Exception:
        pass
    marks = []
    for s in STAGES:
        if os.path.exists(os.path.join(sdir, f"{s}.done")):
            marks.append(f"{s}:done")
        elif cur.startswith(s):
            marks.append(f"[{s}: RUNNING]")
        else:
            marks.append(f"{s}:-")
    print("pipeline  " + "  ".join(marks))
    if any(os.path.exists(os.path.join(sdir, f"{s}.done")) for s in STAGES2) or cur.startswith("p2_"):
        m2 = []
        for s in STAGES2:
            if os.path.exists(os.path.join(sdir, f"{s}.done")):
                m2.append(f"{s}:done")
            elif cur.startswith(s):
                m2.append(f"[{s}: RUNNING]")
            else:
                m2.append(f"{s}:-")
        print("phase 2   " + "  ".join(m2))
    start = read_json(os.path.join(sdir, "pipeline.json"), {})
    if start.get("started"):
        print(f"          started {time.ctime(start['started'])}, elapsed {hm(now - start['started'])}")

    dl = read_json(os.path.join(RESULTS, "collect", "deadline.json"))
    hbs = [read_json(f, {}) for f in sorted(glob.glob(os.path.join(RESULTS, "collect", "heartbeat", "*.json")))]
    if dl or hbs:
        dec = sum(h.get("decisions", 0) for h in hbs)
        bat = sum(h.get("batches", 0) for h in hbs)
        err = sum(h.get("errors", 0) for h in hbs)
        live = [h for h in hbs if now - h.get("t", 0) < 600 and h.get("state") != "finished"]
        fin = sum(1 for h in hbs if h.get("state") == "finished")
        cur_dec = sum(h.get("current", {}).get("decisions", 0) for h in live)
        rem = (dl["deadline"] - now) if dl else 0
        el = (now - dl["started"]) if dl else 0
        rate = (dec + cur_dec) / max(el / 3600.0, 1e-6)
        print(f"\nCOLLECT   {dec + cur_dec} decisions ({dec} saved in {bat} batches), "
              f"{len(live)} workers active, {fin} finished, errors {err}")
        if dl:
            print(f"          elapsed {hm(el)}, remaining {hm(rem)}, ~{rate:.0f} decisions/h, "
                  f"projected total ~{int(rate * (el + max(rem, 0)) / 3600.0)}")
        fps = [h.get("current", {}).get("fps") for h in live if h.get("current", {}).get("fps")]
        if fps:
            print(f"          batch fps per worker: median {sorted(fps)[len(fps) // 2]:.1f} "
                  f"(min {min(fps):.1f}, max {max(fps):.1f})")
        stale = [h.get("worker") for h in hbs if now - h.get("t", 0) > 900 and h.get("state") != "finished"]
        if stale and rem > 0:
            print(f"          WARNING: no heartbeat for >15 min from workers {stale} "
                  f"(check results/logs/collect_w*.log)")
    ds = read_json(os.path.join(RESULTS, "dataset", "stats.json"))
    if ds:
        print(f"\nDATASET   {ds['n_decisions']} decisions, {ds['n_trees']} trees, splits {ds['per_split']}")

    runs = sorted(glob.glob(os.path.join(RESULTS, "runs", "*_s*")))
    if runs:
        print("\nTRAIN")
        for r in runs:
            h = read_json(os.path.join(r, "history.json"), [])
            m = read_json(os.path.join(r, "meta.json"))
            name = os.path.basename(r)
            if m:
                print(f"   {name:16s} done   best epoch {m['best_epoch']}/{m['epochs']}  "
                      f"{m['train_seconds'] / 60:.0f} min  {m['params'] / 1e6:.1f}M params")
            elif h:
                last = h[-1]
                print(f"   {name:16s} epoch {last['epoch']}  val {last['val_score']:.3f}  "
                      f"aucDetach {last.get('auc_detach')}  ({last['t'] / 60:.0f} min)")
            else:
                print(f"   {name:16s} starting")
    off = read_json(os.path.join(RESULTS, "offline", "offline_metrics.json"))
    if off:
        print(f"\nOFFLINE   metrics for {sum(len(v['runs']) for v in off.values())} predictors written")

    cj = glob.glob(os.path.join(RESULTS, "control", "jobs", "*.json"))
    run = read_json(os.path.join(RESULTS, "control", "run.json"))
    if cj or run:
        total = "?"
        try:
            from .eval_control import build_jobs
            cfgp = read_json(os.path.join(sdir, "pipeline.json"), {}).get("config", "full")
            cfg_ = load_config(cfgp)
            total = len(build_jobs(cfg_, 1))
            if (run or {}).get("phase") == 2 or os.path.exists(os.path.join(sdir, "p2_control.done")):
                total += len(build_jobs(cfg_, 2))
        except Exception:
            pass
        rem = (run["deadline"] - now) if run else 0
        print(f"\nCONTROL   {len(cj)}/{total} evaluation jobs done"
              + (f", remaining budget {hm(rem)}" if run else ""))
        for f in sorted(glob.glob(os.path.join(RESULTS, "control", "heartbeat", "*.json"))):
            h = read_json(f, {})
            if now - h.get("t", 0) < 600 and h.get("state") != "finished":
                print(f"   w{h.get('worker')}: {h.get('job', '')}  frame {h.get('frame', 0)}/{h.get('frames', 0)}")

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if out:
            print("\nGPU       " + "   ".join(
                f"#{a.strip()}: {b.strip()}% {int(c) / 1024:.0f}/{int(d) / 1024:.0f}GB"
                for a, b, c, d in (l.split(",") for l in out.splitlines())))
    except Exception:
        pass
    log = os.path.join(RESULTS, "logs", "pipeline.log")
    if os.path.exists(log):
        lines = open(log, errors="replace").read().splitlines()[-6:]
        print("\nlast pipeline log lines:")
        for l in lines:
            print("   " + l[:160])


if __name__ == "__main__":
    main()
