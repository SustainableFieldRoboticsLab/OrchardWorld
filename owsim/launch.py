from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .common import ROOT, RESULTS, load_config, rdir, logger, write_json, read_json


def gpus():
    env = os.environ.get("OW_GPUS")
    if env:
        return [g for g in env.split(",") if g.strip() != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        ids = [l.strip() for l in out.splitlines() if l.strip()]
        return ids or ["0"]
    except Exception:
        return ["cpu"]


def spawn(cmd, gpu, log_path):
    env = dict(os.environ)
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["OW_DEVICE"] = "cpu"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, "2")
    env["PYTHONUNBUFFERED"] = "1"
    f = open(log_path, "a")
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)} (gpu {gpu})\n")
    f.flush()
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def wait_all(procs, log, poll=30):
    while True:
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            break
        time.sleep(poll)
    bad = [p.args for p in procs if p.returncode != 0]
    if bad:
        log(f"{len(bad)} worker(s) exited with an error (see results/logs); continuing")
    return len(bad)


def run_collect(cfg_name, cfg, log):
    cc = cfg["collect"]
    G = gpus()
    n = len(G) * int(cc["workers_per_gpu"])
    st_path = os.path.join(rdir("collect"), "deadline.json")
    st = read_json(st_path)
    if st and st.get("config") == cfg_name and time.time() < st["deadline"]:
        deadline = st["deadline"]
        log(f"resuming collection until {time.ctime(deadline)}")
    elif st and st.get("config") == cfg_name:
        log("collection deadline already passed; nothing to do "
            "(delete results/collect/deadline.json to collect more)")
        return 0
    else:
        deadline = time.time() + 3600.0 * float(cc["hours"])
        write_json(st_path, dict(config=cfg_name, deadline=deadline, started=time.time(),
                                 workers=n, gpus=G))
    logs = rdir("logs")
    procs = []
    for i in range(n):
        g = G[i % len(G)]
        cmd = [sys.executable, "-m", "owsim.collect", "--config", cfg_name,
               "--worker-id", str(i), "--n-workers", str(n), "--deadline", str(deadline)]
        procs.append(spawn(cmd, g, os.path.join(logs, f"collect_w{i:02d}.log")))
        time.sleep(2.0)
    log(f"started {n} collection workers on GPUs {G}; deadline {time.ctime(deadline)}")
    return wait_all(procs, log)


def run_train(cfg_name, cfg, log, phase=1):
    tc = dict(cfg["train"])
    if phase == 2:
        tc.update(cfg.get("phase2", {}).get("train", {}))
    G = gpus()
    per = int(tc.get("runs_per_gpu", 1))
    queue = [(v, s) for s in tc["seeds"] for v in tc["variants"]]
    queue = [(v, s) for v, s in queue
             if not os.path.exists(os.path.join(RESULTS, "runs", f"{v}_s{s}", "meta.json"))]
    log(f"{len(queue)} training runs to do on GPUs {G} ({per} per GPU)")
    logs = rdir("logs")
    slots = {f"{g}:{k}": None for g in G for k in range(per)}
    nbad = 0
    while queue or any(p is not None for p in slots.values()):
        for sk, p in list(slots.items()):
            if p is not None and p.poll() is not None:
                nbad += int(p.returncode != 0)
                slots[sk] = None
            if slots[sk] is None and queue:
                v, s = queue.pop(0)
                g = sk.split(":")[0]
                dev = "cpu" if g == "cpu" else "cuda"
                cmd = [sys.executable, "-m", "owsim.train", "--config", cfg_name,
                       "--variant", v, "--seed", str(s), "--device", dev]
                slots[sk] = spawn(cmd, g, os.path.join(logs, f"train_{v}_s{s}.log"))
                log(f"started {v} seed {s} on GPU {g}")
        time.sleep(10)
    if nbad:
        log(f"{nbad} training run(s) failed (see results/logs/train_*.log)")
    return nbad


def run_eval(cfg_name, cfg, log, phase=1):
    ev = dict(cfg["eval"])
    if phase == 2:
        ev.update(cfg.get("phase2", {}).get("eval", {}))
    G = gpus()
    n = len(G) * int(ev["workers_per_gpu"])
    deadline = time.time() + 3600.0 * float(ev["hours"])
    token = time.strftime("%Y%m%d_%H%M%S")
    logs = rdir("logs")
    procs = []
    for i in range(n):
        g = G[i % len(G)]
        cmd = [sys.executable, "-m", "owsim.eval_control", "--config", cfg_name,
               "--worker-id", str(i), "--run-token", token, "--deadline", str(deadline),
               "--phase", str(phase)]
        procs.append(spawn(cmd, g, os.path.join(logs, f"eval_w{i:02d}.log")))
        time.sleep(2.0)
    write_json(os.path.join(rdir("control"), "run.json"),
               dict(token=token, deadline=deadline, workers=n, gpus=G, started=time.time(),
                    phase=phase))
    log(f"started {n} evaluation workers on GPUs {G}; deadline {time.ctime(deadline)}")
    return wait_all(procs, log)


def run_features(cfg_name, cfg, log, phase=2):
    G = gpus()
    logs = rdir("logs")
    procs = []
    for k, g in enumerate(G):
        dev = "cpu" if g == "cpu" else "cuda"
        cmd = [sys.executable, "-m", "owsim.features", "--part", str(k),
               "--nparts", str(len(G)), "--device", dev]
        procs.append(spawn(cmd, g, os.path.join(logs, f"features_p{k}.log")))
    log(f"V-JEPA 2 feature extraction on GPUs {G}")
    wait_all(procs, log, poll=10)
    r = subprocess.run([sys.executable, "-m", "owsim.features", "--merge", "--nparts", str(len(G))],
                       cwd=ROOT)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "train", "eval", "features"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = logger(f"launch {args.stage}")
    if args.stage == "features":
        sys.exit(run_features(args.config, cfg, log))
    if args.stage == "collect":
        run_collect(args.config, cfg, log)
    elif args.stage == "train":
        run_train(args.config, cfg, log, args.phase)
    else:
        run_eval(args.config, cfg, log, args.phase)
    if args.stage == "train":
        main_run = "full_s0" if args.phase == 1 else "vjepa_s0"
        ok = os.path.exists(os.path.join(RESULTS, "runs", main_run, "meta.json"))
        sys.exit(0 if ok else 1)
    sh = os.path.join(RESULTS, "collect", "shards")
    if args.stage == "collect" and not (os.path.isdir(sh) and os.listdir(sh)):
        log("no data was collected - see results/logs/collect_w*.log")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
