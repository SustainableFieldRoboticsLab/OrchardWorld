from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("OW_RESULTS", os.path.join(ROOT, "results"))


def load_config(name_or_path: str) -> dict:
    import yaml
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(ROOT, "configs", f"{name_or_path}.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = os.path.abspath(p)
    return cfg


def split_of(seed: int) -> str:
    r = int(seed) % 10
    return "test" if r == 0 else ("val" if r == 1 else "train")


def rdir(*parts) -> str:
    p = os.path.join(RESULTS, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def logger(tag: str):
    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)
    return log


def write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def device_name() -> str:
    return os.environ.get("OW_DEVICE", "cuda" if sys.platform.startswith("linux") else "cpu")
