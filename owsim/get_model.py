from __future__ import annotations

import os
import sys
import time

from .vjepa import VJEPA_ID


def main():
    from huggingface_hub import scan_cache_dir, snapshot_download
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default)')}")
    have = False
    try:
        have = any(r.repo_id == VJEPA_ID for r in scan_cache_dir().repos)
    except Exception:
        pass
    if have:
        print(f"{VJEPA_ID}: already in the shared cache")
    else:
        print(f"{VJEPA_ID}: downloading once into the shared cache (~1.3 GB)")
        snapshot_download(VJEPA_ID)
    import numpy as np
    import torch
    from .vjepa import VJEPAEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VJEPAEncoder(device=dev)
    x = np.random.randint(0, 255, (8, 72, 96, 3), dtype=np.uint8)
    c = np.random.randint(0, 255, (8, 64, 64, 3), dtype=np.uint8)
    t0 = time.time()
    gv, cg = enc.encode(x, c)
    if dev == "cuda":
        torch.cuda.synchronize()
    ok = gv.shape == (8, 1024) and cg.shape == (8, 16, 1024) and torch.isfinite(gv.float()).all()
    print(f"encoder test on {dev}: global {tuple(gv.shape)} crop {tuple(cg.shape)} "
          f"({1e3 * (time.time() - t0):.0f} ms for 8 obs)  ->  {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
