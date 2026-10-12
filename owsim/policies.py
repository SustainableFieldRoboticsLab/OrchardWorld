from __future__ import annotations

import time

import numpy as np

from .decision_picker import N_ACTIONS, PICKS, REPOS, SKIP


class RandomPolicy:
    name = "random"

    def __init__(self, rng, probs=(0.2, 0.2, 0.2, 0.15, 0.15, 0.1)):
        self.rng = rng
        self.p = np.asarray(probs, dtype=float) / np.sum(probs)

    def decide(self, picker, obs):
        a = int(self.rng.choice(N_ACTIONS, p=self.p))
        return a, dict(p=float(self.p[a]))


class DetectAndPick:
    name = "detect_and_pick"

    def decide(self, picker, obs):
        return (0 if picker.tgt["n_pick"] == 0 else SKIP), dict(p=1.0)


class ReactiveRetry:
    name = "reactive_retry"

    def __init__(self, max_picks=3):
        self.max_picks = int(max_picks)

    def decide(self, picker, obs):
        return (0 if picker.tgt["n_pick"] < self.max_picks else SKIP), dict(p=1.0)


class WorldModelPolicy:

    def __init__(self, run_dir, device="cuda", allow_skip=True, name=None, calibrated=False):
        from .model import load_run
        self.model, self.meta = load_run(run_dir, device, calibrated=calibrated)
        self.device = device
        self.allow_skip = bool(allow_skip)
        self.theta = float(self.meta.get("policy", {}).get("theta", 0.0))
        self.name = name or ("wm_" + self.meta["variant"])
        self.latency_ms = []
        self.vj = None
        if self.model.use_vjepa:
            from .vjepa import VJEPAEncoder
            self.vj = VJEPAEncoder(device=device)

    def decide(self, picker, obs):
        from .model import score_observation
        t0 = time.perf_counter()
        scores = score_observation(self.model, self.meta, obs, self.device, self.vj)
        self.latency_ms.append(1e3 * (time.perf_counter() - t0))
        cand = list(PICKS) + list(REPOS)
        best = max(cand, key=lambda a: scores[a])
        if self.allow_skip and scores[best] < self.theta:
            best = SKIP
        sc = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        sc[:len(scores)] = scores
        return best, dict(p=1.0, scores=sc)


class ReactiveSelect:
    name = "reactive_select"

    def __init__(self, params_path, theta=None):
        import json
        P = json.load(open(params_path))
        self.cols = P["cols"]
        self.mu = np.asarray(P["mu"])
        self.sd = np.asarray(P["sd"])
        self.coef = {k: (np.asarray(v["w"]), float(v["b"])) for k, v in P["heads"].items()}
        self.tau = float(P["tau_pick_center"])
        self.theta = float(P["theta"] if theta is None else theta)
        self.latency_ms = []

    def score(self, state):
        from .decision_picker import R_HARVEST, R_CONTACT, R_DAMAGE, R_TIME
        x = (np.asarray(state, float)[self.cols] - self.mu) / self.sd
        p = {k: 1.0 / (1.0 + np.exp(-(x @ w + b))) for k, (w, b) in self.coef.items()}
        return (R_HARVEST * p["success"] - R_CONTACT * p["contact"] - R_DAMAGE * p["damage"]
                - R_TIME * self.tau)

    def decide(self, picker, obs):
        t0 = time.perf_counter()
        sc = float(self.score(obs["state"]))
        self.latency_ms.append(1e3 * (time.perf_counter() - t0))
        scores = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        scores[0] = sc
        return (0 if sc >= self.theta else SKIP), dict(p=1.0, scores=scores)


def make_policy(name, rng=None, runs_root=None, device="cuda"):
    if name == "random":
        return RandomPolicy(rng)
    if name == "detect_and_pick":
        return DetectAndPick()
    if name == "reactive_retry":
        return ReactiveRetry()
    if name.startswith("wm_"):
        import os
        core = name[3:]
        cal = core.endswith("_cal")
        core = core[:-4] if cal else core
        noskip = core.endswith("_noskip")
        variant = core[:-7] if noskip else core
        run_dir = os.path.join(runs_root, f"{variant}_s0")
        return WorldModelPolicy(run_dir, device=device, allow_skip=not noskip, name=name,
                                calibrated=cal)
    if name.startswith("reactive_select"):
        import os
        return ReactiveSelect(os.path.join(os.path.dirname(runs_root), "extras", "reactive_select.json"))
    raise ValueError(name)
