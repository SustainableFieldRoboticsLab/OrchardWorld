from __future__ import annotations

import argparse
import math
import os

import numpy as np

from owsim.common import RESULTS, load_config, logger, write_json, device_name
from revision._common import (EVAL_SEEDS, rev_dir, nominal_spec, run_jobs, launch_workers, todo,
                              load_jobs, per_seed_metrics, fmt_ci, fmt_paired, common_args, rng_for)

EXP = "e1_noise"
PLANNER, DAP = "wm_full_cal", "detect_and_pick"

CELLS = {
    "clean":    (0.0, 0.0, 0.0, (PLANNER, DAP)),
    "servo05":  (0.0, 0.0, 0.005, (PLANNER, DAP)),
    "servo10":  (0.0, 0.0, 0.010, (PLANNER, DAP)),
    "servo20":  (0.0, 0.0, 0.020, (PLANNER, DAP)),
    "in05":     (0.005, 0.0, 0.0, (PLANNER,)),
    "in10":     (0.010, 0.0, 0.0, (PLANNER,)),
    "in20":     (0.020, 0.0, 0.0, (PLANNER,)),
    "drop10":   (0.0, 0.10, 0.0, (PLANNER,)),
    "drop20":   (0.0, 0.20, 0.0, (PLANNER,)),
    "combo":    (0.010, 0.10, 0.010, (PLANNER,)),
}


def make_picker_class():
    from owsim.decision_picker import DecisionPicker
    from owsim.simenv import CROP, DEPTH_MAX, downsample2, crop_at

    class NoisyDecisionPicker(DecisionPicker):
        def __init__(self, batch, env, policy, sink, ik, rng, max_decisions=4,
                     sigma_in=0.0, p_drop=0.0, sigma_servo=0.0, noise_rng=None):
            super().__init__(batch, env, policy, sink, ik, rng, max_decisions=max_decisions)
            self._rv_sig_in = float(sigma_in)
            self._rv_p_drop = float(p_drop)
            self._rv_sig_servo = float(sigma_servo)
            self._rv_rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
            self._rv_bias = np.zeros(3)
            self.rv_counts = dict(obs=0, dropped=0, grasp_attempts=0)


        def _capture(self):
            if self._rv_sig_in <= 0.0 and self._rv_p_drop <= 0.0:
                return super()._capture()
            rng = self._rv_rng
            cam = self.B.cam
            e = self.env
            rgb = cam.color_np[e]
            depth = cam.depth_np[e].astype(np.float32)
            depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, DEPTH_MAX)
            depth = np.minimum(depth, DEPTH_MAX)
            depth_true = depth
            if self._rv_sig_in > 0.0:
                valid = depth < DEPTH_MAX
                noisy = np.clip(depth + rng.normal(0.0, self._rv_sig_in, depth.shape), 0.05, DEPTH_MAX)
                depth = np.where(valid, noisy, DEPTH_MAX).astype(np.float32)
            pos_c, R = cam.cams[e]
            H, W = depth.shape
            f = cam.percept.focal_px
            tp = self.tgt["pos"].copy()
            dets = cam.dets_env[e] or []
            det = None
            if dets:
                dd = [np.linalg.norm(x.center_world - tp) for x in dets]
                k = int(np.argmin(dd))
                if dd[k] < 0.06:
                    det = dets[k]
            if det is not None:
                tp = 0.5 * (tp + det.center_world)
                self.tgt["pos"] = tp.copy()
            self.rv_counts["obs"] += 1
            det_f, n_f = det, len(dets)
            if det is not None and self._rv_p_drop > 0.0 and rng.random() < self._rv_p_drop:
                det_f, n_f = None, max(len(dets) - 1, 0)
                self.rv_counts["dropped"] += 1
            tp_f = tp + (rng.normal(0.0, self._rv_sig_in, 3) if self._rv_sig_in > 0.0 else 0.0)
            pc = R.T @ (tp_f - pos_c)
            zc = -pc[2]
            if zc > 0.05:
                u = 0.5 * (W - 1) + f * pc[0] / zc
                v = 0.5 * (H - 1) - f * pc[1] / zc
            else:
                u, v = 0.5 * (W - 1), 0.5 * (H - 1)
            g_rgb = downsample2(rgb).round().astype(np.uint8)
            g_dep = downsample2(depth).astype(np.float16)
            c_rgb = crop_at(rgb, v, u, CROP, 0)
            c_dep = crop_at(depth, v, u, CROP, DEPTH_MAX).astype(np.float16)
            state = self._state_vector(tp_f, pc, u / W, v / H, det_f, n_f)
            priv = self._privileged(pos_c, R, depth_true, f)
            return dict(g_rgb=g_rgb, g_dep=g_dep, c_rgb=c_rgb, c_dep=c_dep, state=state,
                        det=det is not None, uv=(u / W, v / H), priv=priv,
                        pose_ok=self._obs_pose_ok, frame=self.B.frame)


        def _st_grasp(self):
            if self._rv_sig_servo <= 0.0:
                return super()._st_grasp()
            if self._t_state == 1:
                self._rv_bias = self._rv_rng.normal(0.0, self._rv_sig_servo, 3)
                self.rv_counts["grasp_attempts"] += 1
            ap = self._bq()[self._apple_idx, :3]
            d = np.linalg.norm(ap - self._target, axis=1)
            j = int(d.argmin())
            if d[j] > 0.15:
                self._fail("no fruit at detection")
                return
            live = ap[j]
            aim = live + self._rv_bias
            tcp = self._tcp_world()
            if self._t_state == 1:
                self._servo_off = np.zeros(3)
            if self._t_state % 8 == 1:
                self._servo_off = np.clip(self._servo_off + 0.35 * (aim - tcp), -0.09, 0.09)
                q, err = self._ik_to(aim + self._servo_off,
                                     approach_world=self._appr3(self._appr, aim))
                if err > 0.10:
                    self._fail("grasp unreachable")
                    return
                self._set_arm(q)
            tcp_d = np.linalg.norm(tcp - live)
            centred = tcp_d < 0.022 or (tcp_d < self.GRASP_R and self._stall > 40)
            gj = int(self._apple_glob[j])
            if centred and d[j] < 0.10 and not self.apples.detached[gj]:
                self._target_apple = gj
                self.apples.hold(gj, self.wrist)
                self._fingers(self.FINGER_CLOSED)
                self._retract_from = self._target.copy()
                self._retract_dist = 0.0
                self._goto("PULL")
            elif self._stalled(tcp_d) or self._t_state > 240:
                self._fail("grasp stalled")

    return NoisyDecisionPicker


def check_copies():
    import hashlib
    from owsim.common import ROOT
    h = hashlib.md5(open(os.path.join(ROOT, "owsim", "decision_picker.py"), "rb").read()).hexdigest()
    return h == "20eed1bd5f02cf3231ac25741ab226b1"


def build_jobs():
    jobs = []
    for seed in EVAL_SEEDS:
        for cell, (si, pd, ss, ctrls) in CELLS.items():
            for c in ctrls:
                jobs.append(dict(id=f"e1__{cell}__{c}__{seed}", cell=cell, controller=c, seed=seed,
                                 sigma_in=si, p_drop=pd, sigma_servo=ss, foliage=0.8, apples=35))
    return jobs


def _runner(cfg, frames=None, num_envs=None):
    from owsim.policies import make_policy
    from owsim.rollout import run_batch
    Picker = make_picker_class()
    cache = {}
    log = logger("e1")

    def run_one(jb):
        c = jb["controller"]
        if c not in cache:
            cache[c] = make_policy(c, runs_root=os.path.join(RESULTS, "runs"),
                                   device="cuda" if device_name() == "cuda" else "cpu")
        pol = cache[c]
        if hasattr(pol, "latency_ms"):
            pol.latency_ms = []
        spec = nominal_spec(cfg, jb["seed"], jb["foliage"], jb["apples"], frames=frames, num_envs=num_envs)
        made = []

        def factory(B, e, sink, ik, rng):
            p = Picker(B, e, pol, sink, ik, rng, max_decisions=spec["max_decisions"],
                       sigma_in=jb["sigma_in"], p_drop=jb["p_drop"], sigma_servo=jb["sigma_servo"],
                       noise_rng=rng_for("e1", jb["cell"], jb["controller"], jb["seed"], e))
            made.append(p)
            return p

        records, summary = run_batch(spec, pol, log=log, deadline=None, keep_obs=0, picker_factory=factory)
        summary["noise_counts"] = {k: int(sum(p.rv_counts[k] for p in made)) for k in made[0].rv_counts} \
            if made else {}
        summary["theta_used"] = getattr(pol, "theta", None)
        return records, summary

    return run_one


def worker(args):
    if not check_copies():
        raise SystemExit("owsim/decision_picker.py differs from the audited version - see README")
    cfg = load_config(args.config)
    run_jobs(EXP, build_jobs(), _runner(cfg), args.worker_id, args.run_token)


def launch(args):
    launch_workers("revision.e1_noise", EXP, len(todo(EXP, build_jobs())), args.config)
    report(args)


def smoke(args):
    from owsim.eval_control import job_metrics
    log = logger("e1 smoke")
    cfg = load_config(args.config)
    if not check_copies():
        raise SystemExit("owsim/decision_picker.py differs from the audited version (md5); the copied "
                         "_capture/_st_grasp may be out of date - do not launch, send that file")
    run_one = _runner(cfg, frames=3600, num_envs=4)
    out = {}
    for cell in ("combo", "servo20"):
        si, pd, ss, _ = CELLS[cell]
        jb = dict(id=f"smoke__{cell}", cell=cell, controller=PLANNER, seed=EVAL_SEEDS[0],
                  sigma_in=si, p_drop=pd, sigma_servo=ss, foliage=0.8, apples=35)
        records, summary = run_one(jb)
        m = job_metrics(records, summary)
        out[cell] = dict(status=summary["status"], decisions=m["decisions"], picks=m["pick_attempts"],
                         harvested=m["harvested"], noise_counts=summary.get("noise_counts"))
        log(f"{cell}: {out[cell]}")
    write_json(os.path.join(rev_dir(EXP), "smoke.json"), out)
    ok = all(v["status"] in ("ok", "all_done") and v["decisions"] > 0 for v in out.values())
    ok = ok and out["combo"]["noise_counts"].get("obs", 0) > 0
    log("SMOKE " + ("PASSED" if ok else "FAILED - send results/revision/logs and smoke.json"))
    if not ok:
        raise SystemExit(1)


def report(args):
    from owsim.seeds16 import boot_ci, paired, per_seed
    log = logger("e1 report")
    out = rev_dir(EXP)
    rng = np.random.default_rng(0)
    J = load_jobs(EXP)
    Q = ["harvest_success", "contact_rate", "throughput", "safe_return"]
    D, A = {}, {}
    for cell, (_, _, _, ctrls) in CELLS.items():
        for c in ctrls:
            D[(cell, c)] = per_seed_metrics(J, lambda jb, cell=cell, c=c: jb["cell"] == cell and jb["controller"] == c)
            A[(cell, c)] = {q: boot_ci([v[q] for v in D[(cell, c)].values()], rng) for q in Q}
            A[(cell, c)]["n"] = len(D[(cell, c)])
            A[(cell, c)]["truncated"] = sum(v["_status"] not in ("ok", "all_done", "deadline")
                                            for v in D[(cell, c)].values())
    L = ["# E1 - localization-noise robustness (16 evaluation seeds, nominal condition)", "",
         "(A) input noise: planner inputs only; (B) servo: per-attempt aim offset of the grasp servo.",
         "Mean [95 % bootstrap CI over seeds].", "",
         "| cell | controller | n | trunc. | harvest/pick % | contact % | fruit/robot/min | safe return |",
         "|---|---|---|---|---|---|---|---|"]
    for (cell, c), a in A.items():
        if a["n"]:
            L.append(f"| {cell} | {c} | {a['n']} | {a['truncated']} | {fmt_ci(a['harvest_success'])} | "
                     f"{fmt_ci(a['contact_rate'])} | {fmt_ci(a['throughput'], 3)} | {fmt_ci(a['safe_return'], 3)} |")
    L += ["", "## Paired differences vs the clean cell of the same controller", ""]
    tests = {}
    for (cell, c) in A:
        if cell == "clean" or not A[(cell, c)]["n"]:
            continue
        L.append(f"- {cell} / {c}:")
        for q, d in (("harvest_success", 1), ("contact_rate", 1), ("throughput", 3), ("safe_return", 3)):
            r = paired(D[(cell, c)], D[("clean", c)], q, rng)
            tests[f"{cell}|{c}|{q}"] = r
            L.append(f"  - {q}: {fmt_paired(r, d)}")

    L += ["", "## Sanity check: clean cell vs the existing 16-seed results (expected: no difference)", ""]
    for c, key in ((PLANNER, "wm_cal"), (DAP, "detect_and_pick")):
        ref = per_seed(key)
        for q, d in (("harvest_success", 1), ("safe_return", 3)):
            r = paired(D[("clean", c)], ref, q, rng)
            tests[f"sanity|{c}|{q}"] = r
            L.append(f"- {c} {q}: {fmt_paired(r, d)}")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    write_json(os.path.join(out, "stats.json"), dict(agg={f"{k[0]}|{k[1]}": v for k, v in A.items()}, tests=tests))
    log(f"wrote {out}/summary.md, stats.json")


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("mode", choices=["list", "smoke", "launch", "worker", "report"])
    args = ap.parse_args()
    if args.mode == "list":
        J = build_jobs()
        print(len(J), "jobs;", len(todo(EXP, J)), "still to do")
        return
    {"smoke": smoke, "launch": launch, "worker": worker, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
