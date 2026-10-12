from __future__ import annotations

import math

import numpy as np

from .simenv import CROP, DEPTH_MAX, downsample2, crop_at

import treesim.picker as _picker
from treesim import robot as _robot

ACTIONS = ["pick_center", "pick_left", "pick_right", "repos_left", "repos_right", "skip"]
N_ACTIONS = len(ACTIONS)
PICKS = (0, 1, 2)
REPOS = (3, 4)
SKIP = 5
PICK_ANGLE = math.radians(30.0)
REPOS_ANGLE = math.radians(35.0)
OBS_DIST = 0.30
MAX_DECISIONS = 4
FPS = 60.0
CONTACT_DISP = 0.05

R_HARVEST, R_CONTACT, R_DAMAGE, R_TIME = 1.0, 0.25, 1.0, 0.01

FAIL_CODES = ["", "unreachable", "reach stalled", "no fruit at detection",
              "grasp unreachable", "grasp stalled", "stem too strong / pull timeout",
              "missed bucket", "other"]

STATE_NAMES = (
    [f"arm_q{i}" for i in range(9)] + [f"arm_qd{i}" for i in range(9)]
    + ["base_vx", "base_vy", "base_wz"]
    + ["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
    + ["tcp_x", "tcp_y", "tcp_z"]
    + ["tgt_bx", "tgt_by", "tgt_bz", "tgt_cx", "tgt_cy", "tgt_cz",
       "det_found", "det_radius", "det_log_npix", "det_rms_cm", "tgt_u", "tgt_v",
       "tgt_dist", "tgt_height", "n_dets"]
    + ["n_dec_on_target", "n_pick_on_target"]
    + [f"prev_a{i}" for i in range(N_ACTIONS + 1)]
    + ["prev_contact", "prev_detach"]
)
STATE_DIM = len(STATE_NAMES)
N_CONTEXT = 2 + (N_ACTIONS + 1) + 2
ROBOT_STATE_COLS = list(range(0, 28))


def reward(success, contact, damage, tau):
    return (R_HARVEST * float(success) - R_CONTACT * float(contact)
            - R_DAMAGE * float(damage) - R_TIME * float(tau))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot2(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


class DecisionPicker(_picker.AutoPicker):

    def __init__(self, batch, env, policy, sink, ik, rng, max_decisions=MAX_DECISIONS):
        super().__init__(batch.sim, batch.tm, batch.cam, batch.driver,
                         batch.cfg.robot, metrics=None, env=env, ik=ik)
        self.B = batch
        self.policy = policy
        self.sink = sink
        self.rng = rng
        self.max_decisions = int(max_decisions)
        self.tgt = None
        self.act = None
        self._obs_kind = "pre"
        self._obs_ready = None
        self._obs_pose_ok = True
        self._appr = None
        self._repos_goal = None
        self.n_targets = 0
        self.n_approach_fail = 0
        self.n_phantom = 0
        self.n_decisions = 0
        first_shape = {}
        for s, b in enumerate(batch.shape_body):
            first_shape.setdefault(int(b), int(s))
        ssc = batch.tm.model.shape_scale.numpy()
        self._apple_r = {int(b): float(ssc[first_shape[int(b)]][0]) for b in self._apple_idx}

    def _drive_xy(self, vx, vy, w):
        self._cmd_v = 0.0
        ds = self.drv.planar_dof[self.env]
        th = self.drv._target_host
        th[ds + 0] = vx
        th[ds + 1] = vy
        th[ds + 3] = w
        self.sim.control.joint_target_qd.assign(th)

    def _horiz_unit(self, frm, to):
        v = np.asarray(to[:2], float) - np.asarray(frm[:2], float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _appr3(self, a2, world_pt):
        base, _ = self._chassis_pose()
        v = world_pt - base
        tilt = 0.15 * v[2] / max(np.linalg.norm(v[:2]), 1e-3)
        return np.array([a2[0], a2[1], tilt])

    def update(self):
        if self.done:
            return
        super().update()
        if self.act is not None and self.act.get("end_frame") is None:
            self._account()

    def _st_align(self):
        super()._st_align()
        if self.state == "REACH" and self.tgt is None:
            self._begin_target()

    def _fail(self, reason):
        if self.act is None or self.act.get("end_frame") is not None:
            self.n_approach_fail += 1
            if self.tgt is not None:
                self._end_target()
                return
            return super()._fail(reason)
        if self._target_apple >= 0:
            self.apples.release(self._target_apple)
        self._target_apple = -1
        self.act["fail_reason"] = reason
        self._fingers(self.FINGER_OPEN)
        self._end_action()

    def _begin_target(self):
        ap = self._bq()[self._apple_idx, :3]
        d = np.linalg.norm(ap - self._target, axis=1)
        j = int(d.argmin()) if len(d) else -1
        if j < 0 or d[j] > 0.15 or self.apples.detached[int(self._apple_glob[j])]:
            self.n_phantom += 1
            self._blacklist.append((self._target.copy(), 10 ** 12))
            self._target = None
            self._goto("SCAN")
            return
        self.n_targets += 1
        self.tgt = dict(uid=self.n_targets, pos=self._target.copy(),
                        pos0=self._target.copy(),
                        gt_glob=int(self._apple_glob[j]), gt_body=int(self._apple_idx[j]),
                        n_dec=0, n_pick=0, prev_a=-1, prev_contact=0, prev_detach=0,
                        start_frame=self.B.frame)
        self._obs_kind = "pre"
        self._goto("OBSERVE")

    def _end_target(self):
        if self.tgt is not None:
            self._blacklist.append((self.tgt["pos"].copy(), 10 ** 12))
            self._blacklist.append((self.tgt["pos0"].copy(), 10 ** 12))
        if self._target_apple >= 0:
            self.apples.release(self._target_apple)
        self.tgt = None
        self.act = None
        self._target = None
        self._target_apple = -1
        self._set_arm(self.arm_home)
        self._fingers(self.FINGER_OPEN)
        self._drive(0.0, 0.0)
        self._goto("SCAN")

    def _st_observe(self):
        t = self._t_state
        if t == 1:
            self._drive(0.0, 0.0)
            self._obs_ready = None
            self._obs_pose_ok = True
        base, _ = self._chassis_pose()
        tp = self.tgt["pos"]
        a2 = self._horiz_unit(base, tp)
        goal = tp - np.array([a2[0], a2[1], 0.0]) * OBS_DIST
        if t % 15 == 1 and self._obs_ready is None:
            q, err = self._ik_to(goal, approach_world=self._appr3(a2, tp))
            if err > 0.08:
                self._obs_pose_ok = False
                self._set_arm(self.arm_home)
            else:
                self._set_arm(q)
        self._fingers(self.FINGER_OPEN)
        d = float(np.linalg.norm(self._tcp_world() - goal))
        if self._obs_ready is None and ((d < 0.03 and t > 15) or t > 150
                                        or (not self._obs_pose_ok and t > 45)):
            self._obs_ready = self.B.cam.render_id
        if self._obs_ready is not None and self.B.cam.render_id > self._obs_ready:
            obs = self._capture()
            if self._obs_kind == "post":
                self._finish_action(obs)
            else:
                self._decide(obs)

    def _capture(self):
        cam = self.B.cam
        e = self.env
        rgb = cam.color_np[e]
        depth = cam.depth_np[e].astype(np.float32)
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, DEPTH_MAX)
        depth = np.minimum(depth, DEPTH_MAX)
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
        pc = R.T @ (tp - pos_c)
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
        state = self._state_vector(tp, pc, u / W, v / H, det, len(dets))
        priv = self._privileged(pos_c, R, depth, f)
        return dict(g_rgb=g_rgb, g_dep=g_dep, c_rgb=c_rgb, c_dep=c_dep, state=state,
                    det=det is not None, uv=(u / W, v / H), priv=priv,
                    pose_ok=self._obs_pose_ok, frame=self.B.frame)

    def _state_vector(self, tp, pc, un, vn, det, n_dets):
        jq = self._jq()
        jqd = self.B.sim.state_0.joint_qd.numpy()
        base, bq = self._chassis_pose()
        aq = jq[self.B.arm_q[self.env]]
        aqd = jqd[self.B.arm_dofs[self.env]]
        ds = self.drv.planar_dof[self.env]
        bv = np.array([jqd[ds + 0], jqd[ds + 1], jqd[ds + 3]])
        wq = self._bq()[self.wrist]
        bq_conj = np.array([-bq[0], -bq[1], -bq[2], bq[3]])
        tcp_q = _quat_mul(bq_conj, wq[3:])
        tcp_b = _picker._inv_xform(base, bq, self._tcp_world())
        tgt_b = _picker._inv_xform(base, bq, tp)
        s = np.concatenate([
            aq, aqd, bv, tcp_q, tcp_b, tgt_b, pc,
            [1.0 if det is not None else 0.0,
             det.radius if det is not None else 0.0,
             math.log1p(det.npix) if det is not None else 0.0,
             100.0 * det.rms if det is not None else 0.0,
             un, vn, float(np.linalg.norm(tgt_b[:2])), float(tp[2]), float(n_dets)],
            self._context_vec()])
        return s.astype(np.float32)

    def _context_vec(self):
        tg = self.tgt
        prev = np.zeros(N_ACTIONS + 1)
        prev[tg["prev_a"] if tg["prev_a"] >= 0 else N_ACTIONS] = 1.0
        return np.concatenate([[tg["n_dec"], tg["n_pick"]], prev,
                               [tg["prev_contact"], tg["prev_detach"]]])

    def _privileged(self, pos_c, R, depth, f):
        b = self.tgt["gt_body"]
        p = self._bq()[b, :3]
        r = self._apple_r.get(b, 0.03)
        detached = bool(self.apples.detached[self.tgt["gt_glob"]])
        out = dict(gt_x=float(p[0]), gt_y=float(p[1]), gt_z=float(p[2]),
                   gt_detached=detached, occlusion=1.0, in_fov=0.0,
                   stem_strength=float(self.apples.detach_force[self.tgt["gt_glob"]]))
        pc = R.T @ (p - pos_c)
        zc = -pc[2]
        H, W = depth.shape
        if zc <= 0.08 or detached:
            return out
        u = 0.5 * (W - 1) + f * pc[0] / zc
        v = 0.5 * (H - 1) - f * pc[1] / zc
        rp = f * r / zc
        ys, xs = np.mgrid[int(math.floor(v - rp)):int(math.ceil(v + rp)) + 1,
                          int(math.floor(u - rp)):int(math.ceil(u + rp)) + 1]
        disk = (ys - v) ** 2 + (xs - u) ** 2 <= rp ** 2
        ys, xs = ys[disk], xs[disk]
        if len(ys) == 0:
            return out
        inside = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        out["in_fov"] = float(inside.mean())
        if not inside.any():
            return out
        ys, xs = ys[inside], xs[inside]
        dirs = self.cam.percept.dirs[ys, xs].astype(np.float64)
        c = pc.astype(np.float64)
        bdc = dirs @ c
        disc = bdc ** 2 - (c @ c - r * r)
        t_hit = bdc - np.sqrt(np.maximum(disc, 0.0))
        vis = (disc > 0) & (np.abs(depth[ys, xs] - t_hit) < 0.012)
        out["occlusion"] = float(1.0 - vis.sum() / max(len(disk[disk]), 1))
        return out

    def _decide(self, obs):
        tg = self.tgt
        if tg["n_dec"] >= self.max_decisions or bool(self.apples.detached[tg["gt_glob"]]):
            self._end_target()
            return
        obs = dict(obs)
        obs["state"] = obs["state"].copy()
        obs["state"][-N_CONTEXT:] = self._context_vec()
        a, info = self.policy.decide(self, obs)
        a = int(a)
        detached_now = self.B.detached_env(self.env)
        self.act = dict(
            env=self.env, target_uid=tg["uid"], dec_idx=tg["n_dec"], action=a,
            behavior_p=float(info.get("p", 1.0)), scores=info.get("scores"),
            obs_pre=obs, start_frame=self.B.frame, end_frame=None,
            broken0=self.B.broken_env(self.env), detached0=detached_now,
            wood0=self._bq()[self.B.wood_bodies[self.env], :3].copy(),
            gt0=self._bq()[tg["gt_body"], :3].copy(),
            arm_wood=0, base_wood=0, arm_apple=0, wood_disp=0.0, pull_max=0.0,
            placed=False, fail_reason="")
        tg["n_dec"] += 1
        self.n_decisions += 1
        self._target = tg["pos"].copy()
        if a == SKIP:
            self.act["end_frame"] = self.B.frame
            self._finish_action(obs, skipped=True)
            return
        base, _ = self._chassis_pose()
        away = self._horiz_unit(base, self._target)
        if a in PICKS:
            tg["n_pick"] += 1
            side = {0: 0.0, 1: +1.0, 2: -1.0}[a]
            appr = _rot2(away, -side * PICK_ANGLE)
            self._appr = appr / np.linalg.norm(appr)
            self._goto("REACH")
        else:
            side = +1.0 if a == 3 else -1.0
            rel = np.asarray(base[:2], float) - self._target[:2]
            dist = float(np.clip(np.linalg.norm(rel), 0.55, 0.80))
            rel = rel / (np.linalg.norm(rel) + 1e-9) * dist
            left = np.array([-away[1], away[0]])
            cand = [self._target[:2] + _rot2(rel, s * REPOS_ANGLE) for s in (+1.0, -1.0)]
            score = [side * float(np.dot(c - base[:2], left)) for c in cand]
            self._repos_goal = cand[int(np.argmax(score))]
            self._goto("REPOS")

    def _st_reach(self):
        a = self._appr
        pre = self._target - np.array([a[0], a[1], 0.0]) * self.PREGRASP
        if self._t_state % 15 == 1:
            q, err = self._ik_to(pre, approach_world=self._appr3(a, self._target))
            if err > 0.08:
                self._fail("unreachable")
                return
            self._set_arm(q)
        self._fingers(self.FINGER_OPEN)
        d = np.linalg.norm(self._tcp_world() - pre)
        if d < 0.05:
            self._goto("GRASP")
        elif self._stalled(d) or self._t_state > 300:
            self._fail("reach stalled")

    def _st_grasp(self):
        ap = self._bq()[self._apple_idx, :3]
        d = np.linalg.norm(ap - self._target, axis=1)
        j = int(d.argmin())
        if d[j] > 0.15:
            self._fail("no fruit at detection")
            return
        live = ap[j]
        tcp = self._tcp_world()
        if self._t_state == 1:
            self._servo_off = np.zeros(3)
        if self._t_state % 8 == 1:
            self._servo_off = np.clip(self._servo_off + 0.35 * (live - tcp), -0.09, 0.09)
            q, err = self._ik_to(live + self._servo_off,
                                 approach_world=self._appr3(self._appr, live))
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

    def _st_pull(self):
        a = self._appr
        if self._t_state > 25:
            self._retract_dist = min(self._retract_dist + 0.0022, self.RETRACT)
        goal = self._retract_from - np.array([a[0], a[1], 0.0]) * self._retract_dist
        if self._t_state % 10 == 1:
            q, err = self._ik_to(goal, approach_world=self._appr3(a, self._retract_from))
            self._set_arm(q)
        if self.apples.detached[self._target_apple]:
            self._goto("TRANSPORT")
        elif self._t_state > 360:
            self._fail("stem too strong / pull timeout")

    def _st_drop(self):
        if self._t_state == 1:
            self.apples.release(self._target_apple)
            self._fingers(self.FINGER_OPEN)
        if self._t_state < 60:
            return
        base, bq = self._chassis_pose()
        apw = self._bq()[int(self.tm.apple_bodies[self._target_apple]), :3]
        loc = _picker._inv_xform(base, bq, apw)
        placed = (abs(loc[0] - _robot._BUCKET_CENTER_X) < _robot._BUCKET_HALF + 0.02
                  and abs(loc[1]) < _robot._BUCKET_HALF + 0.02
                  and _robot._CHASSIS_Z < loc[2] < _robot._CHASSIS_Z + 0.6)
        self.act["placed"] = bool(placed)
        if not placed:
            self.act["fail_reason"] = "missed bucket"
        self._target_apple = -1
        self._set_arm(self.arm_home)
        self._end_action()

    def _st_repos(self):
        t = self._t_state
        if t == 1:
            self._set_arm(self.arm_home)
            self._fingers(self.FINGER_OPEN)
            self._drive_xy(0.0, 0.0, 0.0)
        if t < 35:
            return
        base, _ = self._chassis_pose()
        err = self._repos_goal - base[:2]
        dist = float(np.linalg.norm(err))
        rel = self._target[:2] - base[:2]
        yaw_err = _wrap(math.atan2(rel[1], rel[0]) - self._base_yaw())
        if (dist < 0.04 and abs(yaw_err) < 0.08) or t > 360:
            self._drive_xy(0.0, 0.0, 0.0)
            if t > 360:
                self.act["fail_reason"] = "other"
            self._end_action()
            return
        v = 1.5 * err
        sp = float(np.linalg.norm(v))
        if sp > 0.30:
            v *= 0.30 / sp
        self._drive_xy(float(v[0]), float(v[1]), float(np.clip(2.0 * yaw_err, -0.8, 0.8)))

    def _account(self):
        B = self.B
        e = self.env
        if self.state in ("PULL", "TRANSPORT", "DROP"):
            if B.pull_np is not None and self.tgt is not None:
                self.act["pull_max"] = max(self.act["pull_max"],
                                           float(B.pull_np[self.tgt["gt_glob"]]))
            return
        if B.last_contacts is not None:
            aw, bw, aa = B.last_contacts
            self.act["arm_wood"] += int(aw[e] > 0)
            self.act["base_wood"] += int(bw[e] > 0)
            self.act["arm_apple"] += int(aa[e] > 0)
        if B.frame % 4 == 0:
            wood = self._bq()[B.wood_bodies[self.env], :3]
            disp = np.linalg.norm(wood - self.act["wood0"], axis=1)
            far = np.linalg.norm(self.act["wood0"] - self.act["gt0"], axis=1) > 0.12
            if far.any():
                self.act["wood_disp"] = max(self.act["wood_disp"], float(disp[far].max()))
        if B.pull_np is not None and self.tgt is not None:
            self.act["pull_max"] = max(self.act["pull_max"],
                                       float(B.pull_np[self.tgt["gt_glob"]]))

    def _end_action(self):
        self.act["end_frame"] = self.B.frame
        self._obs_kind = "post"
        self._drive(0.0, 0.0)
        self._goto("OBSERVE")

    def _finish_action(self, obs_post, skipped=False):
        a = self.act
        tg = self.tgt
        e = self.env
        tau = (a["end_frame"] - a["start_frame"]) / FPS
        det_after = self.B.detached_env(e)
        n0 = self.B.n_env_apples
        tgt_local = tg["gt_glob"] - e * n0
        newly = det_after & ~a["detached0"]
        detach = bool(newly[tgt_local]) or bool(det_after[tgt_local] and not a["detached0"][tgt_local])
        newly[tgt_local] = False
        collateral = int(newly.sum())
        broken = self.B.broken_env(e) - a["broken0"]
        contact_frames = a["arm_wood"] + a["base_wood"]
        contact = contact_frames >= 2 and a["wood_disp"] >= CONTACT_DISP
        damage = (broken > 0) or (collateral > 0)
        success = bool(a["placed"])
        gt1 = self._bq()[tg["gt_body"], :3]
        rec = dict(
            env=e, target_uid=a["target_uid"], dec_idx=a["dec_idx"], action=a["action"],
            behavior_p=a["behavior_p"], skipped=skipped,
            detach=detach, success=success, contact=bool(contact), damage=bool(damage),
            visible_after=bool(obs_post["det"]) and not detach,
            tau=float(tau), target_disp=(float(np.linalg.norm(gt1 - a["gt0"])) if not detach else np.nan),
            wood_disp=float(a["wood_disp"]), pull_max=float(a["pull_max"]),
            arm_wood_frames=int(a["arm_wood"]), base_wood_frames=int(a["base_wood"]),
            arm_apple_frames=int(a["arm_apple"]), branches_broken=int(broken),
            collateral_drops=collateral,
            fail_code=(FAIL_CODES.index(a["fail_reason"]) if a["fail_reason"] in FAIL_CODES
                       else len(FAIL_CODES) - 1),
            reward=reward(success, contact, damage, tau),
            start_frame=a["start_frame"], end_frame=a["end_frame"],
            obs_pre=a["obs_pre"], obs_post=obs_post, scores=a.get("scores"))
        self.sink(rec)
        tg["prev_a"] = a["action"]
        tg["prev_contact"] = int(contact)
        tg["prev_detach"] = int(detach)
        self.act = None
        if skipped or detach or tg["n_dec"] >= self.max_decisions:
            self._end_target()
        else:
            self._decide(obs_post)
