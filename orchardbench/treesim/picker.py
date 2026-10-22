from __future__ import annotations

import math

import numpy as np
import warp as wp
import newton

from . import robot as _robot


class ArmIK:

    def __init__(self, arm_home: list, device=None):
        import newton.ik as ik
        b = newton.ModelBuilder()
        b.add_urdf(_robot.franka_urdf_path(),
                   xform=wp.transform(p=wp.vec3(*_robot._ARM_MOUNT), q=wp.quat_identity()),
                   floating=False, enable_self_collisions=False)

        names = [l.rsplit("/", 1)[-1] for l in b.joint_label]
        self._arm_q = []
        for n, h in _robot._ARM_HOME.items():
            j = names.index(n)
            b.joint_q[b.joint_q_start[j]] = h
            self._arm_q.append(int(b.joint_q_start[j]))
        self.model = b.finalize()
        self.tcp = next(i for i, l in enumerate(self.model.body_label)
                        if l.endswith("fr3_hand_tcp"))
        self.q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3()], dtype=wp.vec3))
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.tcp, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4),
            weight=0.15)
        lim = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper, weight=10.0)
        self.solver = ik.IKSolver(model=self.model, n_problems=1,
                                  objectives=[self.pos_obj, self.rot_obj, lim],
                                  lambda_initial=0.1,
                                  jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self._state = self.model.state()

    def solve(self, target_base: np.ndarray, approach_base: np.ndarray,
              q_init: np.ndarray, iters: int = 32):
        qh = self.q.numpy()
        qh[0, self._arm_q] = q_init
        self.q.assign(qh)
        z = approach_base / (np.linalg.norm(approach_base) + 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, up)
        if np.linalg.norm(y) < 1e-6:
            y = np.array([0.0, 1.0, 0.0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        quat = wp.quat_from_matrix(wp.mat33f(x[0], y[0], z[0],
                                             x[1], y[1], z[1],
                                             x[2], y[2], z[2]))
        self.pos_obj.set_target_position(0, wp.vec3(*map(float, target_base)))
        self.rot_obj.set_target_rotation(0, wp.vec4(quat[0], quat[1], quat[2], quat[3]))
        self.solver.step(self.q, self.q, iterations=iters)
        qs = self.q.numpy()[0]

        self.model.joint_q.assign(qs)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self._state)
        tcp = self._state.body_q.numpy()[self.tcp, :3]
        return qs[self._arm_q].copy(), float(np.linalg.norm(tcp - target_base))


def _inv_xform(p, q, world_pt):
    u, w = q[:3], q[3]
    v = world_pt - p

    u = -u
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _rot(q, v):
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class AutoPicker:

    STANDOFF = (0.62, 0.80)
    SHOULDER = (0.15, 0.64)
    ARM_REACH = 0.83
    REACH_Z = (0.45, 1.32)
    PREGRASP = 0.16
    GRASP_R = 0.042
    RETRACT = 0.32
    FINGER_OPEN, FINGER_CLOSED = 0.04, 0.016
    STALL_FRAMES = 55


    def __init__(self, sim, tm, cam, driver, rp, metrics=None, env: int = 0,
                 ik: "ArmIK | None" = None):
        self.sim = sim
        self.tm = tm
        self.cam = cam
        self.drv = driver
        self.rp = rp
        self.met = metrics
        self.env = int(env)


        self.ik = ik if ik is not None else ArmIK(list(_robot._ARM_HOME.values()))
        rb = tm.robot_data
        self.chassis = int(rb["chassis"][self.env])
        self.wrist = int(rb["wrist"][self.env])
        self.arm_tq = np.asarray(rb["arm_tq"][self.env], dtype=int)
        self.arm_dof = np.asarray(rb["arm_dofs"][self.env], dtype=int)
        self.arm_home = np.asarray(rb["arm_home"], dtype=float)
        self.apples = sim.apples
        if self.apples is None or not len(tm.apple_bodies):
            raise ValueError("autonomous picking needs apples (--apples)")
        self.n_env_apples = (len(tm.apple_bodies) // max(tm.num_envs, 1))
        e0 = self.env * self.n_env_apples
        env_bodies = np.asarray(
            tm.apple_bodies[e0:e0 + self.n_env_apples], dtype=int)


        first_shape = {}
        for s, b in enumerate(tm.model.shape_body.numpy()):
            first_shape.setdefault(int(b), int(s))
        ssc = tm.model.shape_scale.numpy()
        grown = np.array([ssc[first_shape[int(b)]][0] >= 0.015
                          for b in env_bodies], dtype=bool)
        self._apple_idx = env_bodies[grown]
        self._apple_glob = (e0 + np.where(grown)[0])


        self._zone_ref = None
        if len(self._apple_idx):
            ap0 = sim.state_0.body_q.numpy()[self._apple_idx, :3]
            reachable = ((ap0[:, 2] >= self.REACH_Z[0]) &
                         (ap0[:, 2] <= self.REACH_Z[1]))
            cloud = ap0[reachable] if reachable.any() else ap0
            trunk_xy = np.median(cloud[:, :2], axis=0)
            z_lo, z_hi = np.percentile(cloud[:, 2], [5, 95])
            r = np.linalg.norm(cloud[:, :2] - trunk_xy, axis=1)
            self._zone_ref = dict(trunk_xy=trunk_xy, z_lo=float(z_lo),
                                  z_hi=float(max(z_hi, z_lo + 1e-3)),
                                  r_split=float(np.median(r)) if len(r) else 0.0)
            if self.met is not None:
                census: dict = {}
                for pos in cloud:
                    lab = self._zone_of(pos)[0]
                    if lab:
                        census[lab] = census.get(lab, 0) + 1
                self.met.set_apple_census(census)
        self._q_goal = self.arm_home.copy()
        self._q_cmd = self.arm_home.copy()
        self.state = "SCAN"
        self._t_state = 0
        self._target = None
        self._target_apple = -1
        self._blacklist: list = []
        self._frame = 0
        self._track: list = []
        self._scan_dir = 1.0
        self._drop_wait = 0
        self._best_dist = None
        self._stall = 0
        self._orbits = 0
        self._orbit_dir = 1.0
        self._pos_hist: list = []
        self._cmd_v = 0.0


        self._belief = None
        self._unseen = 0
        self._belief_miss = 0
        self._last_phi = 1.15
        self.done = False


    def _bq(self):


        return self.sim.body_q_np()

    def _jq(self):
        return self.sim.joint_q_np()

    def _chassis_pose(self):
        q = self._bq()[self.chassis]
        return q[:3].copy(), q[3:].copy()

    def _tcp_world(self):
        q = self._bq()[self.wrist]
        return q[:3] + _rot(q[3:], np.array([0.0, 0.0, 0.10]))

    def _set_arm(self, q_goal):
        self._q_goal = np.asarray(q_goal, dtype=float)

    def _slew_arm(self, rate=0.045):
        d = np.clip(self._q_goal - self._q_cmd, -rate, rate)
        self._q_cmd = self._q_cmd + d


        self.drv._tq_host[self.arm_tq] = self._q_cmd
        self.sim.control.joint_target_q.assign(self.drv._tq_host)

    def _fingers(self, opening):
        self._q_goal[-2:] = opening
        self._q_cmd[-2:] = opening

    def _drive(self, v, w):


        self._cmd_v = float(v)
        ds = self.drv.planar_dof[self.env]
        yaw = self._base_yaw()
        th = self.drv._target_host
        th[ds + 0] = v * math.cos(yaw)
        th[ds + 1] = v * math.sin(yaw)
        th[ds + 3] = w
        self.sim.control.joint_target_qd.assign(th)

    def _base_yaw(self):
        return float(self._jq()[self.drv.planar_q[self.env] + 3])


    def _update_tracks(self):
        dets = self.cam.dets_env[self.env] or []
        for t in self._track:
            t[2] += 1
        for d in dets:
            for t in self._track:


                if np.linalg.norm(t[0] - d.center_world) < 0.09:
                    t[0] = 0.5 * (t[0] + d.center_world)
                    t[1] += 1
                    t[2] = 0
                    break
            else:
                self._track.append([d.center_world.copy(), 1, 0])
        self._track = [t for t in self._track if t[2] < 90]

        if self.met is not None and dets:
            ap = self._bq()[self._apple_idx, :3]
            tp = sum(1 for d in dets
                     if np.min(np.linalg.norm(ap - d.center_world, axis=1)) < 0.08)
            self.met.detection_frame(tp, len(dets) - tp, len(dets))

    def _confirmed_targets(self):
        out = []
        for pos, hits, age in self._track:


            if hits < 3 or age > 30:
                continue
            if not (self.REACH_Z[0] <= pos[2] <= self.REACH_Z[1]):
                continue
            if any(np.linalg.norm(pos - b) < 0.09 for b, exp in self._blacklist
                   if exp > self._frame):
                continue
            out.append(pos)
        return out


    def _goto(self, state):
        self.state = state
        self._t_state = 0
        self._best_dist = None
        self._stall = 0

    def _zone_of(self, pos):
        zr = self._zone_ref
        if zr is None:
            return (None, None, None)
        t = (float(pos[2]) - zr["z_lo"]) / (zr["z_hi"] - zr["z_lo"])
        v = "lower" if t < 1.0 / 3.0 else ("middle" if t < 2.0 / 3.0 else "upper")
        r = float(np.linalg.norm(np.asarray(pos[:2], float) - zr["trunk_xy"]))
        rad = "inner" if r < zr["r_split"] else "outer"
        return (f"{v}-{rad}", v, rad)

    def _stalled(self, dist, tol=0.005) -> bool:
        if self._best_dist is None or dist < self._best_dist - tol:
            self._best_dist = dist
            self._stall = 0
            return False
        self._stall += 1
        return self._stall >= self.STALL_FRAMES

    def _fail(self, reason):
        if self._target is not None:

            ttl = 5400 if reason in ("unreachable", "grasp unreachable") else 1500
            self._blacklist.append((self._target.copy(), self._frame + ttl))
        if self._target_apple >= 0:
            self.apples.release(self._target_apple)
        if self.met is not None:
            self.met.pick_end(False, reason)
            self.met.count("pick_failures")
        self._target = None
        self._target_apple = -1
        self._set_arm(self.arm_home)
        self._fingers(self.FINGER_OPEN)
        self._goto("SCAN")

    def update(self):
        if self.done:
            return
        self._t_state += 1
        self._frame += 1
        self.drv.follow_ground(frame=self._frame)
        self._update_tracks()


        base_xy = self._bq()[self.chassis, :2].copy()
        self._pos_hist.append((base_xy, abs(getattr(self, "_cmd_v", 0.0)) > 0.12))
        if len(self._pos_hist) > 90:
            self._pos_hist.pop(0)


        stuck = False
        if self.state in ("SCAN", "ALIGN", "ORBIT"):
            if len(self._pos_hist) >= 50:
                seg = self._pos_hist[-50:]
                stuck = (sum(c for _, c in seg) > 38
                         and np.linalg.norm(base_xy - seg[0][0]) < 0.02)
            if not stuck and len(self._pos_hist) == 90:
                stuck = (sum(c for _, c in self._pos_hist) > 70
                         and np.linalg.norm(base_xy - self._pos_hist[0][0]) < 0.05)
        if stuck:
            self._pos_hist.clear()
            if self.met is not None:
                self.met.count("stuck_recoveries")
            if self.state == "ALIGN" and self._target is not None and self._orbits < 2:


                self._orbits += 1
                base, _ = self._chassis_pose()
                rel = self._target[:2] - base[:2]
                err = (math.atan2(rel[1], rel[0]) - self._base_yaw()
                       + math.pi) % (2 * math.pi) - math.pi
                self._orbit_dir = 1.0 if err >= 0.0 else -1.0
                self._goto("ORBIT")
            else:
                self._target = None
                self._target_apple = -1
                self._set_arm(self.arm_home)
                self._scan_dir *= -1.0
                self._goto("RECOVER")
        getattr(self, "_st_" + self.state.lower())()
        self._slew_arm()
        if self.env != 0:
            return


        self.cam.highlight = None if self._target is None else self._target.copy()
        viewer = self.sim.viewer
        if viewer is not None:
            try:
                viewer.log_points("pick target", None, hidden=True)
            except Exception:
                pass


    def _st_scan(self):
        self._fingers(self.FINGER_OPEN)
        targets = self._confirmed_targets()
        if targets:
            base, _ = self._chassis_pose()
            self._target = min(targets, key=lambda p: np.linalg.norm(p[:2] - base[:2]))
            self._orbits = 0
            if self.met is not None:
                lab, zv, zr = self._zone_of(self._target)
                self.met.pick_start(-1, self._target, zone=lab,
                                    zone_v=zv, zone_r=zr)
            self._goto("ALIGN")
            return
        d = self.cam.depth_env[self.env]
        if d is None:
            self._drive(0.0, 0.5 * self._scan_dir)
            return
        rng = float(self.rp.camera_range)
        hit = np.isfinite(d) & (d > 0.3) & (d < 4.0) & (np.abs(d - rng) > 1e-4)


        scene = self.cam.scene_env[self.env]
        if scene is not None and scene.shape == d.shape:
            hit &= scene
        frac = float(hit.mean())
        if frac >= 0.06:


            c = self.cam.centroid_env[self.env]
            if c is not None:
                cxy = np.asarray(c[:2], dtype=float)
                self._belief = (cxy if self._belief is None
                                else 0.85 * self._belief + 0.15 * cxy)
            self._unseen = 0
            self._belief_miss = 0
        else:


            self._unseen += 1
            base, _ = self._chassis_pose()
            if self._belief is None:
                self._drive(0.0, 0.55 * self._scan_dir)
                return
            rel = self._belief - base[:2]
            dist = float(np.linalg.norm(rel))
            b_err = (math.atan2(rel[1], rel[0]) - self._base_yaw()
                     + math.pi) % (2 * math.pi) - math.pi
            if self._unseen < 150 and dist > 0.5:

                w = 1.4 * (b_err + self._scan_dir * self._last_phi)
                self._drive(0.22, float(np.clip(w, -0.75, 0.75)))
            elif abs(b_err) > 0.15:

                self._belief_miss = max(self._belief_miss - 1, 0)
                w = float(np.clip(2.0 * b_err, -0.75, 0.75))
                self._drive(0.0, math.copysign(max(abs(w), 0.3), w))
            else:

                self._belief_miss += 1
                self._drive(0.0, 0.4 * self._scan_dir)
                if self._belief_miss > 100:
                    self._belief = None
            return
        cols = hit.sum(axis=0)
        cx = float((cols * np.arange(d.shape[1])).sum() / max(cols.sum(), 1))
        off = (cx - 0.5 * (d.shape[1] - 1)) / d.shape[1]
        fov_h = math.radians(float(self.rp.camera_fov)) * d.shape[1] / d.shape[0]
        bearing_err = -off * fov_h
        med = float(np.median(d[hit]))
        near = float(np.percentile(d[hit], 8))
        if abs(off) > 0.22:

            self._drive(0.0, float(np.clip(1.6 * bearing_err, -0.7, 0.7)))
        elif med > 1.55 and near > 0.9:

            self._drive(0.28, float(np.clip(1.2 * bearing_err, -0.5, 0.5)))
        else:


            phi = float(np.clip(1.15 + 1.2 * (0.95 - near), 0.7, 1.75))
            self._last_phi = phi
            w = 1.4 * (bearing_err + self._scan_dir * phi)
            self._drive(0.22, float(np.clip(w, -0.75, 0.75)))
        if self._t_state > 5400:
            self._drive(0.0, 0.0)
            self.done = True

    def _standoff_band(self, target_z: float):
        dz = abs(float(target_z) - self.SHOULDER[1])
        planar = math.sqrt(max(self.ARM_REACH ** 2 - dz ** 2, 0.05))
        hi = min(self.STANDOFF[1], self.SHOULDER[0] + 0.92 * planar)
        lo = max(0.45, hi - 0.16)
        return lo, hi


    def _st_align(self):
        base, bq = self._chassis_pose()
        rel = self._target[:2] - base[:2]
        dist = float(np.linalg.norm(rel))
        bearing = math.atan2(rel[1], rel[0])
        err = (bearing - self._base_yaw() + math.pi) % (2 * math.pi) - math.pi
        so_lo, so_hi = self._standoff_band(self._target[2])
        v = 0.0
        if dist > so_hi:
            v = min(0.5, 1.2 * (dist - so_hi) + 0.08)
        elif dist < so_lo:
            v = max(-0.3, -1.2 * (so_lo - dist) - 0.05)
        self._drive(v if abs(err) < 0.5 else 0.0, float(np.clip(1.8 * err, -0.9, 0.9)))
        if abs(err) < 0.10 and so_lo <= dist <= so_hi:
            self._drive(0.0, 0.0)
            self._goto("REACH")
            return


        goal_err = abs(dist - float(np.clip(dist, so_lo, so_hi))) + 0.4 * abs(err)
        if self._stalled(goal_err, tol=0.01) and self._stall >= 2 * self.STALL_FRAMES:


            if self._orbits < 2:
                self._orbits += 1
                self._orbit_dir = 1.0 if err >= 0.0 else -1.0
                self._goto("ORBIT")
            else:
                self._fail("blocked (orbits exhausted)")
        elif self._t_state > 600:
            self._fail("align timeout")


    def _st_recover(self):
        if self._t_state < 95:
            self._drive(-0.32, 0.0)
        elif self._t_state < 140:
            self._drive(0.0, 0.7 * self._scan_dir)
        else:
            self._drive(0.0, 0.0)
            self._goto("SCAN")


    def _st_orbit(self):
        if self._t_state < 55:
            self._drive(-0.30, 0.0)
            return
        base, _ = self._chassis_pose()
        rel = self._target[:2] - base[:2]
        bearing = math.atan2(rel[1], rel[0])
        want = bearing + self._orbit_dir * 1.25
        err = (want - self._base_yaw() + math.pi) % (2 * math.pi) - math.pi
        self._drive(0.35 if abs(err) < 0.6 else 0.0,
                    float(np.clip(2.0 * err, -0.9, 0.9)))
        if self._t_state > 235:
            self._goto("ALIGN")

    def _ik_to(self, world_pt, approach_world=None):
        base, bq = self._chassis_pose()
        tgt = _inv_xform(base, bq, world_pt)
        if approach_world is None:
            a = world_pt - base
            approach_world = np.array([a[0], a[1], 0.15 * a[2]])
        app = _rot(np.array([-bq[0], -bq[1], -bq[2], bq[3]]), approach_world)
        if np.linalg.norm(app) < 1e-6:
            app = np.array([1.0, 0.0, 0.0])
        q, err = self.ik.solve(tgt, app, self._q_cmd[:len(self.ik._arm_q)])
        return q, err


    def _st_reach(self):
        base, _ = self._chassis_pose()
        away = self._target[:2] - base[:2]
        away = away / (np.linalg.norm(away) + 1e-9)
        pre = self._target - np.array([away[0], away[1], 0.0]) * self.PREGRASP
        if self._t_state % 15 == 1:
            q, err = self._ik_to(pre)
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
            self._servo_off = np.clip(self._servo_off + 0.35 * (live - tcp),
                                      -0.09, 0.09)
            q, err = self._ik_to(live + self._servo_off)
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
            if self.met is not None:
                self.met.pick_event("grasp")
                self.met.count("grasps")
            self._goto("PULL")
        elif self._stalled(tcp_d) or self._t_state > 240:
            self._fail("grasp stalled")


    def _st_pull(self):
        base, _ = self._chassis_pose()
        away = base[:2] - self._retract_from[:2]
        away = away / (np.linalg.norm(away) + 1e-9)


        if self._t_state > 25:
            self._retract_dist = min(self._retract_dist + 0.0022, self.RETRACT)
        goal = self._retract_from + np.array([away[0], away[1], 0.0]) * self._retract_dist
        if self._t_state % 10 == 1:
            q, err = self._ik_to(goal)
            self._set_arm(q)
        if self.met is not None:
            self.met.pick_pull(float(self.apples.pull_forces()[self._target_apple]))
        if self.apples.detached[self._target_apple]:
            if self.met is not None:
                self.met.pick_event("detach")
                self.met.count("picks")
            self._goto("TRANSPORT")
        elif self._t_state > 360:
            self._fail("stem too strong / pull timeout")


    def _st_transport(self):
        base, bq = self._chassis_pose()
        over = base + _rot(bq, np.array([_robot._BUCKET_CENTER_X, 0.0,
                                         _robot._CHASSIS_Z + _robot._CHASSIS[2]
                                         + _robot._BUCKET_WALL_H + 0.11]))
        if self._t_state % 15 == 1:
            q, err = self._ik_to(over)
            self._set_arm(q)
        d = np.linalg.norm(self._tcp_world() - over)


        apw = self._bq()[int(self.tm.apple_bodies[self._target_apple])]
        aspeed = float(np.linalg.norm(
            self.sim.state_0.body_qd.numpy()[int(self.tm.apple_bodies[self._target_apple]), :3]))
        centred = np.linalg.norm(apw[:2] - over[:2]) < 0.10
        if (d < 0.07 and centred and aspeed < 0.35) \
                or self._stalled(d, tol=0.01) and self._stall > 2 * self.STALL_FRAMES \
                or self._t_state > 420:
            self._goto("DROP")


    def _st_drop(self):
        if self._t_state == 1:
            self.apples.release(self._target_apple)
            self._fingers(self.FINGER_OPEN)
        if self._t_state < 60:
            return
        base, bq = self._chassis_pose()
        apw = self._bq()[int(self.tm.apple_bodies[self._target_apple]), :3]
        loc = _inv_xform(base, bq, apw)
        placed = (abs(loc[0] - _robot._BUCKET_CENTER_X) < _robot._BUCKET_HALF + 0.02
                  and abs(loc[1]) < _robot._BUCKET_HALF + 0.02
                  and _robot._CHASSIS_Z < loc[2] < _robot._CHASSIS_Z + 0.6)
        if self.met is not None:
            self.met.pick_end(placed, None if placed else "missed bucket")
            self.met.count("places" if placed else "missed_bucket")
        self._blacklist.append((self._target.copy(), self._frame + 10 ** 9))
        self._target = None
        self._target_apple = -1
        self._set_arm(self.arm_home)
        self._goto("SCAN")
