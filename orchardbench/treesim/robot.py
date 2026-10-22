from __future__ import annotations

import math
import os

import numpy as np
import warp as wp
import newton

from .config import RobotParams


_ARM_HOME = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -math.pi / 4.0,
    "fr3_joint3": 0.0,
    "fr3_joint4": -1.9,
    "fr3_joint5": 0.0,
    "fr3_joint6": 2.77,
    "fr3_joint7": math.pi / 4.0,
    "fr3_finger_joint1": 0.02,
    "fr3_finger_joint2": 0.02,
}

_CHASSIS = (0.48, 0.40, 0.14)
_CHASSIS_Z = 0.17
_CHASSIS_COLOR = (0.15, 0.15, 0.17)
_WHEEL_COLOR = (0.05, 0.05, 0.05)
_ARM_MOUNT = (0.15, 0.0, 0.31)
_BUCKET_CENTER_X = -0.26
_BUCKET_HALF = 0.15
_BUCKET_WALL_H = 0.15
_BUCKET_COLOR = (0.45, 0.32, 0.18)


def franka_urdf_path() -> str:
    import newton.utils
    return str(newton.utils.download_asset("franka_emika_panda")
               / "urdf/fr3_franka_hand.urdf")


def build_robot(builder: "newton.ModelBuilder", rp: RobotParams) -> dict:
    X = wp.vec3(1.0, 0.0, 0.0)
    Y = wp.vec3(0.0, 1.0, 0.0)
    Z = wp.vec3(0.0, 0.0, 1.0)


    chassis = builder.add_link(label="ridgeback")
    ccfg = builder.ShapeConfig(density=600.0, mu=0.8, collision_group=3)
    builder.add_shape_box(chassis,
                          xform=wp.transform(p=wp.vec3(0.0, 0.0, _CHASSIS_Z), q=wp.quat_identity()),
                          hx=_CHASSIS[0], hy=_CHASSIS[1], hz=_CHASSIS[2],
                          cfg=ccfg, color=_CHASSIS_COLOR)

    wcfg = builder.ShapeConfig(density=0.0, mu=0.8, collision_group=3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            builder.add_shape_cylinder(
                chassis,
                xform=wp.transform(p=wp.vec3(sx * 0.30, sy * 0.36, 0.10),
                                   q=wp.quat_from_axis_angle(X, math.pi / 2)),
                radius=0.10, half_height=0.04, cfg=wcfg, color=_WHEEL_COLOR)


    bcfg = builder.ShapeConfig(density=0.0, mu=0.9, collision_group=3)
    bx, bz = _BUCKET_CENTER_X, _CHASSIS_Z + _CHASSIS[2]
    hw, wh, wt = _BUCKET_HALF, _BUCKET_WALL_H, 0.012
    builder.add_shape_box(chassis, xform=wp.transform(p=wp.vec3(bx, 0.0, bz + 0.006),
                                                      q=wp.quat_identity()),
                          hx=hw + wt, hy=hw + wt, hz=0.006, cfg=bcfg, color=_BUCKET_COLOR)
    for sx, sy, hxx, hyy in ((1, 0, wt, hw + wt), (-1, 0, wt, hw + wt),
                             (0, 1, hw + wt, wt), (0, -1, hw + wt, wt)):
        builder.add_shape_box(
            chassis,
            xform=wp.transform(p=wp.vec3(bx + sx * (hw + wt), sy * (hw + wt), bz + wh * 0.5),
                               q=wp.quat_identity()),
            hx=hxx, hy=hyy, hz=wh * 0.5, cfg=bcfg, color=_BUCKET_COLOR)


    def vdof(axis, kd, effort):
        return builder.JointDofConfig(
            axis=axis, target_pos=0.0, target_ke=0.0, target_kd=kd,
            limit_lower=-1.0e9, limit_upper=1.0e9, limit_ke=0.0, limit_kd=0.0,
            effort_limit=effort,
            actuator_mode=newton.JointTargetMode.VELOCITY)

    zdof = builder.JointDofConfig(
        axis=Z, target_pos=0.0, target_ke=4.0e5, target_kd=2.0e4,
        limit_lower=-1.0e9, limit_upper=1.0e9, limit_ke=0.0, limit_kd=0.0,
        effort_limit=1.0e5,
        actuator_mode=newton.JointTargetMode.POSITION)

    x0, y0 = rp.position
    de = float(getattr(rp, "drive_effort", 900.0))
    te = float(getattr(rp, "turn_effort", 350.0))
    planar = builder.add_joint_d6(
        parent=-1, child=chassis,
        linear_axes=[vdof(X, rp.drive_kd, de), vdof(Y, rp.drive_kd, de), zdof],
        angular_axes=[vdof(Z, rp.turn_kd, te)],
        parent_xform=wp.transform(p=wp.vec3(float(x0), float(y0), 0.0),
                                  q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))


    builder.joint_q[builder.joint_q_start[planar] + 3] = float(rp.yaw)
    builder.add_articulation([planar], label="ridgeback_franka")


    nb0 = builder.body_count
    nj0 = builder.joint_count
    builder.add_urdf(
        franka_urdf_path(), parent_body=chassis,
        xform=wp.transform(p=wp.vec3(*_ARM_MOUNT), q=wp.quat_identity()),
        floating=False, enable_self_collisions=False)


    arm_jids: list[int] = []
    arm_home: list[float] = []
    for j in range(nj0, builder.joint_count):
        name = builder.joint_label[j].rsplit("/", 1)[-1]
        home = _ARM_HOME.get(name)
        if home is None:
            continue
        qs = builder.joint_q_start[j]
        ds = builder.joint_qd_start[j]
        finger = "finger" in name
        builder.joint_q[qs] = home


        builder.joint_target_ke[ds] = rp.arm_kp * (0.5 if finger else 1.0)
        builder.joint_target_kd[ds] = rp.arm_kd * (0.3 if finger else 1.0)
        builder.joint_target_mode[ds] = newton.JointTargetMode.POSITION
        arm_jids.append(int(j))
        arm_home.append(float(home))

    wrist = next(b for b in range(nb0, builder.body_count)
                 if builder.body_label[b].endswith("fr3_hand"))
    return dict(
        chassis=chassis,
        wrist=int(wrist),
        planar_joint=int(planar),
        arm_jids=arm_jids,
        arm_home=arm_home,
        nbody=builder.body_count - chassis,
    )


@wp.kernel
def _twig_brush(body_q: wp.array(dtype=wp.transform),
                body_qd: wp.array(dtype=wp.spatial_vector),
                twig_body: wp.array(dtype=wp.int32),
                twig_hl: wp.array(dtype=wp.float32),
                twig_r: wp.array(dtype=wp.float32),
                twig_env: wp.array(dtype=wp.int32),
                twig_on: wp.array(dtype=wp.int32),
                proxy_body: wp.array(dtype=wp.int32),
                proxy_off: wp.array(dtype=wp.vec3),
                proxy_r: wp.array(dtype=wp.float32),
                proxy_env: wp.array(dtype=wp.int32),
                n_proxy: int, k: float, cd: float, clat: float, fmax: float,
                body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if twig_on[i] == 0:
        return
    tb = twig_body[i]
    X = body_q[tb]
    a = wp.transform_get_translation(X)
    axis = wp.transform_vector(X, wp.vec3(0.0, 0.0, 1.0))
    rt = twig_r[i]
    e = twig_env[i]
    hl = twig_hl[i]
    com = a + axis * hl
    vt = wp.spatial_top(body_qd[tb])
    wt = wp.spatial_bottom(body_qd[tb])
    f = wp.vec3(0.0, 0.0, 0.0)
    tau = wp.vec3(0.0, 0.0, 0.0)
    for p in range(n_proxy):
        if proxy_env[p] != e:
            continue
        pb = proxy_body[p]
        Xp = body_q[pb]
        po = wp.transform_get_translation(Xp)
        c = wp.transform_point(Xp, proxy_off[p])

        t = wp.clamp(wp.dot(c - a, axis), 0.0, 2.0 * hl)
        q = a + axis * t
        dvec = q - c
        dist = wp.length(dvec)
        pen = proxy_r[p] + rt - dist
        if pen > 0.0 and dist > 1.0e-6:
            n = dvec / dist

            vq = vt + wp.cross(wt, q - a)
            vc = wp.spatial_top(body_qd[pb]) + wp.cross(wp.spatial_bottom(body_qd[pb]), c - po)
            vrel_n = wp.dot(vq - vc, n)
            fn = k * pen
            if vrel_n < 0.0:
                fn = fn - cd * vrel_n
            fn = wp.min(fn, fmax)
            push = n * fn


            sp = wp.length(vc)
            if sp > 0.05:
                vh = vc / sp
                lat = n - wp.dot(n, vh) * vh
                ll = wp.length(lat)
                if ll < 0.25:
                    lat = wp.cross(vh, axis)
                    ll = wp.length(lat)
                if ll > 1.0e-4:
                    push = push + (lat / ll) * wp.min(clat * pen * sp, fmax)
            f = f + push
            tau = tau + wp.cross(q - com, push)

            wp.atomic_add(body_f, pb,
                          wp.spatial_vector(-push, wp.cross(c - po, -push)))
    if wp.length(f) > 0.0:
        wp.atomic_add(body_f, tb, wp.spatial_vector(f, tau))


class TwigBrush:


    _CHASSIS_PROXIES = [(0.34, 0.26, 0.20), (0.34, -0.26, 0.20),
                        (-0.34, 0.26, 0.20), (-0.34, -0.26, 0.20),
                        (0.44, 0.0, 0.20), (-0.44, 0.0, 0.20)]
    _CHASSIS_PR = 0.26
    _ARM_PR = 0.085

    def __init__(self, tm):
        model = tm.model
        dev = model.device
        ne = max(int(tm.num_envs), 1)
        bpe = model.body_count // ne

        ssc = model.shape_scale.numpy()
        sb = model.shape_body.numpy()
        st = model.shape_type.numpy() if hasattr(model, "shape_type") else None
        tb, thl, trr, te = [], [], [], []
        n_seg = tm.n_bodies
        for e in range(ne):
            s0 = e * (model.shape_count // ne)
            for k in range(n_seg):
                r, hl = float(ssc[s0 + k][0]), float(ssc[s0 + k][1])
                if r < 0.010:
                    tb.append(int(sb[s0 + k]))
                    thl.append(hl)
                    trr.append(r)
                    te.append(e)
        pb, po, pr, pe = [], [], [], []
        for e in range(ne):
            ch = int(tm.robot_data["chassis"][e])
            for ox, oy, oz in self._CHASSIS_PROXIES:
                pb.append(ch); po.append((ox, oy, oz)); pr.append(self._CHASSIS_PR); pe.append(e)
            wrist = int(tm.robot_data["wrist"][e])
            for lb in range(ch + 1, min(wrist + 4, (e + 1) * bpe)):
                pb.append(lb); po.append((0.0, 0.0, 0.0)); pr.append(self._ARM_PR); pe.append(e)
        self.n = len(tb)
        self.n_proxy = len(pb)
        if self.n == 0 or self.n_proxy == 0:
            self.n = 0
            return
        self.twig_body = wp.array(np.asarray(tb, dtype=np.int32), device=dev)
        self.twig_hl = wp.array(np.asarray(thl, dtype=np.float32), device=dev)
        self.twig_r = wp.array(np.asarray(trr, dtype=np.float32), device=dev)
        self.twig_env = wp.array(np.asarray(te, dtype=np.int32), device=dev)
        self._tb_host = np.asarray(tb, dtype=np.int64)
        self._on_host = np.ones(self.n, dtype=np.int32)
        self.twig_on = wp.array(self._on_host, device=dev)
        self.proxy_body = wp.array(np.asarray(pb, dtype=np.int32), device=dev)
        self.proxy_off = wp.array(np.asarray(po, dtype=np.float32), dtype=wp.vec3, device=dev)
        self.proxy_r = wp.array(np.asarray(pr, dtype=np.float32), device=dev)
        self.proxy_env = wp.array(np.asarray(pe, dtype=np.int32), device=dev)
        self.dev = dev


        self._k = float(os.environ.get("TWIG_K", 1200.0))
        self._cd = float(os.environ.get("TWIG_CD", 45.0))
        self._clat = float(os.environ.get("TWIG_CLAT", 150.0))
        self._fmax = float(os.environ.get("TWIG_FMAX", 30.0))

    def deactivate_bodies(self, bodies):
        if self.n == 0:
            return
        hit = np.isin(self._tb_host, np.asarray(list(bodies), dtype=np.int64))
        if hit.any():
            self._on_host[hit] = 0
            self.twig_on.assign(self._on_host)

    def apply(self, state):
        if self.n == 0:
            return


        wp.launch(_twig_brush, dim=self.n,
                  inputs=[state.body_q, state.body_qd, self.twig_body, self.twig_hl,
                          self.twig_r, self.twig_env, self.twig_on, self.proxy_body,
                          self.proxy_off, self.proxy_r, self.proxy_env,
                          self.n_proxy, self._k, self._cd, self._clat, self._fmax,
                          state.body_f],
                  device=self.dev)


def take_over_wasd(viewer) -> bool:
    gui = getattr(viewer, "gui", None)
    if gui is None or not hasattr(gui, "update_camera_from_keys"):
        return False
    orig = gui.update_camera_from_keys

    def filtered(dt, is_key_down, _orig=orig):
        try:
            import pyglet
            k = pyglet.window.key
            blocked = (k.W, k.A, k.S, k.D)
        except Exception:
            return _orig(dt, is_key_down)
        return _orig(dt, lambda key: False if key in blocked else is_key_down(key))

    gui.update_camera_from_keys = filtered
    return True


class RobotDriver:

    def __init__(self, sim, tm, rp: RobotParams):
        self.sim = sim
        self.rp = rp
        rb = tm.robot_data
        self.planar_q = np.asarray(rb["planar_q"], dtype=int)
        self.planar_dof = np.asarray(rb["planar_dof"], dtype=int)
        self.planar_tq = np.asarray(rb.get("planar_tq", rb["planar_q"]), dtype=int)
        self.wrist = np.asarray(rb["wrist"], dtype=int)
        self._target_host = sim.control.joint_target_qd.numpy()
        self._terrain = getattr(tm, "terrain_height", None)
        self._last = None
        self._fg_frame = -1


        self._tq_host = sim.control.joint_target_q.numpy()


        for env_tq in rb["arm_tq"]:
            self._tq_host[np.asarray(env_tq, dtype=int)] = rb["arm_home"]
        sim.control.joint_target_q.assign(self._tq_host)

    def update(self, viewer=None):
        fwd = trn = 0.0
        if viewer is not None and hasattr(viewer, "is_key_down"):
            gui = getattr(viewer, "gui", None)
            if gui is None or not getattr(gui, "is_capturing", lambda: False)():
                fwd = (1.0 if viewer.is_key_down("w") else 0.0) - \
                      (1.0 if viewer.is_key_down("s") else 0.0)
                trn = (1.0 if viewer.is_key_down("a") else 0.0) - \
                      (1.0 if viewer.is_key_down("d") else 0.0)
        v = fwd * self.rp.drive_speed
        w = trn * self.rp.turn_speed


        q = self.sim.state_0.joint_q.numpy()
        follow = self._terrain is not None
        if self._last == (v, w) and v == 0.0 and w == 0.0 and not follow:
            return
        self._last = (v, w)
        for qs, ds in zip(self.planar_q, self.planar_dof):
            yaw = float(q[qs + 3])
            self._target_host[ds + 0] = v * math.cos(yaw)
            self._target_host[ds + 1] = v * math.sin(yaw)
            self._target_host[ds + 3] = w
        self.sim.control.joint_target_qd.assign(self._target_host)
        if follow:
            self.follow_ground()

    def follow_ground(self, frame: int | None = None):
        if self._terrain is None:
            return
        if frame is not None:
            if frame == self._fg_frame:
                return
            self._fg_frame = frame
        bq = self.sim.state_0.body_q.numpy()
        rb = self.sim.tree.robot_data
        for tqs, ch in zip(self.planar_tq, rb["chassis"]):
            x, y = float(bq[ch, 0]), float(bq[ch, 1])
            self._tq_host[tqs + 2] = float(self._terrain(x, y))
        self.sim.control.joint_target_q.assign(self._tq_host)


class WristCamera:


    _LOCAL_POS = wp.vec3(0.11, 0.0, 0.03)

    def __init__(self, model, viewer, tm, rp: RobotParams, perceive: bool = True):
        from newton.sensors import SensorTiledCamera
        self.viewer = viewer
        self.model = model
        self.rp = rp
        self.wrist = np.asarray(tm.robot_data["wrist"], dtype=int)
        self.num_envs = max(int(tm.num_envs), 1)
        self.every = max(int(rp.camera_every), 1)
        self._frame = 0

        self.sensor = SensorTiledCamera(model=model)
        self.sensor.utils.create_default_light(enable_shadows=False)
        W, H = int(rp.camera_width), int(rp.camera_height)
        self.rays = self.sensor.utils.compute_pinhole_camera_rays(
            W, H, math.radians(float(rp.camera_fov)))
        self.depth = self.sensor.utils.create_depth_image_output(W, H, 1)
        self.color = self.sensor.utils.create_color_image_output(W, H, 1)
        n = self.num_envs
        self.depth_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=self.depth.device)


        self.percept = None
        self.n_detect = 1
        self.highlight = None
        ne = self.num_envs
        self.dets_env = [[] for _ in range(ne)]
        self.depth_env = [None] * ne
        self.cam_env = [None] * ne
        self.scene_env = [None] * ne
        self.centroid_env = [None] * ne


        self._self_bodies = [
            np.arange(int(tm.robot_data["chassis"][e]), int(self.wrist[e]) + 4)
            for e in range(ne)]
        if perceive:
            from .perception import FruitPerception
            self.percept = FruitPerception(W, H, float(rp.camera_fov))


    @property
    def detections(self):
        return self.dets_env[0]

    @property
    def last_depth(self):
        return self.depth_env[0]

    @property
    def last_cam(self):
        return self.cam_env[0]

    @property
    def last_scene(self):
        return self.scene_env[0]

    def _mark_world_point(self, rgba, world_pt, cam0):
        pos, R = cam0
        pc = R.T @ (np.asarray(world_pt, dtype=float) - pos)
        if pc[2] > -0.05:
            return
        f = self.percept.focal_px
        H, W = rgba.shape[:2]
        cx = int(round(0.5 * (W - 1) + f * (pc[0] / -pc[2])))
        cy = int(round(0.5 * (H - 1) - f * (pc[1] / -pc[2])))
        if not (0 <= cx < W and 0 <= cy < H):
            return
        r = 9
        y0, y1 = max(cy - r, 0), min(cy + r + 1, H)
        x0, x1 = max(cx - r, 0), min(cx + r + 1, W)
        col = (255, 140, 20)
        rgba[y0:y1, x0:min(x0 + 2, W), :3] = col
        rgba[y0:y1, max(x1 - 2, 0):x1, :3] = col
        rgba[y0:min(y0 + 2, H), x0:x1, :3] = col
        rgba[max(y1 - 2, 0):y1, x0:x1, :3] = col

    def update(self, state):
        self._frame += 1
        if (self._frame - 1) % self.every:
            return

        bq = state.body_q.numpy()
        tfs = []
        cams = []
        for wb in self.wrist:
            p, quat = bq[wb, :3], bq[wb, 3:]
            u, s = quat[:3], quat[3]
            rot = lambda v: v + 2.0 * np.cross(u, np.cross(u, v) + s * v)
            approach = rot(np.array([0.0, 0.0, 1.0]))
            pos = p + rot(np.array([self._LOCAL_POS[0], self._LOCAL_POS[1],
                                    self._LOCAL_POS[2]]))


            focus = p + rot(np.array([0.0, 0.0, 0.10])) + approach * 0.28
            fwdv = focus - pos
            fwdv = fwdv / (np.linalg.norm(fwdv) + 1e-9)
            up_ref = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(fwdv, up_ref))) > 0.98:
                up_ref = rot(np.array([0.0, -1.0, 0.0]))
            right = np.cross(fwdv, up_ref); right /= (np.linalg.norm(right) + 1e-9)
            upv = np.cross(right, fwdv)


            R = wp.mat33f(right[0], upv[0], -fwdv[0],
                          right[1], upv[1], -fwdv[1],
                          right[2], upv[2], -fwdv[2])
            cams.append((pos.copy(), np.column_stack([right, upv, -fwdv])))
            tfs.append(wp.transformf(wp.vec3f(*map(float, pos)), wp.quat_from_matrix(R)))
        cam = wp.array([tfs], dtype=wp.transformf)

        self.model.bvh_refit_shapes(state)
        from newton.sensors import SensorTiledCamera as _S
        clear = _S.ClearData(clear_depth=float(self.rp.camera_range))
        self.sensor.update(state, cam, self.rays,
                           color_image=self.color, depth_image=self.depth,
                           clear_data=clear)
        self.sensor.utils.to_rgba_from_depth(
            self.depth, depth_range=(0.0, float(self.rp.camera_range)),
            out_buffer=self.depth_rgba)


        if self.percept is not None:

            depth_all = self.depth.numpy()[:, 0]
            rng_far = float(self.rp.camera_range)
            for e in range(min(max(int(self.n_detect), 1), self.num_envs)):
                d = depth_all[e]
                came = cams[e]
                self.depth_env[e] = d
                self.cam_env[e] = came
                dets = self.percept.detect(d, came[0], came[1])


                selfp = bq[self._self_bodies[e], :3]


                H, W = d.shape
                dflat = d.ravel()
                valid = (np.isfinite(dflat) & (dflat > 0.05)
                         & (np.abs(dflat - rng_far) > 1e-4))
                idx = np.nonzero(valid)[0]
                dirs = self.percept.dirs.reshape(-1, 3)
                pwv = (came[0].astype(np.float32)
                       + (dirs[idx] @ came[1].T.astype(np.float32))
                       * dflat[idx, None].astype(np.float32))
                keep = pwv[:, 2] > 0.16
                sub = pwv[keep]
                sel = idx[keep]
                if len(sub):
                    sp32 = selfp.astype(np.float32)
                    d2 = ((sub[:, None, :] - sp32[None, :, :]) ** 2).sum(-1).min(1)
                    ok = d2 > 0.32 ** 2
                else:
                    ok = np.zeros(0, dtype=bool)
                scene = np.zeros(H * W, dtype=bool)
                scene[sel[ok]] = True
                self.scene_env[e] = scene.reshape(H, W)


                kept_pw = sub[ok]
                dk = dflat[sel[ok]]
                treem = (dk > 0.3) & (dk < 4.0)
                self.centroid_env[e] = (kept_pw[treem].mean(axis=0)
                                        if int(treem.sum()) >= 60 else None)
                self.dets_env[e] = [
                    det for det in dets
                    if np.linalg.norm(selfp - det.center_world, axis=1).min() > 0.20]
            try:
                dets0 = self.dets_env[0]
                if dets0:
                    n = len(dets0)
                    pts = wp.array([wp.vec3(*map(float, det.center_world))
                                    for det in dets0], dtype=wp.vec3)
                    radii = wp.array([float(det.radius) + 0.012
                                      for det in dets0], dtype=wp.float32)


                    cols = wp.array([wp.vec3(0.1, 1.0, 0.25)] * n, dtype=wp.vec3)
                    self.viewer.log_points("fruit detections", pts, radii=radii,
                                           colors=cols)
                else:
                    self.viewer.log_points("fruit detections", None, hidden=True)
            except Exception:
                pass


        try:
            rgba0 = self.depth_rgba.numpy()[0]
            if self.percept is not None:
                rgba0 = self.percept.draw_overlay(rgba0, self.dets_env[0],
                                                  self.depth_env[0])
                if self.highlight is not None:
                    self._mark_world_point(rgba0, self.highlight, cams[0])
            self.viewer.log_image("wrist depth", rgba0[None])
        except Exception:
            pass
