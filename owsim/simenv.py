from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OB_ROOT = os.path.join(os.path.dirname(HERE), "orchardbench")
if OB_ROOT not in sys.path:
    sys.path.insert(0, OB_ROOT)

import warp as wp
import newton
import newton.viewer

from treesim.config import TreeConfig, preset, StiffnessModel
from treesim import builder as _builder
from treesim.sim import Sim
from treesim import robot as _robot


_FRANKA = os.environ.get("OW_FRANKA_URDF") or os.path.join(
    os.path.dirname(HERE), "assets", "franka_emika_panda", "urdf", "fr3_franka_hand.urdf")
if os.path.exists(_FRANKA):
    _robot.franka_urdf_path = lambda: _FRANKA


GLOBAL_HW = (72, 96)
CROP = 64
DEPTH_MAX = 2.5


class OWCamera(_robot.WristCamera):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.render_id = 0
        self.color_np = None
        self.depth_np = None
        self.cams = None

    def set_light(self, direction, shadows: bool = False):
        d = np.asarray(direction, dtype=float)
        d = d / (np.linalg.norm(d) + 1e-9)
        self.sensor.utils.create_default_light(
            enable_shadows=bool(shadows), direction=wp.vec3f(*map(float, d)))

    def update(self, state):
        will_render = ((self._frame) % self.every) == 0
        super().update(state)
        if not will_render:
            return
        c = self.color.numpy()[:, 0]
        self.color_np = c.view(np.uint8).reshape(c.shape + (4,))[..., :3]
        self.depth_np = self.depth.numpy()[:, 0]
        self.cams = list(self.cam_env)
        self.render_id += 1


def make_config(seed: int, foliage: float, apples: int, device: str) -> TreeConfig:
    cfg = TreeConfig(lsystem=preset("apple"))
    cfg.seed = int(seed)
    cfg.device = device
    cfg.deformable = True
    cfg.physics.model = StiffnessModel("beam")
    cfg.breaking.enabled = True
    cfg.breaking.mode = "detach"
    if foliage > 0:
        cfg.foliage.set_density(float(foliage))
    cfg.fruit.enabled = True
    cfg.fruit.max_count = int(apples)
    cfg.physics.mj_solver = "cg"
    cfg.robot.enabled = True
    cfg.robot.camera = True
    return cfg


class Batch:

    def __init__(self, seed: int, num_envs: int, foliage: float, apples: int,
                 light_dir=(-0.4, 0.3, -1.0), shadows: bool = False,
                 device: str = "cuda", substeps: int = 3):
        self.seed = int(seed)
        self.num_envs = int(num_envs)
        self.foliage = float(foliage)
        self.apples_cfg = int(apples)
        self.light_dir = tuple(float(x) for x in light_dir)
        self.shadows = bool(shadows)
        self.cfg = make_config(seed, foliage, apples, device)
        self.tm = _builder.generate_and_build(
            self.cfg, max_bodies=60000, num_envs=self.num_envs,
            randomize_envs=self.num_envs > 1, dr_strength=1.0)
        self.sim = Sim(self.tm, solver="mujoco", fps=60, substeps=substeps,
                       enable_breaking=True, collisions=True)
        self.viewer = newton.viewer.ViewerNull(num_frames=10 ** 9)
        self.sim.set_viewer(self.viewer)
        self.driver = _robot.RobotDriver(self.sim, self.tm, self.cfg.robot)
        self.cam = OWCamera(self.tm.model, self.viewer, self.tm, self.cfg.robot)
        self.cam.n_detect = self.num_envs
        self.cam.set_light(self.light_dir, self.shadows)
        self.frame = 0
        self.contact_tol = 0.002
        self.last_contacts = None
        self.pull_np = None
        self._index_maps()


    def _index_maps(self):
        tm, model = self.tm, self.tm.model
        ne = self.num_envs
        nb = model.body_count
        self.bpe = nb // ne
        rb = tm.robot_data
        self.chassis = np.asarray(rb["chassis"], dtype=int)
        self.wrist = np.asarray(rb["wrist"], dtype=int)
        body_kind = np.zeros(nb, dtype=np.int8)
        apple_b = np.asarray(tm.apple_bodies, dtype=int)
        body_kind[apple_b] = 1
        for e in range(ne):
            body_kind[self.chassis[e]] = 2
            body_kind[self.chassis[e] + 1:self.wrist[e] + 4] = 3
        for lb in tm.leaf_bodies:
            body_kind[int(lb)] = 4
        self.body_kind = body_kind
        self.body_env = np.minimum(np.arange(nb) // max(self.bpe, 1), ne - 1)
        sb = model.shape_body.numpy()
        self.shape_body = sb
        self.shape_kind = np.where(sb >= 0, body_kind[np.maximum(sb, 0)], -1)
        self.shape_env = np.where(sb >= 0, self.body_env[np.maximum(sb, 0)], -1)
        self.wood_bodies = [np.nonzero((body_kind == 0) & (self.body_env == e))[0]
                            for e in range(ne)]
        n_ap = len(apple_b) // ne
        self.n_env_apples = n_ap
        self.apple_bodies = apple_b
        self.n_brk_env = len(tm.brk_joint_id) // ne if tm.brk_joint_id is not None else 0

        nj = model.joint_count
        jq_s = model.joint_q_start.numpy()[:nj]
        jqd_s = model.joint_qd_start.numpy()[:nj]
        qd2q = {int(d): int(q) for q, d in zip(jq_s, jqd_s)}
        self.arm_dofs = [np.asarray(x, dtype=int) for x in rb["arm_dofs"]]
        self.arm_q = [np.asarray([qd2q.get(int(d), int(d)) for d in x], dtype=int)
                      for x in rb["arm_dofs"]]
        self.planar_q = np.asarray(rb["planar_q"], dtype=int)
        self.planar_dof = np.asarray(rb["planar_dof"], dtype=int)


    def step(self):
        self.sim.step()
        self.frame += 1

    def probe(self, need_contacts: bool, need_pull: bool):
        self.last_contacts = self.contact_pairs() if need_contacts else None
        self.pull_np = self.sim.apples.pull_forces() if need_pull else None

    def body_q(self):
        return self.sim.body_q_np()

    def finite(self) -> bool:
        bq = self.body_q()
        return bool(np.isfinite(bq).all())


    def contact_pairs(self):
        ne = self.num_envs
        aw = np.zeros(ne, dtype=int)
        bw = np.zeros(ne, dtype=int)
        aa = np.zeros(ne, dtype=int)
        c = self.sim.contacts
        try:
            n = int(c.rigid_contact_count.numpy()[0])
        except Exception:
            return aw, bw, aa
        if n <= 0:
            return aw, bw, aa
        n = min(n, c.rigid_contact_shape0.shape[0])
        s0 = c.rigid_contact_shape0.numpy()[:n]
        s1 = c.rigid_contact_shape1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)

        kk0 = np.where(s0 >= 0, self.shape_kind[np.maximum(s0, 0)], -1)
        kk1 = np.where(s1 >= 0, self.shape_kind[np.maximum(s1, 0)], -1)
        ok &= (kk0 >= 2) | (kk1 >= 2)
        if not ok.any():
            return aw, bw, aa


        idx = np.nonzero(ok)[0]
        s0, s1 = s0[idx], s1[idx]
        bq = self.body_q()
        p0 = _to_world(bq, self.shape_body[s0], c.rigid_contact_point0.numpy()[idx])
        p1 = _to_world(bq, self.shape_body[s1], c.rigid_contact_point1.numpy()[idx])
        nrm = c.rigid_contact_normal.numpy()[idx]
        thick = c.rigid_contact_margin0.numpy()[idx] + c.rigid_contact_margin1.numpy()[idx]
        dist = np.einsum("ij,ij->i", nrm, p1 - p0) - thick
        touch = dist <= self.contact_tol
        s0, s1 = s0[touch], s1[touch]
        k0, k1 = self.shape_kind[s0], self.shape_kind[s1]
        env = np.maximum(self.shape_env[s0], self.shape_env[s1])
        arm_wood = ((k0 == 3) & (k1 == 0)) | ((k0 == 0) & (k1 == 3))
        base_wood = ((k0 == 2) & (k1 == 0)) | ((k0 == 0) & (k1 == 2))
        arm_apple = ((k0 == 3) & (k1 == 1)) | ((k0 == 1) & (k1 == 3))
        for mask, out in ((arm_wood, aw), (base_wood, bw), (arm_apple, aa)):
            if mask.any():
                np.add.at(out, env[mask], 1)
        return aw, bw, aa

    def broken_env(self, e: int) -> int:
        br = self.sim.breaker
        if br is None or self.n_brk_env == 0:
            return 0
        return int(br.broken[e * self.n_brk_env:(e + 1) * self.n_brk_env].sum())

    def detached_env(self, e: int) -> np.ndarray:
        n = self.n_env_apples
        return self.sim.apples.detached[e * n:(e + 1) * n].copy()


def _to_world(bq: np.ndarray, bodies: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = pts.astype(np.float64).copy()
    m = bodies >= 0
    if m.any():
        X = bq[bodies[m]]
        p, q = X[:, :3], X[:, 3:]
        u, w = q[:, :3], q[:, 3:4]
        v = pts[m].astype(np.float64)
        t = 2.0 * np.cross(u, v)
        out[m] = p + v + w * t + np.cross(u, t)
    return out


def random_light(rng: np.random.Generator, altered: bool = False):
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(8, 18) if altered else rng.uniform(35, 80))
    d = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
    return d, bool(altered)


def downsample2(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    h2, w2 = H // 2, W // 2
    x = img[:2 * h2, :2 * w2].astype(np.float32)
    x = x.reshape(h2, 2, w2, 2, *img.shape[2:]).mean(axis=(1, 3))
    return x


def crop_at(img: np.ndarray, cy: float, cx: float, size: int, fill) -> np.ndarray:
    H, W = img.shape[:2]
    out = np.empty((size, size) + img.shape[2:], dtype=img.dtype)
    out[...] = fill
    y0 = int(round(cy)) - size // 2
    x0 = int(round(cx)) - size // 2
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1, xs1 = min(y0 + size, H), min(x0 + size, W)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return out
