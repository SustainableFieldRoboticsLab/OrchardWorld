from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .config import FruitParams
from .skeleton import TreeSkeleton


@dataclass
class ApplePlacement:
    parent_seg: int
    attach: np.ndarray
    radius: float
    color: tuple


def place_apples(skel: TreeSkeleton, fp: FruitParams, seed: int = 0) -> list[ApplePlacement]:
    rng = np.random.default_rng(seed + 4242)
    max_order = max(s.order for s in skel.segments)
    thr = min(fp.min_order, max_order)


    lo, hi = skel.bounds()
    span = max(hi - lo)
    eligible = [s for s in skel.segments
                if s.order >= thr and s.radius_start < 0.02]
    rng.shuffle(eligible)

    out: list[ApplePlacement] = []
    for s in eligible:
        if len(out) >= fp.max_count:
            break

        rad_xy = float(np.hypot(s.midpoint[0] - (lo[0] + hi[0]) / 2,
                                s.midpoint[1] - (lo[1] + hi[1]) / 2))
        outerness = np.clip(rad_xy / (0.5 * span + 1e-6), 0.0, 1.0)
        if rng.random() > fp.prob_per_spur * (0.4 + 0.6 * outerness):
            continue

        t = rng.uniform(0.5, 1.0)
        attach = s.start + s.direction * (t * s.length)
        col = fp.colors[rng.integers(0, len(fp.colors))]

        col = tuple(float(np.clip(c + rng.normal(0, 0.04), 0, 1)) for c in col)


        r = float(rng.uniform(*fp.radius))
        r = float(np.clip(round(r / 0.002) * 0.002, fp.radius[0], fp.radius[1]))
        out.append(ApplePlacement(parent_seg=s.index, attach=attach, radius=r, color=col))
    return out


@wp.kernel
def _apple_tether(body_q: wp.array(dtype=wp.transform),
                  body_qd: wp.array(dtype=wp.spatial_vector),
                  apple_body: wp.array(dtype=wp.int32),
                  parent_body: wp.array(dtype=wp.int32),
                  offset: wp.array(dtype=wp.vec3),
                  hang_drop: wp.array(dtype=wp.float32),
                  k: float, c: float,
                  body_mass: wp.array(dtype=wp.float32),
                  body_com: wp.array(dtype=wp.vec3),
                  react_scale: float, react_max: float, grav: float,
                  detached: wp.array(dtype=wp.int32),
                  held: wp.array(dtype=wp.int32),
                  hold_body: wp.array(dtype=wp.int32),
                  hold_off: wp.vec3, hold_k: float, hold_c: float,
                  pull: wp.array(dtype=wp.float32),
                  tension: wp.array(dtype=wp.float32),
                  body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    ab = apple_body[i]


    entry = wp.spatial_top(body_f[ab])
    fh = wp.vec3(0.0, 0.0, 0.0)
    if held[i] != 0:
        hp = wp.transform_point(body_q[hold_body[i]], hold_off)
        p = wp.transform_get_translation(body_q[ab])
        v = wp.spatial_top(body_qd[ab])
        kh = hold_k
        if detached[i] != 0:


            kh = 2.5 * hold_k
        fh = -kh * (p - hp) - hold_c * v
        body_f[ab] = body_f[ab] + wp.spatial_vector(fh, wp.vec3(0.0, 0.0, 0.0))
    pull[i] = wp.length(entry + fh)
    if detached[i] != 0:


        return
    pb = parent_body[i]
    attach = wp.transform_point(body_q[pb], offset[i])
    target = attach - wp.vec3(0.0, 0.0, hang_drop[i])
    p = wp.transform_get_translation(body_q[ab])
    v = wp.spatial_top(body_qd[ab])
    fs = -k * (p - target)
    tension[i] = wp.length(fs)
    f = fs - c * v

    body_f[ab] = body_f[ab] + wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))

    if react_scale > 0.0:


        fr = -(fs - wp.vec3(0.0, 0.0, body_mass[ab] * grav)) * react_scale
        mag = wp.length(fr)
        if mag > react_max:
            fr = fr * (react_max / mag)

        com = wp.transform_point(body_q[pb], body_com[pb])

        wp.atomic_add(body_f, pb, wp.spatial_vector(fr, wp.cross(attach - com, fr)))


class AppleField:

    def __init__(self, tree, config=None):
        self.tree = tree
        self.model = tree.model
        dev = self.model.device
        d = tree.apple_data
        fr = (config or tree.config).fruit
        self.n = len(d["apple_body"])
        self.apple_body = wp.array(np.asarray(d["apple_body"], dtype=np.int32), device=dev)
        self.parent_body = wp.array(np.asarray(d["parent_body"], dtype=np.int32), device=dev)
        self.offset = wp.array(np.asarray(d["offset"], dtype=np.float32), dtype=wp.vec3, device=dev)
        self.hang_drop = wp.array(np.asarray(d["hang_drop"], dtype=np.float32), device=dev)
        self.detach_force = np.asarray(d["detach_force"], dtype=np.float64)

        self.k = float(fr.tether_stiffness)

        self.c = float(fr.tether_damping_ratio) * 2.0 * float(np.sqrt(self.k * fr.mass))
        self._hyst = max(int(getattr(fr, "detach_hysteresis", 1)), 1)
        self.detach_drag = float(getattr(fr, "detach_drag", 120.0))
        self.react_scale = float(getattr(fr, "branch_reaction", 0.0))
        self.react_max = float(getattr(fr, "reaction_max", 10.0))
        self.grav = abs(float((config or tree.config).physics.gravity))
        self.body_mass = self.model.body_mass
        self.body_com = self.model.body_com

        self.detached = np.zeros(self.n, dtype=bool)
        self._over = np.zeros(self.n, dtype=np.int32)
        self.broken_count = 0
        self._flag = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._flag_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self._pull = wp.zeros(max(self.n, 1), dtype=wp.float32, device=dev)
        self._tension = wp.zeros(max(self.n, 1), dtype=wp.float32, device=dev)
        self._over_t = np.zeros(self.n, dtype=np.int32)
        self._hyst_t = max(int(getattr(fr, "tension_hysteresis", 12)), 1)
        self._tension_factor = float(getattr(fr, "tension_factor", 1.2))


        self._held = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._held_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self._hold_body = wp.zeros(max(self.n, 1), dtype=wp.int32, device=dev)
        self._hold_body_host = np.zeros(max(self.n, 1), dtype=np.int32)
        self.hold_off = wp.vec3(0.0, 0.0, 0.10)


        self.hold_k = 500.0
        self.hold_c = 2.0 * float(np.sqrt(self.hold_k * fr.mass))
        self.dev = dev

    def hold(self, i: int, hand_body: int):
        self._held_host[int(i)] = 1
        self._hold_body_host[int(i)] = int(hand_body)
        self._held.assign(self._held_host)
        self._hold_body.assign(self._hold_body_host)

    def release(self, i: int):
        self._held_host[int(i)] = 0
        self._held.assign(self._held_host)

    def pull_forces(self) -> np.ndarray:
        return self._pull.numpy()[:self.n]

    def apply(self, state):
        if self.n == 0:
            return
        wp.launch(_apple_tether, dim=self.n,
                  inputs=[state.body_q, state.body_qd, self.apple_body, self.parent_body,
                          self.offset, self.hang_drop, self.k, self.c,
                          self.body_mass, self.body_com,
                          self.react_scale, self.react_max, self.grav,
                          self._flag, self._held, self._hold_body,
                          self.hold_off, self.hold_k, self.hold_c,
                          self._pull, self._tension, state.body_f],
                  device=self.dev)

    def update(self, state) -> int:
        if self.n == 0 or self.detached.all():
            return 0
        pull = self._pull.numpy()[:self.n]
        over = (pull > self.detach_force) & (~self.detached)
        self._over[over] += 1
        self._over[~over] = 0
        tension = self._tension.numpy()[:self.n]
        over_t = (tension > self._tension_factor * self.detach_force) & (~self.detached)
        self._over_t[over_t] += 1
        self._over_t[~over_t] = 0
        newly = ((self._over >= self._hyst) | (self._over_t >= self._hyst_t)) \
            & (~self.detached)
        if not newly.any():
            return 0
        self.detached[newly] = True
        self._flag_host[:self.n][newly] = 1
        self._flag.assign(self._flag_host)
        self.broken_count = int(self.detached.sum())
        return int(newly.sum())
