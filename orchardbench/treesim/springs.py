from __future__ import annotations

import numpy as np
import warp as wp

from .builder import TreeModel
from . import physics


@wp.kernel
def _spring_torque(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    parent_body: wp.array(dtype=wp.int32),
    child_body: wp.array(dtype=wp.int32),
    q_rest: wp.array(dtype=wp.quat),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    j = wp.tid()
    k = kp[j]
    if k <= 0.0:
        return
    pb = parent_body[j]
    cb = child_body[j]
    qp = wp.transform_get_rotation(body_q[pb])
    qc = wp.transform_get_rotation(body_q[cb])

    q_rel = wp.mul(wp.quat_inverse(qp), qc)
    dq = wp.mul(q_rel, wp.quat_inverse(q_rest[j]))
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    sh = wp.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2])
    angle = 2.0 * wp.atan2(sh, dq[3])
    axis = wp.vec3(0.0, 0.0, 0.0)
    if sh > 1.0e-7:
        axis = wp.vec3(dq[0], dq[1], dq[2]) / sh
    rotvec = axis * angle

    tau = wp.quat_rotate(qp, -k * rotvec)
    rel_w = wp.spatial_top(body_qd[cb]) - wp.spatial_top(body_qd[pb])
    tau = tau - kd[j] * rel_w

    w = wp.spatial_vector(tau, wp.vec3(0.0, 0.0, 0.0))
    wp.atomic_add(body_f, cb, w)
    wp.atomic_sub(body_f, pb, w)


@wp.kernel
def _measure_moment(
    body_q: wp.array(dtype=wp.transform),
    parent_body: wp.array(dtype=wp.int32),
    child_body: wp.array(dtype=wp.int32),
    q_rest: wp.array(dtype=wp.quat),
    kp: wp.array(dtype=wp.float32),
    moment: wp.array(dtype=wp.float32),
):
    j = wp.tid()
    pb = parent_body[j]
    cb = child_body[j]
    qp = wp.transform_get_rotation(body_q[pb])
    qc = wp.transform_get_rotation(body_q[cb])
    q_rel = wp.mul(wp.quat_inverse(qp), qc)
    dq = wp.mul(q_rel, wp.quat_inverse(q_rest[j]))
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])
    sh = wp.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2])
    angle = 2.0 * wp.atan2(sh, dq[3])
    moment[j] = kp[j] * angle


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


class MomentMeter:

    def __init__(self, parent_body, child_body, kp, rest_body_q, device):
        self.device = device
        bq = np.asarray(rest_body_q)
        parent_body = np.asarray(parent_body, dtype=np.int32)
        child_body = np.asarray(child_body, dtype=np.int32)
        self.n = len(parent_body)
        qrest = np.array([_qmul(_qconj(bq[p, 3:7]), bq[c, 3:7])
                          for p, c in zip(parent_body, child_body)],
                         dtype=np.float32).reshape(-1, 4)
        self.parent_body = wp.array(parent_body, device=device)
        self.child_body = wp.array(child_body, device=device)
        self.q_rest = wp.array(qrest, dtype=wp.quat, device=device)
        self.kp = wp.array(np.asarray(kp, dtype=np.float32), device=device)
        self._moment = wp.zeros(max(self.n, 1), dtype=wp.float32, device=device)

    def moments(self, state) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        wp.launch(_measure_moment, dim=self.n,
                  inputs=[state.body_q, self.parent_body, self.child_body,
                          self.q_rest, self.kp, self._moment],
                  device=self.device)
        return self._moment.numpy()[:self.n]


class SpringField:

    def __init__(self, tree: TreeModel, rest_body_q=None):
        self.tree = tree
        self.model = tree.model
        dev = self.model.device
        skel = tree.skeleton
        phys = tree.config.physics


        bq = None if rest_body_q is None else np.asarray(rest_body_q)


        seg_mass = {s.index: physics.segment_mass(s, phys.wood_density) for s in skel.segments}
        subtree_mass = {}
        for s in sorted(skel.segments, key=lambda s: s.depth, reverse=True):
            subtree_mass[s.index] = seg_mass[s.index] + sum(
                subtree_mass[c] for c in s.children)

        parents, children, qrest, kps, kds, mmax, inertia = [], [], [], [], [], [], []
        for seg in skel.segments:
            if seg.parent < 0:
                continue
            par = skel[seg.parent]
            pb = int(tree.seg_to_body[seg.parent])
            cb = int(tree.seg_to_body[seg.index])
            parents.append(pb)
            children.append(cb)
            if bq is not None:
                qrest.append(_qmul(_qconj(bq[pb, 3:7]), bq[cb, 3:7]))
            else:
                qrest.append(_qmul(_qconj(par.frame), seg.frame))
            kp, kd = physics.stiffness_for(seg, phys)
            kps.append(kp)
            kds.append(kd)
            mmax.append(physics.rupture_moment(seg, tree.config.breaking))


            L = max(seg.length, 1e-3)
            I = seg_mass[seg.index] * L * L / 3.0 + \
                (subtree_mass[seg.index] - seg_mass[seg.index]) * L * L
            inertia.append(max(I, 1e-9))

        self.n = len(parents)
        self.parent_body = wp.array(np.array(parents, dtype=np.int32), device=dev)
        self.child_body = wp.array(np.array(children, dtype=np.int32), device=dev)
        self.q_rest = wp.array(np.array(qrest, dtype=np.float32), dtype=wp.quat, device=dev)
        self.kp_phys = np.array(kps, dtype=np.float64)
        self.kd_phys = np.array(kds, dtype=np.float64)
        self.inertia = np.array(inertia, dtype=np.float64)
        self.kp = wp.array(self.kp_phys.astype(np.float32), device=dev)
        self.kd = wp.array(self.kd_phys.astype(np.float32), device=dev)
        self.m_max = np.array(mmax, dtype=np.float64)
        self._moment = wp.zeros(self.n, dtype=wp.float32, device=dev)

    def cap_for_dt(self, dt: float, safety: float = 0.5):
        kp_max = self.inertia * (2.0 * safety / dt) ** 2
        kp = np.minimum(self.kp_phys, kp_max)
        kd_max = safety * self.inertia / dt
        kd = np.minimum(np.maximum(self.kd_phys, 0.2 * np.sqrt(kp * self.inertia)), kd_max)
        self.kp.assign(kp.astype(np.float32))
        self.kd.assign(kd.astype(np.float32))
        return kp

    def apply(self, state):
        wp.launch(_spring_torque, dim=self.n,
                  inputs=[state.body_q, state.body_qd, self.parent_body,
                          self.child_body, self.q_rest, self.kp, self.kd,
                          state.body_f], device=self.model.device)

    def moments(self, state) -> np.ndarray:
        wp.launch(_measure_moment, dim=self.n,
                  inputs=[state.body_q, self.parent_body, self.child_body,
                          self.q_rest, self.kp, self._moment],
                  device=self.model.device)
        return self._moment.numpy()
