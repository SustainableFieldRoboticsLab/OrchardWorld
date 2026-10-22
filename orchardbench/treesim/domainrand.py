from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .skeleton import TreeSkeleton
from .config import TreeConfig
from .fruit import ApplePlacement
from .foliage import LeafPlacement


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q, v):
    u = np.array([q[0], q[1], q[2]])
    w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _axis_angle_quat(axis, angle):
    n = np.linalg.norm(axis)
    if n < 1e-12 or angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    h = 0.5 * angle
    s = np.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(h)])


_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class DRParams:
    strength: float = 1.0


    randomize_geometry: bool = False


    scale_lo: float = 0.86
    scale_hi: float = 1.16
    rad_bias_lo: float = 0.90
    rad_bias_hi: float = 1.16


    len_sigma: float = 0.10
    len_lo: float = 0.78
    len_hi: float = 1.26
    ang_sigma_deg: float = 5.0
    rad_sigma: float = 0.09
    rad_lo: float = 0.76
    rad_hi: float = 1.30


    droop_lo: float = 0.55
    droop_hi: float = 1.60
    droop_base_deg: float = 1.3
    lean_max_deg: float = 5.0
    spread_lo: float = 0.85
    spread_hi: float = 1.18
    twist_max_deg: float = 22.0


    wood_bright_lo: float = 0.80
    wood_bright_hi: float = 1.22
    wood_warm_lo: float = 0.90
    wood_warm_hi: float = 1.14
    leaf_bright_lo: float = 0.78
    leaf_bright_hi: float = 1.28
    leaf_yellow_lo: float = 0.85
    leaf_yellow_hi: float = 1.40
    apple_hue_sigma: float = 0.06


    dens_lo: float = 0.86
    dens_hi: float = 1.16
    mod_lo: float = 0.78
    mod_hi: float = 1.30
    rupt_lo: float = 0.82
    rupt_hi: float = 1.25


    apple_size_lo: float = 0.82
    apple_size_hi: float = 1.22
    apple_mass_lo: float = 0.80
    apple_mass_hi: float = 1.25
    apple_visible_lo: float = 0.55


    apple_hidden_radius: float = 0.0012


    leaf_lo: float = 0.82
    leaf_hi: float = 1.22


    def _u(self, rng, lo, hi):
        s = self.strength
        lo = 1.0 - (1.0 - lo) * s
        hi = 1.0 + (hi - 1.0) * s
        return float(rng.uniform(lo, hi))

    def _nclip(self, rng, sigma, lo, hi):
        s = self.strength
        v = 1.0 + rng.normal(0.0, sigma * s)
        lo = 1.0 - (1.0 - lo) * s
        hi = 1.0 + (hi - 1.0) * s
        return float(np.clip(v, lo, hi))


def env_rng(base_seed: int, env_index: int) -> np.random.Generator:
    return np.random.default_rng([int(base_seed) & 0x7FFFFFFF, int(env_index)])


def dim_rng(base_seed: int) -> np.random.Generator:
    return np.random.default_rng([(int(base_seed) ^ 0x9E3779B1) & 0x7FFFFFFF, 0])


def perturb_skeleton(base: TreeSkeleton, rng, dr: DRParams, dim_rng=None) -> TreeSkeleton:
    skel = copy.deepcopy(base)
    segs = skel.segments
    n = len(segs)

    b_start = [np.asarray(s.start, float) for s in base.segments]
    b_frame = [_qnorm(np.asarray(s.frame, float)) for s in base.segments]
    b_len = [max(float(s.length), 1e-6) for s in base.segments]
    b_r0 = [float(s.radius_start) for s in base.segments]
    b_r1 = [float(s.radius_end) for s in base.segments]


    drd = rng if dim_rng is None else dim_rng
    scale = dr._u(drd, dr.scale_lo, dr.scale_hi)
    rad_bias = dr._u(drd, dr.rad_bias_lo, dr.rad_bias_hi)
    ang_sigma = np.deg2rad(dr.ang_sigma_deg) * dr.strength


    droop = dr._u(rng, dr.droop_lo, dr.droop_hi)
    droop_gain = (droop - 1.0) * np.deg2rad(dr.droop_base_deg)
    spread = dr._u(rng, dr.spread_lo, dr.spread_hi)
    twist = np.deg2rad(rng.uniform(-dr.twist_max_deg, dr.twist_max_deg)) * dr.strength
    lean_ang = np.deg2rad(rng.uniform(0.0, dr.lean_max_deg)) * dr.strength
    lean_az = rng.uniform(0.0, 2.0 * np.pi)
    q_lean = _axis_angle_quat(np.array([np.cos(lean_az), np.sin(lean_az), 0.0]), lean_ang)

    new_start = [None] * n
    new_end = [None] * n
    new_frame = [None] * n

    for i, s in enumerate(segs):
        f_len = dr._nclip(drd, dr.len_sigma, dr.len_lo, dr.len_hi)
        f_rad = dr._nclip(drd, dr.rad_sigma, dr.rad_lo, dr.rad_hi)
        length = b_len[i] * f_len * scale


        if ang_sigma > 0.0:
            ang = float(rng.normal(0.0, ang_sigma))
            jq = _axis_angle_quat(rng.normal(size=3), ang)
        else:
            jq = np.array([0.0, 0.0, 0.0, 1.0])

        p = s.parent
        if p < 0:

            new_start[i] = b_start[i] * scale
            new_frame[i] = _qnorm(_qmul(q_lean, b_frame[i]))
        else:
            q_rel = _qmul(_qconj(b_frame[p]), b_frame[i])
            fork = segs[p].order != s.order
            if fork:


                if spread != 1.0:
                    d = _qrot(q_rel, _Z)
                    tilt = float(np.arccos(np.clip(d[2], -1.0, 1.0)))
                    axis = np.cross(d, _Z)
                    q_rel = _qmul(_axis_angle_quat(axis, (1.0 - spread) * tilt), q_rel)
                if twist != 0.0:
                    q_rel = _qmul(_axis_angle_quat(_Z, twist), q_rel)
            q_rel = _qmul(jq, q_rel)
            new_frame[i] = _qnorm(_qmul(new_frame[p], q_rel))


            if droop_gain != 0.0 and s.order > 0:
                h = _qrot(new_frame[i], _Z)
                horiz = float(np.hypot(h[0], h[1]))
                if horiz > 1e-6:
                    axis = np.cross(h, np.array([0.0, 0.0, -1.0]))
                    new_frame[i] = _qnorm(_qmul(
                        _axis_angle_quat(axis, droop_gain * horiz), new_frame[i]))
            new_start[i] = new_end[p]

        heading = _qrot(new_frame[i], _Z)
        new_end[i] = new_start[i] + heading * length

        s.start = new_start[i]
        s.end = new_end[i]
        s.frame = new_frame[i]
        s.radius_start = b_r0[i] * f_rad * rad_bias * scale
        s.radius_end = b_r1[i] * f_rad * rad_bias * scale

    skel._reindex()


    skel.lift_above_ground(0.10)
    return skel


def perturb_config(base_cfg: TreeConfig, rng, dr: DRParams, env_index: int) -> TreeConfig:
    c = copy.deepcopy(base_cfg)


    c.seed = int(base_cfg.seed) + 1 + int(env_index)

    c.physics.wood_density = float(np.clip(
        base_cfg.physics.wood_density * dr._u(rng, dr.dens_lo, dr.dens_hi), 500.0, 1000.0))
    c.physics.youngs_modulus = float(
        base_cfg.physics.youngs_modulus * dr._u(rng, dr.mod_lo, dr.mod_hi))
    c.breaking.rupture_stress = float(
        base_cfg.breaking.rupture_stress * dr._u(rng, dr.rupt_lo, dr.rupt_hi))
    c.fruit.mass = float(np.clip(
        base_cfg.fruit.mass * dr._u(rng, dr.apple_mass_lo, dr.apple_mass_hi), 0.09, 0.26))

    leaf = dr._u(rng, dr.leaf_lo, dr.leaf_hi)
    c.foliage.leaf_length = base_cfg.foliage.leaf_length * leaf
    c.foliage.leaf_width = base_cfg.foliage.leaf_width * leaf


    wb = dr._u(rng, dr.wood_bright_lo, dr.wood_bright_hi)
    ww = dr._u(rng, dr.wood_warm_lo, dr.wood_warm_hi)
    c.render.wood_tint = (wb * ww, wb, wb / max(ww, 1e-3))
    lb = dr._u(rng, dr.leaf_bright_lo, dr.leaf_bright_hi)
    ly = dr._u(rng, dr.leaf_yellow_lo, dr.leaf_yellow_hi)
    r0, g0, b0 = base_cfg.foliage.leaf_color
    c.foliage.leaf_color = (float(np.clip(r0 * lb * ly, 0.0, 1.0)),
                            float(np.clip(g0 * lb, 0.0, 1.0)),
                            float(np.clip(b0 * lb / max(ly, 1e-3), 0.0, 1.0)))
    return c


def _fraction_along(seg, attach) -> float:
    t = float(np.dot(np.asarray(attach) - seg.start, seg.direction) / max(seg.length, 1e-6))
    return float(np.clip(t, 0.0, 1.0))


def remap_apples(canonical, base: TreeSkeleton, skel_e: TreeSkeleton,
                 rng, dr: DRParams):
    m = len(canonical)
    if m == 0:
        return []
    n_vis = int(round(dr._u(rng, dr.apple_visible_lo, 1.0) * m))
    n_vis = int(np.clip(n_vis, 1, m))
    visible = set(rng.choice(m, size=n_vis, replace=False).tolist())

    hue = rng.normal(0.0, dr.apple_hue_sigma * dr.strength, size=3)

    out = []
    for j, ap in enumerate(canonical):
        ps = ap.parent_seg
        t = _fraction_along(base[ps], ap.attach)
        eseg = skel_e[ps]
        attach_e = eseg.start + eseg.direction * (t * eseg.length)
        if j in visible:


            r = float(np.clip(ap.radius * dr._u(rng, dr.apple_size_lo, dr.apple_size_hi),
                              0.024, 0.034))


            r = float(np.clip(round(r / 0.002) * 0.002, 0.024, 0.034))
        else:
            r = float(dr.apple_hidden_radius)
        col = tuple(float(np.clip(c + h + rng.normal(0.0, 0.03), 0.0, 1.0))
                    for c, h in zip(ap.color, hue))
        out.append(ApplePlacement(parent_seg=ps, attach=attach_e, radius=r, color=col))
    return out


def remap_leaves(canonical, base: TreeSkeleton, skel_e: TreeSkeleton):
    out = []
    for lp in canonical:
        ps = lp.parent_seg
        bseg = base[ps]
        eseg = skel_e[ps]
        t = _fraction_along(bseg, lp.attach)
        attach_e = eseg.start + eseg.direction * (t * eseg.length)
        local_q = _qmul(_qconj(_qnorm(np.asarray(bseg.frame, float))),
                        _qnorm(np.asarray(lp.frame, float)))
        frame_e = _qnorm(_qmul(_qnorm(np.asarray(eseg.frame, float)), local_q))
        out.append(LeafPlacement(parent_seg=ps, attach=attach_e, frame=frame_e,
                                 length=lp.length, width=lp.width))
    return out
