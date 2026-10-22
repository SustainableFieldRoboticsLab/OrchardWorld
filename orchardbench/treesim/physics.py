from __future__ import annotations

import math

import numpy as np

from .config import PhysicsParams, BreakParams, StiffnessModel
from .skeleton import Segment


def stiffness_for(seg: Segment, phys: PhysicsParams,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    r = max(seg.radius_start, 1e-4)
    l = max(seg.length, 1e-3)

    if phys.model == StiffnessModel.BEAM:
        kp = phys.beam_factor * phys.youngs_modulus * (r ** 4) / l
    elif phys.model == StiffnessModel.RUDIMENTARY:
        kp = phys.phi_upper * (phys.phi_decay ** seg.order)
        kp = max(kp, phys.phi_lower)
    else:
        kp = phys.max_stiffness

    kd = phys.damping_ratio * kp


    if rng is not None and phys.dynamics_jitter > 0.0:
        kp *= max(0.05, 1.0 + rng.normal(0.0, phys.dynamics_jitter))
        kd = phys.damping_ratio * kp

    kp = float(np.clip(kp, phys.min_stiffness, phys.max_stiffness))
    kd = float(max(kd, 0.0))
    return kp, kd


def rupture_moment(seg: Segment, brk: BreakParams) -> float:
    r = max(seg.radius_start, 1e-4)
    return brk.safety_factor * brk.rupture_stress * math.pi * (r ** 3) / 4.0


def segment_mass(seg: Segment, density: float) -> float:
    r = seg.mean_radius
    l = seg.length
    cyl = math.pi * r * r * l
    caps = (4.0 / 3.0) * math.pi * r ** 3
    return density * (cyl + caps)


def summarize(skel, phys: PhysicsParams) -> str:
    kps = [stiffness_for(s, phys)[0] for s in skel.segments if s.parent >= 0]
    masses = [segment_mass(s, phys.wood_density) for s in skel.segments]
    if not kps:
        return "no compliant joints"
    return (f"stiffness Kp range {min(kps):.3g}..{max(kps):.3g} N*m/rad | "
            f"total mass {sum(masses):.2f} kg | "
            f"model={phys.model.value}")
