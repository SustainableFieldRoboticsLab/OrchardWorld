from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FoliageParams
from .skeleton import TreeSkeleton


@dataclass
class LeafPlacement:
    parent_seg: int
    attach: np.ndarray
    frame: np.ndarray
    length: float
    width: float


def _rodrigues(v, axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def leaf_mesh(length: float, width: float, fold: float = 0.55,
              curl: float = 0.30, droop: float = 0.35, nseg: int = 5):
    import newton
    ts = np.linspace(0.0, 1.0, nseg + 1)
    verts: list[tuple] = []
    rows: list[tuple] = []
    for t in ts:
        w = 0.5 * width * (np.sin(np.pi * min(t, 0.995) ** 0.8) ** 0.85 + 0.03)
        z = length * t
        y_rib = curl * length * t * t - droop * length * t * t * t
        y_edge = y_rib + fold * w
        i0 = len(verts)
        verts.append((-w, y_edge, z))
        verts.append((0.0, y_rib, z))
        verts.append((w, y_edge, z))
        rows.append((i0, i0 + 1, i0 + 2))
    idx: list[int] = []
    for r in range(nseg):
        l0, m0, r0 = rows[r]
        l1, m1, r1 = rows[r + 1]
        idx += [l0, m0, l1, m0, m1, l1,
                m0, r0, m1, r0, r1, m1]

    back = []
    for k in range(0, len(idx), 3):
        back += [idx[k], idx[k + 2], idx[k + 1]]
    return newton.Mesh(np.asarray(verts, dtype=np.float32),
                       np.asarray(idx + back, dtype=np.int32),
                       compute_inertia=False, is_solid=False)


LEAF_SIZE_CLASSES = (0.72, 1.0, 1.35)


def leaf_meshes(fp: FoliageParams):
    return [leaf_mesh(fp.leaf_length * s, fp.leaf_width * s)
            for s in LEAF_SIZE_CLASSES]


def _frame_quat(H, L, U):
    from .lsystem import _frame_to_quat
    return _frame_to_quat(H / np.linalg.norm(H), L / np.linalg.norm(L), U / np.linalg.norm(U))


def place_leaves(skel: TreeSkeleton, fp: FoliageParams,
                 seed: int = 0) -> list[LeafPlacement]:
    rng = np.random.default_rng(seed + 777)
    out: list[LeafPlacement] = []
    max_order = max(s.order for s in skel.segments)
    thr = min(fp.min_order_for_leaves, max_order)

    for seg in skel.segments:
        if seg.order < thr:
            continue

        if not (seg.is_terminal or seg.order >= thr):
            continue
        H = seg.direction

        ref = np.array([0.0, 0.0, 1.0]) if abs(H[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        L = np.cross(ref, H); L /= np.linalg.norm(L)
        U = np.cross(H, L)
        nleaf = fp.leaves_per_terminal if seg.is_terminal else max(1, fp.leaves_per_terminal // 2)
        for k in range(nleaf):

            t = (k + 1) / (nleaf + 1)
            base = seg.start + H * (t * seg.length)
            roll = np.deg2rad(137.5 * k + rng.uniform(0, 360))
            pitch = np.deg2rad(rng.uniform(45, 75))
            Lr = _rodrigues(L, H, roll)
            Ur = _rodrigues(U, H, roll)
            outdir = _rodrigues(H, Lr, pitch)
            Uo = _rodrigues(Ur, Lr, pitch)
            jitter = 1.0 + rng.normal(0, 0.15)
            out.append(LeafPlacement(
                parent_seg=seg.index,
                attach=base.copy(),
                frame=_frame_quat(outdir, Lr, Uo),
                length=fp.leaf_length * max(0.4, jitter),
                width=fp.leaf_width * max(0.4, jitter),
            ))
    return out
