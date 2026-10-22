from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:

    index: int
    parent: int
    start: np.ndarray
    end: np.ndarray
    radius_start: float
    radius_end: float
    order: int
    depth: int
    children: list[int] = field(default_factory=list)


    frame: "np.ndarray | None" = None

    @property
    def axis(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        v = self.axis
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def mean_radius(self) -> float:
        return 0.5 * (self.radius_start + self.radius_end)

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0


class TreeSkeleton:

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._reindex()


    def _reindex(self) -> None:
        for s in self.segments:
            s.children.clear()
        by_index = {s.index: s for s in self.segments}
        for s in self.segments:
            if s.parent >= 0 and s.parent in by_index:
                by_index[s.parent].children.append(s.index)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, i: int) -> Segment:
        return self.segments[i]

    @property
    def roots(self) -> list[Segment]:
        return [s for s in self.segments if s.parent < 0]

    @property
    def terminals(self) -> list[Segment]:
        return [s for s in self.segments if s.is_terminal]

    def descendants(self, index: int) -> list[int]:
        out: list[int] = []
        stack = list(self.segments[index].children)
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.segments[i].children)
        return out


    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.array([s.start for s in self.segments] +
                       [s.end for s in self.segments])
        return pts.min(axis=0), pts.max(axis=0)

    def height(self) -> float:
        lo, hi = self.bounds()
        return float(hi[2] - lo[2])

    def rescale(self, target_height: float, base_radius: float) -> None:
        lo, hi = self.bounds()
        h = hi[2] - lo[2]
        if h < 1e-9:
            return
        s = target_height / h

        cur_base = max((seg.radius_start for seg in self.segments), default=base_radius)


        radius_scale = base_radius / cur_base if cur_base > 1e-9 else s
        anchor = self.roots[0].start.copy() if self.roots else lo
        for seg in self.segments:
            seg.start = (seg.start - anchor) * s
            seg.end = (seg.end - anchor) * s
            seg.radius_start *= radius_scale
            seg.radius_end *= radius_scale

    def lift_above_ground(self, clearance: float = 0.10) -> int:
        def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array([aw*bx + ax*bw + ay*bz - az*by,
                             aw*by - ax*bz + ay*bw + az*bx,
                             aw*bz + ax*by - ay*bx + az*bw,
                             aw*bw - ax*bx - ay*by - az*bz])

        def qconj(q):
            return np.array([-q[0], -q[1], -q[2], q[3]])

        def qrot(q, v):
            u, w = q[:3], q[3]
            return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

        def axis_angle(axis, ang):
            n = np.linalg.norm(axis)
            if n < 1e-12 or ang == 0.0:
                return np.array([0.0, 0.0, 0.0, 1.0])
            h = 0.5 * ang
            return np.array([*(axis / n * np.sin(h)), np.cos(h)])

        Z = np.array([0.0, 0.0, 1.0])
        old_frame = [np.asarray(s.frame, float) for s in self.segments]
        new_frame = [None] * len(self.segments)
        new_end = [None] * len(self.segments)
        lifted = 0
        for i, s in enumerate(self.segments):
            L = max(s.length, 1e-9)
            if s.parent < 0:
                start = s.start
                f = old_frame[i]
            else:
                start = new_end[s.parent]
                q_rel = qmul(qconj(old_frame[s.parent]), old_frame[i])
                f = qmul(new_frame[s.parent], q_rel)
            h = qrot(f, Z)
            need_hz = (clearance - start[2]) / L
            if h[2] < need_hz and need_hz <= 1.0:
                ang = float(np.arcsin(np.clip(need_hz, -1.0, 1.0)) -
                            np.arcsin(np.clip(h[2], -1.0, 1.0)))
                f = qmul(axis_angle(np.cross(h, Z), ang), f)
                f /= np.linalg.norm(f)
                h = qrot(f, Z)
                lifted += 1
            new_frame[i] = f
            new_end[i] = start + h * L
            s.start = start
            s.end = new_end[i]
            s.frame = f
        return lifted

    def total_length(self) -> float:
        return float(sum(s.length for s in self.segments))

    def summary(self) -> str:
        lo, hi = self.bounds()
        orders = [s.order for s in self.segments]
        radii = [s.mean_radius for s in self.segments]
        return (
            f"TreeSkeleton: {len(self.segments)} segments | "
            f"height={self.height():.3f} m | "
            f"orders 0..{max(orders)} | "
            f"radius {min(radii)*1e3:.2f}..{max(radii)*1e3:.2f} mm | "
            f"total wood length={self.total_length():.2f} m | "
            f"bbox=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}).."
            f"({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})"
        )
