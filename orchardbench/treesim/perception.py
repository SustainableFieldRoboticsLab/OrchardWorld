from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError:
    _ndi = None


@dataclass
class FruitDetection:
    center_world: np.ndarray
    radius: float
    px: tuple
    npix: int
    rms: float


class FruitPerception:

    def __init__(self, width: int, height: int, fov_deg: float,
                 min_r: float = 0.017, max_r: float = 0.048,
                 max_range: float = 1.35, min_range: float = 0.12,
                 edge_jump: float = 0.015, max_rms: float = 0.005,
                 min_pix: int = 14):
        self.W, self.H = int(width), int(height)
        self.fov_deg = float(fov_deg)
        self.focal_px = 0.5 * self.H / math.tan(0.5 * math.radians(fov_deg))
        self.min_r, self.max_r = min_r, max_r
        self.min_range, self.max_range = min_range, max_range
        self.edge_jump = edge_jump
        self.max_rms = max_rms
        self.min_pix = min_pix


        self.max_elong2 = 3.0
        self.cyl_margin = 1.3


        f = 0.5 * self.H / math.tan(0.5 * math.radians(fov_deg))
        u = (np.arange(self.W) - 0.5 * (self.W - 1))
        v = (0.5 * (self.H - 1) - np.arange(self.H))
        uu, vv = np.meshgrid(u, v)
        d = np.stack([uu / f, vv / f, -np.ones_like(uu)], axis=-1)
        self.dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)


    def detect(self, depth: np.ndarray, cam_pos: np.ndarray,
               cam_R: np.ndarray) -> list[FruitDetection]:
        if _ndi is None:
            return []
        d = depth
        valid = np.isfinite(d) & (d > self.min_range) & (d < self.max_range)
        if valid.sum() < self.min_pix:
            return []

        jump_x = np.abs(np.diff(d, axis=1)) > self.edge_jump
        jump_y = np.abs(np.diff(d, axis=0)) > self.edge_jump
        edges = np.zeros_like(valid)
        edges[:, 1:] |= jump_x
        edges[:, :-1] |= jump_x
        edges[1:, :] |= jump_y
        edges[:-1, :] |= jump_y
        labels, n = _ndi.label(valid & ~edges)
        if n == 0:
            return []
        pts_cam = self.dirs * d[..., None]


        flat = labels.ravel()
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=n + 1)
        starts = np.concatenate([[0], np.cumsum(counts)])

        out: list[FruitDetection] = []
        for lab in range(1, n + 1):
            npx = int(counts[lab])
            if npx < self.min_pix or npx > 6000:
                continue
            idx = order[starts[lab]:starts[lab + 1]]
            ys, xs = np.divmod(idx, self.W)
            p = pts_cam[ys, xs].astype(np.float64)

            ext = p.max(0) - p.min(0)
            if max(ext[0], ext[1]) > 4.5 * self.max_r:
                continue
            c, r, rms = _sphere_fit(p)
            if c is None or not (self.min_r <= r <= self.max_r) or rms > self.max_rms:
                continue


            q = p - p.mean(0)
            wv, Vv = np.linalg.eigh(q.T @ q / len(q))
            if wv[2] > self.max_elong2 * max(wv[1], 1e-12):
                continue
            axis = Vv[:, 2]
            rad = q - np.outer(q @ axis, axis)
            rr = np.linalg.norm(rad, axis=1)
            cyl_rms = float(rr.std())
            if cyl_rms < self.cyl_margin * max(rms, 1e-4):
                continue


            surf = p.mean(0)
            if np.dot(c - surf, surf) < 0.0:
                continue


            zdist = float(np.linalg.norm(c))
            exp_px = math.pi * (r * self.focal_px / max(zdist, 1e-3)) ** 2
            if not (0.15 * exp_px <= npx <= 2.5 * exp_px):
                continue


            if not self._isolated(d, valid, labels, lab, ys, xs):
                continue
            cw = cam_pos + cam_R @ c
            out.append(FruitDetection(center_world=cw, radius=float(r),
                                      px=(float(ys.mean()), float(xs.mean())),
                                      npix=npx, rms=float(rms)))

        out.sort(key=lambda f: -f.npix)
        merged: list[FruitDetection] = []
        for f in out:
            if all(np.linalg.norm(f.center_world - g.center_world) > (f.radius + g.radius)
                   for g in merged):
                merged.append(f)
        return merged

    def _isolated(self, d, valid, labels, lab, ys, xs, min_frac: float = 0.55) -> bool:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        y0 = max(y0 - 4, 0); x0 = max(x0 - 4, 0)
        y1 = min(y1 + 5, self.H); x1 = min(x1 + 5, self.W)
        m = labels[y0:y1, x0:x1] == lab


        inner = _ndi.binary_dilation(m, iterations=1)
        ring = _ndi.binary_dilation(inner, iterations=2) & ~inner
        if ring.sum() == 0:
            return True
        dl = d[y0:y1, x0:x1]
        vl = valid[y0:y1, x0:x1]


        dz = np.abs(dl[ring] - np.median(dl[m]))
        contrast = (~vl[ring]) | (dz > 4.0 * self.edge_jump)
        return float(contrast.mean()) >= min_frac


    def draw_overlay(self, rgba: np.ndarray, dets: list[FruitDetection],
                     depth: np.ndarray) -> np.ndarray:
        img = rgba.copy()
        H, W = img.shape[:2]
        for det in dets:
            cy, cx = int(round(det.px[0])), int(round(det.px[1]))
            zdist = float(depth[min(max(cy, 0), H - 1), min(max(cx, 0), W - 1)])

            rp = max(int(det.radius / max(zdist, 1e-3) * self.focal_px), 3)
            _circle(img, cx, cy, rp, (40, 255, 60))
            img[max(cy - 1, 0):cy + 2, max(cx - 6, 0):cx + 7, :3] = (255, 60, 40)
            img[max(cy - 6, 0):cy + 7, max(cx - 1, 0):cx + 2, :3] = (255, 60, 40)
        return img


def _sphere_fit(p: np.ndarray):
    A = np.concatenate([2.0 * p, np.ones((len(p), 1))], axis=1)
    b = (p * p).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, 0.0, 1e9
    c = sol[:3]
    r2 = sol[3] + float(c @ c)
    if r2 <= 0.0:
        return None, 0.0, 1e9
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.linalg.norm(p - c, axis=1) - r) ** 2)))
    return c, r, rms


def _circle(img: np.ndarray, cx: int, cy: int, r: int, color):
    H, W = img.shape[:2]
    th = np.linspace(0.0, 2.0 * np.pi, max(16, int(4 * r)))
    xs = np.clip((cx + r * np.cos(th)).astype(int), 0, W - 1)
    ys = np.clip((cy + r * np.sin(th)).astype(int), 0, H - 1)
    img[ys, xs, :3] = color
