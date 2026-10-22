from __future__ import annotations

import copy

import numpy as np
import warp as wp


_PICKER_SHARED = {"sim", "tm", "cam", "drv", "rp", "met", "ik", "apples", "B", "policy",
                  "sink", "rng", "_apple_r", "_apple_idx", "_apple_glob", "_zone_ref",
                  "arm_home", "arm_tq", "arm_dof", "chassis", "wrist", "env", "n_env_apples"}


class EnvIndex:

    def __init__(self, batch):
        m = batch.tm.model
        ne = batch.num_envs
        jw = m.joint_world.numpy() if m.joint_world is not None else None
        nj = m.joint_count
        q_s = m.joint_q_start.numpy()
        qd_s = m.joint_qd_start.numpy()
        q_end = list(q_s[1:nj]) + [m.joint_coord_count]
        qd_end = list(qd_s[1:nj]) + [m.joint_dof_count]
        if jw is None:
            jw = np.minimum(np.arange(nj) // max(nj // ne, 1), ne - 1)
        self.coords, self.dofs = [], []
        for e in range(ne):
            js = np.nonzero(jw == e)[0]
            self.coords.append(np.concatenate([np.arange(q_s[j], q_end[j]) for j in js]).astype(np.int64)
                               if len(js) else np.zeros(0, np.int64))
            self.dofs.append(np.concatenate([np.arange(qd_s[j], qd_end[j]) for j in js]).astype(np.int64)
                             if len(js) else np.zeros(0, np.int64))
        bw = m.body_world.numpy() if m.body_world is not None else batch.body_env
        self.bodies = [np.nonzero(bw == e)[0] for e in range(ne)]
        sb = m.shape_body.numpy()
        self.shapes = [np.nonzero((sb >= 0) & np.isin(sb, self.bodies[e]))[0] for e in range(ne)]
        na = batch.n_env_apples
        self.apples = [np.arange(e * na, (e + 1) * na) for e in range(ne)]
        nb = batch.n_brk_env
        self.brk = [np.arange(e * nb, (e + 1) * nb) for e in range(ne)]
        rb = batch.tm.robot_data
        self.arm_tq = [np.asarray(x, dtype=np.int64) for x in rb["arm_tq"]]
        self.planar_tq = [np.arange(int(rb["planar_tq"][e]), int(rb["planar_tq"][e]) + 4)
                          for e in range(ne)]
        self.planar_dof = [np.arange(int(rb["planar_dof"][e]), int(rb["planar_dof"][e]) + 4)
                           for e in range(ne)]


def _get(a):
    return a.numpy()


class EnvSnapshot:

    def __init__(self, batch, idx: EnvIndex, e: int, picker):
        self.e = e
        sim = batch.sim
        st = [sim.state_0, sim.state_1]
        c, d, b = idx.coords[e], idx.dofs[e], idx.bodies[e]
        self.phys = []
        for s in st:
            self.phys.append(dict(joint_q=_get(s.joint_q)[c].copy(), joint_qd=_get(s.joint_qd)[d].copy(),
                                  body_q=_get(s.body_q)[b].copy(), body_qd=_get(s.body_qd)[b].copy()))
        drv = batch.driver
        self.tq = drv._tq_host[np.concatenate([idx.arm_tq[e], idx.planar_tq[e]])].copy()
        self.tqd = drv._target_host[idx.planar_dof[e]].copy()
        ap = sim.apples
        a = idx.apples[e]
        self.apples = dict(detached=ap.detached[a].copy(), over=ap._over[a].copy(),
                           over_t=ap._over_t[a].copy(), flag=ap._flag_host[a].copy(),
                           held=ap._held_host[a].copy(), hold_body=ap._hold_body_host[a].copy())
        br = sim.breaker
        self.brk = None
        if br is not None and len(idx.brk[e]):
            k = idx.brk[e]
            self.brk = dict(broken=br.broken[k].copy(), over=br._over[k].copy(), bf=br._bf_host[k].copy())
            if getattr(br, "_direct", False):
                w = e if br._gain_host.shape[0] > 1 else 0
                wf = e if br._fric_host.shape[0] > 1 else 0
                self.brk.update(gain=br._gain_host[w].copy(), bias=br._bias_host[w].copy(),
                                fric=br._fric_host[wf].copy(), age=br._age[k].copy(),
                                fell=br._fell[k].copy(), frozen=br._frozen[k].copy())
            if br._colgroup_host is not None:
                self.brk["colgroup"] = br._colgroup_host[idx.shapes[e]].copy()
        self.drag = sim._drag_scale_host[b].copy()
        excl = _PICKER_SHARED | set(getattr(picker, "_snap_exclude", ()))
        self.picker = {k: copy.deepcopy(v) for k, v in picker.__dict__.items() if k not in excl}

    def restore(self, batch, idx: EnvIndex, picker):
        e = self.e
        sim = batch.sim
        c, d, b = idx.coords[e], idx.dofs[e], idx.bodies[e]
        for s, ph in zip([sim.state_0, sim.state_1], self.phys):
            for name, ix in (("joint_q", c), ("joint_qd", d), ("body_q", b), ("body_qd", b)):
                arr = getattr(s, name)
                h = arr.numpy()
                h[ix] = ph[name]
                arr.assign(h)
        drv = batch.driver
        drv._tq_host[np.concatenate([idx.arm_tq[e], idx.planar_tq[e]])] = self.tq
        drv._target_host[idx.planar_dof[e]] = self.tqd
        sim.control.joint_target_q.assign(drv._tq_host)
        sim.control.joint_target_qd.assign(drv._target_host)
        ap = sim.apples
        a = idx.apples[e]
        ap.detached[a] = self.apples["detached"]
        ap._over[a] = self.apples["over"]
        ap._over_t[a] = self.apples["over_t"]
        ap._flag_host[a] = self.apples["flag"]
        ap._held_host[a] = self.apples["held"]
        ap._hold_body_host[a] = self.apples["hold_body"]
        ap._flag.assign(ap._flag_host)
        ap._held.assign(ap._held_host)
        ap._hold_body.assign(ap._hold_body_host)
        ap.broken_count = int(ap.detached.sum())
        br = sim.breaker
        if self.brk is not None:
            k = idx.brk[e]
            br.broken[k] = self.brk["broken"]
            br._over[k] = self.brk["over"]
            br._bf_host[k] = self.brk["bf"]
            br.broken_flag.assign(br._bf_host)
            br.broken_count = int(br.broken.sum())
            if "gain" in self.brk:
                w = e if br._gain_host.shape[0] > 1 else 0
                wf = e if br._fric_host.shape[0] > 1 else 0
                br._gain_host[w] = self.brk["gain"]
                br._bias_host[w] = self.brk["bias"]
                br._fric_host[wf] = self.brk["fric"]
                br._age[k] = self.brk["age"]
                br._fell[k] = self.brk["fell"]
                br._frozen[k] = self.brk["frozen"]
                br._gain_wp.assign(br._gain_host)
                br._bias_wp.assign(br._bias_host)
                br._fric_wp.assign(br._fric_host)
            if "colgroup" in self.brk:
                br._colgroup_host[idx.shapes[e]] = self.brk["colgroup"]
                sim.model.shape_collision_group.assign(br._colgroup_host)
        sim._drag_scale_host[b] = self.drag
        sim._drag_scale.assign(sim._drag_scale_host)

        try:
            mask = np.zeros(batch.num_envs, dtype=bool)
            mask[e] = True
            sim.solver.reset(sim.state_0, world_mask=wp.array(mask, dtype=wp.bool,
                                                              device=sim.model.device), flags=0)
        except Exception:
            pass

        sim._bq_np_step = -1
        sim._jq_np_step = -1
        for k, v in self.picker.items():
            setattr(picker, k, copy.deepcopy(v))
