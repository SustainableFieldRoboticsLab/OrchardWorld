from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import load_config, rdir, logger, write_json
from .dataset import load
from .decision_picker import SKIP, PICKS
from .model import (WorldModel, BIN_HEADS, prep_images, future_depth_target,
                    count_params)


_TORCH_DT = {np.dtype("uint8"): torch.uint8, np.dtype("float16"): torch.float16,
             np.dtype("float32"): torch.float32}


def labels_from_table(table):
    yb = np.stack([table[k].values.astype(np.float32) for k in BIN_HEADS], 1)
    cf = (table["arm_wood_frames"] + table["base_wood_frames"]).values.astype(np.float32)
    yc = np.stack([table["tau"].values / 10.0,
                   table["target_disp"].values * 10.0,
                   table["wood_disp"].values * 10.0,
                   np.log1p(cf) / 3.0], 1).astype(np.float32)
    return yb, yc


class GPUData:

    def __init__(self, arrs, table, device, log, vjepa=False):
        t0 = time.time()
        self.dev = device
        self.T = {}
        keys = ["g_rgb", "g_dep", "c_rgb", "c_dep", "state", "uv"]
        self.vjepa = vjepa
        for tag in ("pre", "post"):
            for k in keys + (["vj_g", "vj_c"] if vjepa else []):
                name = f"vj_{tag}_{k[3:]}" if k.startswith("vj_") else f"{tag}_{k}"
                if name not in arrs:
                    raise SystemExit(f"missing {name}.npy - run the phase-2 'features' stage first")
                src = arrs[name]
                dst = torch.empty(src.shape, dtype=_TORCH_DT[np.dtype(src.dtype)], device=device)
                for i in range(0, len(src), 8192):
                    dst[i:i + 8192] = torch.from_numpy(np.ascontiguousarray(src[i:i + 8192])).to(device)
                self.T[f"{tag}_{k}"] = dst
        yb, yc = labels_from_table(table)
        self.yb = torch.from_numpy(yb).to(device)
        self.yc = torch.from_numpy(yc).to(device)
        self.action = torch.from_numpy(table["action"].values.astype(np.int64)).to(device)
        self.next_idx = torch.from_numpy(table["next_idx"].values.astype(np.int64)).to(device)
        log(f"data on {device}: {len(table)} rows in {time.time() - t0:.0f}s")

    def obs(self, model, tag, idx, mean, std, augment=False):
        T = self.T
        g, c = prep_images(model, T[f"{tag}_g_rgb"][idx], T[f"{tag}_g_dep"][idx],
                           T[f"{tag}_c_rgb"][idx], T[f"{tag}_c_dep"][idx],
                           T[f"{tag}_uv"][idx].float(), augment=augment)
        s = (T[f"{tag}_state"][idx].float() - mean) / std
        if self.vjepa:
            return g, c, s, T[f"{tag}_vj_g"][idx], T[f"{tag}_vj_c"][idx]
        return g, c, s, None, None


def ens_bce(lb_all, y, pos_w):
    K = lb_all.shape[0]
    return F.binary_cross_entropy_with_logits(
        lb_all.float(), y.unsqueeze(0).expand(K, -1, -1), pos_weight=pos_w)


def compute_losses(model, D, idx, mean, std, pos_w, augment):
    g, c, s, vg, vc = D.obs(model, "pre", idx, mean, std, augment)
    a = D.action[idx]
    h = model.encode(g, c, s, vg, vc)
    h1 = model.step(h, a)
    lb_all, lc = model.heads_all(h1)
    lb, _ = model.heads(h1)
    yb, yc = D.yb[idx], D.yc[idx]
    l_bin = ens_bce(lb_all, yb, pos_w)
    m = torch.isfinite(yc)
    l_cont = F.smooth_l1_loss(lc.float()[m], yc[m]) if m.any() else lc.sum() * 0
    l_dep = F.l1_loss(model.future_depth(h1).float(), future_depth_target(D.T["post_g_dep"][idx]))
    with torch.no_grad():
        gp, cp, sp, vgp, vcp = D.obs(model, "post", idx, mean, std, False)
        hp = model.encode(gp, cp, sp, vgp, vcp)
    l_lat = (1 - F.cosine_similarity(model.proj(h1).float(), hp.float(), dim=-1)).mean()
    nxt = D.next_idx[idx]
    ok = nxt >= 0
    if ok.any():
        nx = nxt[ok]
        ok2 = D.action[nx] != SKIP
        if ok2.any():
            h2 = model.step(h1[ok][ok2], D.action[nx][ok2])
            lb2_all, _ = model.heads_all(h2)
            l_two = ens_bce(lb2_all, D.yb[nx][ok2], pos_w)
        else:
            l_two = l_bin * 0
    else:
        l_two = l_bin * 0
    total = l_bin + 0.5 * l_cont + 0.5 * l_dep + 0.2 * l_lat + 0.5 * l_two
    parts = dict(bin=l_bin.item(), cont=l_cont.item(), dep=l_dep.item(), lat=l_lat.item(),
                 two=l_two.item())
    return total, parts, torch.sigmoid(lb.float())


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)["train"]
    name = f"{args.variant}_s{args.seed}"
    out = rdir("runs", name)
    log = logger(f"train {name}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    table, arrs = load()
    usable = (table["skipped"] == 0).values & (table["batch_status"].fillna("nan") != "nan").values
    finite = np.ones(len(table), dtype=bool)
    for k in ("pre_state", "post_state"):
        for i in range(0, len(table), 8192):
            finite[i:i + 8192] &= np.isfinite(np.asarray(arrs[k][i:i + 8192])).all(1)
    if (~finite).any():
        log(f"dropping {int((~finite).sum())} row(s) with non-finite robot state: "
            f"{np.nonzero(~finite)[0][:10].tolist()}")
    usable &= finite
    tr = np.nonzero(usable & (table["split"] == "train").values)[0]
    va = np.nonzero(usable & (table["split"] == "val").values)[0]
    if len(va) < 20:
        va = tr[: max(len(tr) // 5, 1)]
    log(f"train {len(tr)}  val {len(va)}")
    from .model import VARIANTS
    D = GPUData(arrs, table, dev, log, vjepa=bool(VARIANTS[args.variant].get("vjepa")))
    st = np.asarray(arrs["pre_state"][tr], dtype=np.float64)
    mean = st.mean(0)
    std = np.maximum(st.std(0), 1e-3)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev)
    yb_tr = D.yb[torch.as_tensor(tr, device=dev)]
    pos = yb_tr.mean(0).clamp(1e-3, 1 - 1e-3)
    pos_w = ((1 - pos) / pos).clamp(1.0, 20.0)
    model = WorldModel(args.variant, D=int(cfg.get("width", 256))).to(dev)
    n_par = count_params(model)
    log(f"model params: {n_par / 1e6:.2f} M; pos_weight {pos_w.cpu().numpy().round(2)}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    bs = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    steps_per_epoch = max(math.ceil(len(tr) / bs), 1)
    total_steps = epochs * steps_per_epoch
    warm = min(300, total_steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))
    amp = dev.type == "cuda"
    hist = []
    best = (float("inf"), -1)
    t_start = time.time()
    va_t = torch.as_tensor(va, device=dev)
    max_s = 60.0 * float(cfg.get("max_minutes", 1e9))
    for ep in range(epochs):
        if time.time() - t_start > max_s:
            log(f"time budget of {max_s / 60:.0f} min reached; stopping after {ep} epochs")
            break
        model.train()
        perm = torch.as_tensor(np.random.permutation(tr), device=dev)
        agg = {}
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            if len(idx) < 2:
                continue
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                loss, parts, _ = compute_losses(model, D, idx, mean_t, std_t, pos_w, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_epoch
        model.eval()
        vparts, probs = {}, []
        nvb = max(math.ceil(len(va) / 512), 1)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            for i in range(nvb):
                idx = va_t[i * 512:(i + 1) * 512]
                if len(idx) == 0:
                    continue
                _, parts, p = compute_losses(model, D, idx, mean_t, std_t, pos_w, False)
                probs.append(p.cpu().numpy())
                for k, v in parts.items():
                    vparts[k] = vparts.get(k, 0.0) + v * len(idx) / len(va)
        P = np.concatenate(probs) if probs else np.zeros((0, len(BIN_HEADS)))
        yv = table.iloc[va]
        pick = yv["action"].isin(PICKS).values
        val_score = vparts.get("bin", 0) + 0.5 * vparts.get("cont", 0) + 0.5 * vparts.get("two", 0)
        rec = dict(epoch=ep + 1, train=agg, val=vparts, val_score=val_score,
                   auc_detach=auroc(yv["detach"].values[pick], P[pick, 0]) if len(P) else None,
                   auc_success=auroc(yv["success"].values[pick], P[pick, 1]) if len(P) else None,
                   auc_contact=auroc(yv["contact"].values, P[:, 2]) if len(P) else None,
                   lr=sched.get_last_lr()[0], t=time.time() - t_start)
        hist.append(rec)
        if val_score < best[0]:
            best = (val_score, ep + 1)
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        log(f"ep {ep + 1:3d}/{epochs} train {agg.get('bin', 0):.3f} val {val_score:.3f} "
            f"aucD {rec['auc_detach']} aucS {rec['auc_success']} aucC {rec['auc_contact']} "
            f"({time.time() - t_start:.0f}s)")
        write_json(os.path.join(out, "history.json"), hist)
    meta = dict(variant=args.variant, seed=args.seed, width=int(cfg.get("width", 256)),
                params=n_par, epochs=epochs, best_epoch=best[1], best_val=best[0],
                n_train=int(len(tr)), n_val=int(len(va)), train_seconds=time.time() - t_start,
                state_mean=mean.tolist(), state_std=std.tolist(),
                pos_weight=pos_w.cpu().numpy().tolist(),
                device=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu")
    write_json(os.path.join(out, "meta.json"), meta)
    log(f"done: best epoch {best[1]} val {best[0]:.4f}, {meta['train_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
