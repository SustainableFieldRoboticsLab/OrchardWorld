from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decision_picker import (N_ACTIONS, STATE_DIM, PICKS, REPOS, R_HARVEST, R_CONTACT,
                              R_DAMAGE, R_TIME)
from .simenv import DEPTH_MAX

BIN_HEADS = ["detach", "success", "contact", "damage", "visible_after"]
CONT_HEADS = ["tau", "target_disp", "wood_disp", "contact_frames"]
FUT_HW = (18, 24)

VARIANTS = {
    "full":      dict(rgb=True, depth=True, state=True, action=True),
    "rgbd":      dict(rgb=True, depth=True, state=False, action=True),
    "rgb":       dict(rgb=True, depth=False, state=False, action=True),
    "rgb_state": dict(rgb=True, depth=False, state=True, action=True),
    "no_action": dict(rgb=True, depth=True, state=True, action=False),
    "tdmpc":     dict(rgb=True, depth=True, state=True, action=True, simnorm=True, ensemble=5),
    "vjepa":     dict(rgb=True, depth=True, state=True, action=True, simnorm=True, ensemble=5,
                      vjepa=True),
}
KAPPA = 0.5
VARIANT_LABEL = {
    "full": "World model (RGB-D + robot state)",
    "rgbd": "World model (RGB-D)",
    "rgb": "World model (RGB)",
    "rgb_state": "World model (RGB + robot state)",
    "no_action": "World model (no action input)",
    "tdmpc": "TD-MPC2-style world model",
    "vjepa": "TD-MPC2-style + V-JEPA 2 features",
}


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = self.c1(F.silu(self.n1(x)))
        y = self.c2(F.silu(self.n2(y)))
        return x + y


class ConvTokenizer(nn.Module):

    def __init__(self, cin, D, hw):
        super().__init__()
        ch = [48, 96, 192, 256]
        layers = [nn.Conv2d(cin, ch[0], 3, padding=1), ResBlock(ch[0])]
        for i in range(3):
            layers += [nn.GroupNorm(8, ch[i]), nn.SiLU(),
                       nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                       ResBlock(ch[i + 1])]
            if i >= 1:
                layers.append(ResBlock(ch[i + 1]))
        layers += [nn.GroupNorm(8, ch[3]), nn.SiLU(), nn.Conv2d(ch[3], D, 1)]
        self.net = nn.Sequential(*layers)
        n = math.ceil(hw[0] / 8) * math.ceil(hw[1] / 8)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        y = self.net(x)
        return y.flatten(2).transpose(1, 2) + self.pos


class FiLMBlock(nn.Module):
    def __init__(self, D, A):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.film = nn.Linear(A, 2 * D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, h, a):
        g, b = self.film(a).chunk(2, dim=-1)
        return h + self.mlp(self.ln(h) * (1 + g) + b)


class SimNorm(nn.Module):

    def __init__(self, group=8):
        super().__init__()
        self.g = group

    def forward(self, x):
        sh = x.shape
        return F.softmax(x.view(*sh[:-1], -1, self.g), dim=-1).view(sh)


class WorldModel(nn.Module):
    def __init__(self, variant="full", D=256, enc_layers=4, dyn_blocks=4,
                 global_hw=(72, 96), crop=64):
        super().__init__()
        self.variant = variant
        v = VARIANTS[variant]
        self.use_rgb, self.use_depth = v["rgb"], v["depth"]
        self.use_state, self.use_action = v["state"], v["action"]
        self.use_vjepa = bool(v.get("vjepa", False))
        self.ensemble = int(v.get("ensemble", 1))
        self.simnorm = bool(v.get("simnorm", False))
        self.cnn_rgb = self.use_rgb and not self.use_vjepa
        cimg = 3 * self.cnn_rgb + 1 * self.use_depth
        self.global_tok = ConvTokenizer(cimg + 1, D, global_hw)
        self.crop_tok = ConvTokenizer(max(cimg, 1), D, (crop, crop))
        if self.use_vjepa:
            from .vjepa import FEAT_DIM, CROP_GRID
            self.vj_g = nn.Sequential(nn.LayerNorm(FEAT_DIM), nn.Linear(FEAT_DIM, D))
            self.vj_c = nn.Sequential(nn.LayerNorm(FEAT_DIM), nn.Linear(FEAT_DIM, D))
            self.vj_pos = nn.Parameter(torch.zeros(1, 1 + CROP_GRID * CROP_GRID, D))
            nn.init.normal_(self.vj_pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.type_emb = nn.Parameter(torch.zeros(1, 3, D))
        if self.use_state:
            self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, D), nn.GELU(), nn.Linear(D, D))
        layer = nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.enc_norm = SimNorm() if self.simnorm else nn.LayerNorm(D)
        A = 64
        self.act_emb = nn.Embedding(N_ACTIONS + 1, A)
        self.dyn = nn.ModuleList([FiLMBlock(D, A) for _ in range(dyn_blocks)])
        self.dyn_norm = SimNorm() if self.simnorm else nn.LayerNorm(D)
        self.head_bin = nn.ModuleList([
            nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Dropout(0.1 if self.ensemble > 1 else 0.0),
                          nn.Linear(D, len(BIN_HEADS))) for _ in range(self.ensemble)])
        self.head_cont = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, len(CONT_HEADS)))
        self.dec = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, FUT_HW[0] * FUT_HW[1]))
        self.proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)

    def encode(self, g, c, s, vg=None, vc=None):
        tg = self.global_tok(g) + self.type_emb[:, 0:1]
        tc = self.crop_tok(c) + self.type_emb[:, 1:2]
        toks = [self.cls.expand(g.shape[0], -1, -1), tg, tc]
        if self.use_vjepa:
            tv = torch.cat([self.vj_g(vg.float()).unsqueeze(1), self.vj_c(vc.float())], 1)
            toks.append(tv + self.vj_pos)
        if self.use_state:
            toks.append(self.state_mlp(s).unsqueeze(1) + self.type_emb[:, 2:3])
        x = self.encoder(torch.cat(toks, dim=1))
        return self.enc_norm(x[:, 0])

    def step(self, h, a):
        if not self.use_action:
            a = torch.full_like(a, N_ACTIONS)
        e = self.act_emb(a)
        x = h
        for blk in self.dyn:
            x = blk(x, e)
        return self.dyn_norm(x)

    calib = None

    def heads_all(self, h1):
        lb = torch.stack([hb(h1) for hb in self.head_bin])
        if self.calib is not None:
            a, b = self.calib
            lb = lb * a.to(lb.dtype) + b.to(lb.dtype)
        return lb, self.head_cont(h1)

    def heads(self, h1):
        lb, lc = self.heads_all(h1)
        if lb.shape[0] == 1:
            return lb[0], lc
        p = torch.sigmoid(lb.float()).mean(0).clamp(1e-5, 1 - 1e-5)
        return torch.logit(p), lc

    def future_depth(self, h1):
        return torch.sigmoid(self.dec(h1)).view(-1, 1, *FUT_HW)


def heatmap(uv, hw, sigma=2.5):
    H, W = hw
    ys = torch.arange(H, device=uv.device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=uv.device, dtype=torch.float32).view(1, 1, W)
    u = uv[:, 0].clamp(-0.5, 1.5).view(-1, 1, 1) * W
    v = uv[:, 1].clamp(-0.5, 1.5).view(-1, 1, 1) * H
    return torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma ** 2)).unsqueeze(1)


RGB_MEAN = torch.tensor([0.25, 0.22, 0.20]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.22, 0.20, 0.20]).view(1, 3, 1, 1)


def prep_images(model, g_rgb, g_dep, c_rgb, c_dep, uv, augment=False):
    dev = g_rgb.device

    def rgbn(x):
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if augment:
            B = x.shape[0]
            gain = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
            bias = torch.empty(B, 3, 1, 1, device=dev).uniform_(-0.04, 0.04)
            x = (x * gain + bias).clamp(0, 1)
        return (x - RGB_MEAN.to(dev)) / RGB_STD.to(dev)

    def depn(x):
        return (x.float().unsqueeze(1) / DEPTH_MAX) * 2.0 - 1.0

    gl, cr = [], []
    if model.cnn_rgb:
        gl.append(rgbn(g_rgb))
        cr.append(rgbn(c_rgb))
    if model.use_depth:
        gl.append(depn(g_dep))
        cr.append(depn(c_dep))
    if not cr:
        cr.append(torch.zeros_like(depn(c_dep)))
    gl.append(heatmap(uv, g_rgb.shape[1:3]))
    return torch.cat(gl, 1), torch.cat(cr, 1)


def future_depth_target(g_dep):
    x = g_dep.float().unsqueeze(1) / DEPTH_MAX
    return F.adaptive_avg_pool2d(x, FUT_HW)


def expected_reward(pb, pc, lam=None):
    tau = pc[..., 0] * 10.0
    return (R_HARVEST * pb[..., 1] - R_CONTACT * pb[..., 2] - R_DAMAGE * pb[..., 3]
            - R_TIME * tau)


@torch.no_grad()
def plan_scores(model, h, gamma=0.95):
    B = h.shape[0]
    dev = h.device
    out = torch.zeros(B, 5, device=dev)

    def member_rewards(hh):
        lb, lc = model.heads_all(hh)
        return expected_reward(torch.sigmoid(lb.float()), lc.float().unsqueeze(0))

    for a in list(PICKS) + list(REPOS):
        h1 = model.step(h, torch.full((B,), a, device=dev, dtype=torch.long))
        r = member_rewards(h1)
        if a in REPOS:
            best = None
            for a2 in PICKS:
                h2 = model.step(h1, torch.full((B,), a2, device=dev, dtype=torch.long))
                r2 = member_rewards(h2)
                best = r2 if best is None else torch.maximum(best, r2)
            r = r + gamma * torch.clamp(best, min=0.0)
        out[:, a] = r.mean(0) - (KAPPA * r.std(0) if r.shape[0] > 1 else 0.0)
    return out


def load_run(run_dir, device, calibrated=False):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    model = WorldModel(meta["variant"], D=meta.get("width", 256))
    sd = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    meta["state_mean_t"] = torch.tensor(meta["state_mean"], dtype=torch.float32, device=device)
    meta["state_std_t"] = torch.tensor(meta["state_std"], dtype=torch.float32, device=device)
    pol = os.path.join(run_dir, "policy_cal.json" if calibrated else "policy.json")
    if os.path.exists(pol):
        meta["policy"] = json.load(open(pol))
    if calibrated:
        c = json.load(open(os.path.join(run_dir, "calib.json")))
        model.calib = (torch.tensor(c["a"], dtype=torch.float32, device=device),
                       torch.tensor(c["b"], dtype=torch.float32, device=device))
        meta["calibrated"] = True
    return model, meta


@torch.no_grad()
def score_observation(model, meta, obs, device, vj=None):
    t = lambda x, dt=None: torch.as_tensor(np.array(x)[None], device=device, dtype=dt)
    g, c = prep_images(model, t(obs["g_rgb"]), t(obs["g_dep"], torch.float32),
                       t(obs["c_rgb"]), t(obs["c_dep"], torch.float32),
                       t(obs["uv"], torch.float32))
    s = (t(obs["state"], torch.float32) - meta["state_mean_t"]) / meta["state_std_t"]
    vg = vc = None
    if model.use_vjepa:
        vg, vc = vj.encode(np.array(obs["g_rgb"])[None], np.array(obs["c_rgb"])[None])
    h = model.encode(g, c, s, vg, vc)
    return plan_scores(model, h)[0].float().cpu().numpy()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
