from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

VJEPA_ID = os.environ.get("OW_VJEPA_ID", "facebook/vjepa2-vitl-fpc64-256")
FEAT_DIM = 1024
CROP_GRID = 4
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class VJEPAEncoder:
    def __init__(self, device="cuda", model_id=VJEPA_ID, dtype=torch.bfloat16, _model=None):
        from transformers import AutoModel
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        if _model is None and os.environ.get("OW_VJEPA_TEST"):

            from transformers import VJEPA2Config, VJEPA2Model
            _model = VJEPA2Model(VJEPA2Config(
                hidden_size=FEAT_DIM, num_hidden_layers=1, num_attention_heads=16,
                pred_hidden_size=32, pred_num_hidden_layers=1, pred_num_attention_heads=4))
            model_id = "__test__"
        if _model is not None:
            self.model = _model
        else:
            try:
                self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype,
                                                       attn_implementation="sdpa")
            except TypeError:
                self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype,
                                                       attn_implementation="sdpa")
        self.model.to(self.device).eval().requires_grad_(False)
        self.size = int(getattr(self.model.config, "crop_size", 256))
        mean, std = IMNET_MEAN, IMNET_STD
        try:
            from transformers import AutoVideoProcessor
            p = AutoVideoProcessor.from_pretrained(model_id)
            mean, std = tuple(p.image_mean), tuple(p.image_std)
        except Exception:
            pass
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _tokens(self, rgb_u8):
        x = torch.as_tensor(np.asarray(rgb_u8)) if not torch.is_tensor(rgb_u8) else rgb_u8
        x = x.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                          antialias=True)
        x = (x - self.mean) / self.std
        clip = x.unsqueeze(1).expand(-1, 2, -1, -1, -1).to(self.dtype)
        out = self.model(pixel_values_videos=clip, skip_predictor=True).last_hidden_state
        g = int(round(out.shape[1] ** 0.5))
        return out.float().view(out.shape[0], g, g, -1)

    @torch.no_grad()
    def encode(self, g_rgb, c_rgb):
        tg = self._tokens(g_rgb)
        tc = self._tokens(c_rgb)
        gv = tg.mean(dim=(1, 2))
        cg = F.adaptive_avg_pool2d(tc.permute(0, 3, 1, 2), CROP_GRID)
        cg = cg.flatten(2).transpose(1, 2)
        return gv.half(), cg.half()
