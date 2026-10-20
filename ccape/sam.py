"""Phase 3: TopoSAM = frozen SAM ViT-B encoder + trainable prompt encoder / mask decoder + high-res refiner.

The refiner sees [normalised image, upsampled SAM logit] at native resolution and predicts a residual on top of
SAM's logit, so it starts from SAM and only has to recover the thin structures SAM's 256x256 output cannot draw.
Its Dropout2d makes the whole model an MC-Dropout sampler (phase-1 uncertainty / conformal work unchanged).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import UNet

SAM_PATH = Path(__file__).resolve().parent.parent / "external" / "sam" / "sam-vit-base"
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
S = 1024


def sam_frame(rgb_u8):
    """(N,H,W,3) uint8 -> (N,3,1024,1024) SAM input (longest side resized to 1024, zero pad), scale factor."""
    x = torch.from_numpy(rgb_u8).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    s = S / max(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    return F.pad(x, (0, S - x.shape[-1], 0, S - x.shape[-2])), s


class TopoSAM(nn.Module):
    def __init__(self, refine=True, p=0.2):
        super().__init__()
        from transformers import SamModel
        self.sam = SamModel.from_pretrained(str(SAM_PATH))
        self.sam.vision_encoder.requires_grad_(False)
        self.refiner = UNet(c=16, depth=3, p=p, cin=4) if refine else None
        if self.refiner is not None:            # start as identity on SAM's logit
            nn.init.zeros_(self.refiner.head.weight); nn.init.zeros_(self.refiner.head.bias)

    @torch.no_grad()
    def embed(self, x1024, chunk=4):
        self.sam.vision_encoder.eval()
        with torch.autocast("cuda", torch.bfloat16):
            return torch.cat([self.sam.get_image_embeddings(x1024[i:i + chunk]).float() for i in range(0, len(x1024), chunk)])

    def forward(self, emb, img, pts, lab, scale, hw):
        """emb (B,256,64,64), img (B,3,h,w) normalised, zero-padded to multiples of 8, hw = true (H,W) inside it,
        pts (B,N,2) xy in native pixels, lab (B,N) 1/0, scale = native->1024 factor. -> logits (B,h,w)."""
        o = self.sam(image_embeddings=emb, input_points=(pts * scale)[:, None], input_labels=lab[:, None],
                     multimask_output=False)
        low = o.pred_masks[:, 0, 0]                                   # (B,256,256) over the 1024 frame
        (H0, W0), (h, w) = hw, img.shape[-2:]
        hh, ww = min(256, int(np.ceil(H0 * scale / 4))), min(256, int(np.ceil(W0 * scale / 4)))
        up = F.interpolate(low[:, None, :hh, :ww], size=(H0, W0), mode="bilinear", align_corners=False)
        up = F.pad(up, (0, w - W0, 0, h - H0), value=-10.0)
        if self.refiner is None:
            return up[:, 0]
        return up[:, 0] + self.refiner(torch.cat([img, up], 1))

    def mc_mode(self):
        self.eval()
        if self.refiner is not None:
            self.refiner.mc_mode()
        return self


def randconv(rgb, rng, sizes=(1, 3, 5, 7)):
    """RandConv (Xu et al., ICLR'21): random linear filter on RGB, per-channel min-max rescale, random mix with the
    original. Keeps geometry/topology, scrambles colour and texture statistics. (N,H,W,3) uint8 -> same."""
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    k = int(rng.choice(sizes))
    w = torch.from_numpy(rng.normal(0, 1 / np.sqrt(3 * k * k), (3, 3, k, k))).float()
    y = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w)
    lo, hi = y.amin((2, 3), keepdim=True), y.amax((2, 3), keepdim=True)
    y = (y - lo) / (hi - lo + 1e-6)
    a = float(rng.uniform(0, 1))
    out = a * x + (1 - a) * y
    return (out.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def pad8(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, -w % 8, 0, -h % 8))


class Domain:
    """Everything TopoSAM needs for a set of images: cached SAM embeddings for 4 flips, padded inputs,
    fixed evaluation prompts. flip f in 0..3: bit0 = horizontal, bit1 = vertical."""

    def __init__(self, model, d, dev="cuda", frac=0.1, k_eval=10, seed=0, flips=4, polarity=False, texture=0):
        from .data import _norm
        from .sparse import scribbles
        self.d, self.dev = d, dev
        self.N, self.H, self.W = d["lab"].shape
        self.lab = torch.from_numpy(d["lab"]).to(dev)
        self.scale = S / max(self.H, self.W)
        sc = [scribbles(g > 0, f > 0, frac, seed=seed * 1000 + i) for i, (g, f) in enumerate(zip(d["lab"], d["fov"]))]
        self.pos = torch.from_numpy(np.stack([s[0] for s in sc])).to(dev)
        self.neg = torch.from_numpy(np.stack([s[1] for s in sc])).to(dev)
        self.pos_xy = [np.stack(np.nonzero(s[0])[::-1], 1) for s in sc]
        self.neg_xy = [np.stack(np.nonzero(s[1])[::-1], 1) for s in sc]
        # appearance transforms t (identity, inversion, RandConv textures) x flips f: variant v = t * flips + f.
        # Inversion -> polarity invariance (vessels dark, roads bright); RandConv -> texture invariance.
        rgbs = [d["rgb"]] + ([255 - d["rgb"]] if polarity else [])
        rgbs += [randconv(d["rgb"], np.random.default_rng(seed * 100 + j)) for j in range(texture)]
        fovs = d["fov"] > 0
        self.imgs = [torch.from_numpy(np.stack([_norm(r, f) for r, f in zip(R, fovs)])).to(dev) for R in rgbs]
        self.img = self.imgs[0]
        embs = []
        for R in rgbs:
            for f in range(flips):
                rgb = R
                if f & 1: rgb = rgb[:, :, ::-1]
                if f & 2: rgb = rgb[:, ::-1]
                embs.append(model.embed(sam_frame(np.ascontiguousarray(rgb))[0].to(dev)).to(torch.bfloat16))
        self.emb = torch.stack(embs)                                    # (transforms*flips, N, 256, 64, 64)
        self.flips = flips
        r = np.random.default_rng(seed + 12345)
        self.eval_prompts = [self.prompts(i, 0, k_eval, r) for i in range(self.N)]

    def flip(self, x, f):
        dims = [d for d, b in ((-1, 1), (-2, 2)) if f & b]
        return x.flip(dims) if dims else x

    def prompts(self, i, f, k, rng):
        """k positive + k negative clicks. A class with no scribble (e.g. a road-free tile) gets k padding points
        with label -10, which SAM's prompt encoder ignores, so every image has exactly 2k prompt slots."""
        pts, labs = [], []
        for xy, l in ((self.pos_xy[i], 1), (self.neg_xy[i], 0)):
            if len(xy) == 0:
                pts.append(np.zeros((k, 2), np.float32)); labs.append(np.full(k, -10)); continue
            p = xy[rng.choice(len(xy), k, replace=len(xy) < k)].astype(np.float32)
            if f & 1: p[:, 0] = self.W - 1 - p[:, 0]
            if f & 2: p[:, 1] = self.H - 1 - p[:, 1]
            pts.append(p); labs.append(np.full(k, l))
        return np.concatenate(pts), np.concatenate(labs).astype(np.int64)

    def batch(self, idx, v, k, rng):
        f, t = v % self.flips, v // self.flips
        pl = [self.prompts(i, f, k, rng) for i in idx]
        pts = torch.from_numpy(np.stack([p for p, _ in pl])).to(self.dev)
        lab = torch.from_numpy(np.stack([l for _, l in pl])).to(self.dev)
        g = lambda t: pad8(self.flip(t[idx], f))
        return self.emb[v, idx].float(), g(self.imgs[t]), pts, lab, g(self.lab), g(self.pos), g(self.neg)

    @torch.no_grad()
    def predict(self, model, i, T=0, chunk=10, prompt_ens=False, k_ens=10):
        """-> p_det (H,W), p_mc (T,H,W) or None. p_det uses the fixed evaluation prompts of image i.
        prompt_ens: each stochastic sample also redraws its k_ens + k_ens clicks from the image's scribbles
        (prompt ensemble: SAM's sensitivity to *where* the user clicked becomes epistemic signal)."""
        pts, lab = (torch.from_numpy(a)[None].to(self.dev) for a in self.eval_prompts[i])
        e, x = self.emb[0, i:i + 1].float(), pad8(self.img[i:i + 1])
        hw = (self.H, self.W)
        run = lambda n, P, L: torch.sigmoid(model(e.expand(n, -1, -1, -1), x.expand(n, -1, -1, -1), P, L,
                                                  self.scale, hw).float())[:, :self.H, :self.W]
        with torch.autocast("cuda", torch.bfloat16):
            model.eval(); p_det = run(1, pts, lab)[0].cpu().numpy()
            p_mc = None
            if T:
                model.mc_mode()
                out = []
                rng = np.random.default_rng(1000 + i)
                for k in range(0, T, chunk):
                    n = min(chunk, T - k)
                    if prompt_ens:
                        pl = [self.prompts(i, 0, k_ens, rng) for _ in range(n)]
                        P = torch.from_numpy(np.stack([p for p, _ in pl])).to(self.dev)
                        L = torch.from_numpy(np.stack([l for _, l in pl])).to(self.dev)
                    else:
                        P, L = pts.expand(n, -1, -1), lab.expand(n, -1)
                    out.append(run(n, P, L))
                p_mc = torch.cat(out).cpu().numpy()
        model.eval()
        return p_det, p_mc


def clicks(pos, neg, k, rng):
    """Sample k positive + k negative clicks (xy) from scribble masks -> pts (2k,2), lab (2k,)."""
    out_p, out_l = [], []
    for m, l in ((pos, 1), (neg, 0)):
        ys, xs = np.nonzero(m)
        i = rng.choice(len(ys), k, replace=len(ys) < k) if len(ys) else np.zeros(0, int)
        out_p.append(np.stack([xs[i], ys[i]], 1)); out_l.append(np.full(len(i), l))
    return np.concatenate(out_p).astype(np.float32), np.concatenate(out_l)
