"""DRIVE loading. Everything is tiny (40 images), so we keep it all in memory / on GPU."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
H, W = 584, 565
PH, PW = 592, 576            # padded to a multiple of 16 for the U-Net
CALIB_IDS = (37, 38, 39, 40)  # held out of training: model selection + conformal calibration


def _fov(rgb):
    return binary_erosion(rgb[..., 0] > 20, iterations=2)


def _norm(rgb, fov):
    x = rgb.astype(np.float32) / 255.0
    m, s = x[fov].mean(0), x[fov].std(0) + 1e-6
    return ((x - m) / s * fov[..., None]).transpose(2, 0, 1)


def load(split):
    """split in {'train', 'calib', 'test'} -> dict: img (N,3,H,W) float, lab/fov (N,H,W) float, rgb (N,H,W,3) uint8, ids."""
    folder = "val" if split == "test" else "train"
    out = {"img": [], "lab": [], "fov": [], "rgb": [], "ids": []}
    for p in sorted((ROOT / folder / "input").glob("*.tif")):
        i = int(p.stem)
        if folder == "train" and ((i in CALIB_IDS) != (split == "calib")):
            continue
        rgb = np.array(Image.open(p))
        lab = np.array(Image.open(next((ROOT / folder / "label").glob(f"{p.stem}*.png")))) > 127
        fov = _fov(rgb)
        out["img"].append(_norm(rgb, fov)); out["lab"].append(lab & fov); out["fov"].append(fov)
        out["rgb"].append(rgb); out["ids"].append(i)
    return _stack(out)


def _stack(out):
    return {k: v if k == "ids" else np.stack(v) if k == "rgb" else np.stack(v).astype(np.float32) for k, v in out.items()}


ROADS = ROOT.parent / "roads" / "eval"
ROAD_SPLITS = {"train": slice(0, 40), "calib": slice(40, 60), "test": slice(60, 120)}


def load_roads(split, crop=1024):
    """Massachusetts Roads (eval tiles, 1500^2): central crop. 40 few-shot train / 20 calib / 60 test tiles."""
    out = {"img": [], "lab": [], "fov": [], "rgb": [], "ids": []}
    o = (1500 - crop) // 2
    for p in sorted((ROADS / "images").glob("*_sat.jpg"))[ROAD_SPLITS[split]]:
        rgb = np.array(Image.open(p))[o:o + crop, o:o + crop]
        lab = np.array(Image.open(ROADS / "masks" / p.name.replace("_sat.jpg", "_mask.png")))[o:o + crop, o:o + crop] > 127
        fov = rgb.max(-1) > 0                      # a few tiles have no-data (black) regions
        out["img"].append(_norm(rgb, fov)); out["lab"].append(lab & fov); out["fov"].append(fov)
        out["rgb"].append(rgb); out["ids"].append(p.stem)
    return _stack(out)


def pad(x):
    """Pad (..., H, W) tensor to (..., PH, PW)."""
    return F.pad(x, (0, PW - x.shape[-1], 0, PH - x.shape[-2]))


class CropSampler:
    """Random square crops + dihedral augmentation, done entirely on GPU."""

    def __init__(self, d, size, device):
        self.x = torch.from_numpy(d["img"]).to(device)
        self.y = torch.from_numpy(d["lab"]).to(device)
        self.size, self.dev = size, device

    def __call__(self, bs):
        n, _, h, w = self.x.shape
        s = self.size
        idx = torch.randint(n, (bs,))
        ys, xs = torch.randint(h - s + 1, (bs,)), torch.randint(w - s + 1, (bs,))
        xb = torch.stack([self.x[i, :, a:a + s, b:b + s] for i, a, b in zip(idx, ys, xs)])
        yb = torch.stack([self.y[i, a:a + s, b:b + s] for i, a, b in zip(idx, ys, xs)])
        k = int(torch.randint(4, ()))
        xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (1, 2))
        if torch.rand(()) < 0.5:
            xb, yb = xb.flip(3), yb.flip(2)
        # photometric jitter: per-sample contrast/brightness
        xb = xb * (1 + 0.2 * (torch.rand(bs, 1, 1, 1, device=self.dev) - 0.5)) + 0.2 * (torch.rand(bs, 1, 1, 1, device=self.dev) - 0.5)
        return xb.contiguous(), yb.contiguous()
