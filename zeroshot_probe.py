"""Why does vessel -> road zero-shot fail? Probe the two obvious domain gaps without any training:
  polarity: vessels are DARK lines on a bright fundus, roads are BRIGHT lines on dark terrain  -> invert intensities
  scale:    roads are ~2x wider than DRIVE vessels                                              -> downscale input
Predictions are upsampled back and scored against the original 1024^2 road masks.
    python zeroshot_probe.py --n 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ccape.data import _norm, load_roads
from ccape.infer import infer
from ccape.metrics import all_metrics
from ccape.model import UNet
from ccape.sam import Domain, TopoSAM


def transform(d, invert, scale):
    rgb = d["rgb"]
    if invert:
        rgb = 255 - rgb
    if scale != 1:
        n = int(rgb.shape[1] * scale)
        rgb = np.stack([np.array(Image.fromarray(r).resize((n, n), Image.BILINEAR)) for r in rgb])
    fov = rgb.max(-1) > 0 if not invert else np.ones(rgb.shape[:3], bool)
    lab = np.stack([np.array(Image.fromarray(l.astype(np.uint8) * 255).resize(fov.shape[1:][::-1], Image.NEAREST)) > 127
                    for l in d["lab"]]).astype(np.float32)
    return {"rgb": rgb, "img": np.stack([_norm(r, f) for r, f in zip(rgb, fov)]), "lab": lab,
            "fov": fov.astype(np.float32), "ids": d["ids"]}


def up(p, n):
    return F.interpolate(torch.from_numpy(p)[None, None], size=(n, n), mode="bilinear", align_corners=False)[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sam", nargs="+", default=["runs/sam_drive/best.pt", "runs/sam_drive_pol/best.pt"])
    ap.add_argument("--unet", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    d0 = load_roads("test")
    d0 = {k: v[:a.n] for k, v in d0.items()}
    gt, fv = d0["lab"] > 0, d0["fov"] > 0
    res = {}
    models = {Path(c).parent.name: c for c in a.sam}
    enc = TopoSAM().to(dev)
    variants = [(inv, sc) for sc in (1.0, 0.5) for inv in (False, True)]
    doms = {(inv, sc): Domain(enc, transform(d0, inv, sc), dev, seed=3, flips=1) for inv, sc in variants}
    for name, ck in models.items():
        m = TopoSAM().to(dev); m.load_state_dict(torch.load(ck, map_location=dev), strict=False)
        P = {v: [up(doms[v].predict(m, i)[0], 1024) for i in range(a.n)] for v in variants}
        for sc in (1.0, 0.5):
            P[("ens", sc)] = [(x + y) / 2 for x, y in zip(P[(False, sc)], P[(True, sc)])]
        for v, ps in P.items():
            ms = [all_metrics(p, g, f) for p, g, f in zip(ps, gt, fv)]
            key = f"{name} | invert={v[0]} | scale={v[1]}"
            res[key] = {k: float(np.mean([mm[k] for mm in ms])) for k in ms[0]}
            print(key, json.dumps({k: round(x, 3) for k, x in res[key].items()}), flush=True)
    Path("results/zeroshot_probe_pol.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
