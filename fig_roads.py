"""Side-by-side road figure: input | GT | U-Net (scribbles) | TopoSAM few-shot (ours) + its MC std.
    python fig_roads.py --tiles 0 7 21
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ccape.data import load_roads
from ccape.infer import infer
from ccape.model import UNet
from ccape.sam import Domain, TopoSAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", type=int, default=[0, 7, 21])
    ap.add_argument("--out", default="results/phase3/fig_roads.png")
    a = ap.parse_args()
    dev = "cuda"
    d = load_roads("test")
    sel = {k: (v[a.tiles] if k != "ids" else [v[i] for i in a.tiles]) for k, v in d.items()}
    sam = TopoSAM().to(dev); sam.load_state_dict(torch.load("runs/sam_roads_fs/last.pt", map_location=dev), strict=False)
    dom = Domain(sam, sel, dev, seed=8, flips=1)
    unet = UNet().to(dev); unet.load_state_dict(torch.load("runs/unet_roads_sp/best.pt", map_location=dev))
    fig, ax = plt.subplots(len(a.tiles), 5, figsize=(20, 4.2 * len(a.tiles)))
    for r in range(len(a.tiles)):
        pu, _ = infer(unet, sel["img"][r], 1, dev)
        ps, pm = dom.predict(sam, r, 10)
        pts, lab = dom.eval_prompts[r]
        cols = [(sel["rgb"][r], "input + click prompts"), (sel["lab"][r], "ground truth"),
                (pu > 0.5, "U-Net + CAPE (scribbles)"), (ps > 0.5, "TopoSAM few-shot (ours)"), (pm.std(0), "TopoSAM MC std")]
        for c, (im, t) in enumerate(cols):
            ax[r, c].imshow(im, cmap=None if c == 0 else ("magma" if c == 4 else "gray"))
            ax[r, c].set_title(t); ax[r, c].axis("off")
        ok = lab >= 0
        ax[r, 0].scatter(pts[ok & (lab == 1), 0], pts[ok & (lab == 1), 1], c="lime", s=12)
        ax[r, 0].scatter(pts[ok & (lab == 0), 0], pts[ok & (lab == 0), 1], c="red", s=12)
    plt.tight_layout(); plt.savefig(a.out, dpi=90); plt.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
