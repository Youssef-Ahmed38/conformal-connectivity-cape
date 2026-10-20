"""Multi-seed evaluation: per-run per-image metrics + per-pair connectivity, cached in results/seeds/ (resumable).
Aggregation (mean ± std over seeds, paired Wilcoxon over images, CRC/conformal per seed) is in aggregate_seeds.py.
    python eval_seeds.py --group drive      # phase 1 + 2 runs on the 20 DRIVE test images
    python eval_seeds.py --group roads      # phase 3 runs on the 80 road tiles (20 calib + 60 test)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccape.clickmap import heatmaps
from ccape.data import load, load_roads
from ccape.infer import infer
from ccape.metrics import all_metrics
from ccape.model import UNet
from ccape.sam import Domain, TopoSAM
from ccape.topo_uq import pair_scores

OUT = Path("results/seeds")
S = lambda base, n=3: [base] + [f"{base}_s{i}" for i in range(1, n)]
DRIVE = {  # family -> run dirs (seed 0, 1, 2); all U-Nets, checkpoint best.pt
    "dice": S("dice"), "cldice": S("cldice"), "cape": S("cape"),
    "sp_none": S("sp_none"), "sp_random": S("sp_random"), "sp_unc": S("sp_unc"), "sp_unc_div": S("sp_unc_div"),
    "sp_unc_div_r12": ["sp_unc_div_r12"], "sp_unc_div_r48": S("sp_unc_div_r48"),
    "sp_unc_div_noisy": [f"sp_unc_div_noisy_s{i}" for i in range(3)],
}
ROADS = {  # family -> [(run dir, kind, checkpoint)]
    "unet_roads_sp": [(r, "unet", "best.pt") for r in S("unet_roads_sp")],
    "unet_roads_clk": [(f"unet_roads_clk_s{i}", "unetc", "best.pt") for i in range(3)],
    "toposam_roads_scratch": [(r, "sam", "last.pt") for r in S("sam_roads_scratch")],
    "toposam_roads_fs": [(r, "sam", "last.pt") for r in ["sam_roads_fs", "sam_roads_fs_s1", "sam_roads_fs_s2"]],
    "toposam_roads_fs_tex": [(f"sam_roads_fs_tex_s{i}", "sam", "last.pt") for i in range(3)],
}


def save(run, mets, pairs):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{run}.npz", **{f"{i}_{f}": v for i, P in enumerate(pairs) for f, v in P.items()})
    (OUT / f"{run}.json").write_text(json.dumps(mets), encoding="utf-8")


def pair_stats(P):
    y, det = P["y"], P["det"] > 0.5
    return {"pair_err": float((det != y).mean()), "breaks": float((~det & y).mean()), "merges": float((det & ~y).mean())}


def run_drive(dev, T):
    te = load("test")
    for fam, runs in DRIVE.items():
        for r in runs:
            ck = Path("runs") / r / "best.pt"
            if (OUT / f"{r}.json").exists() or not ck.exists():
                continue
            m = UNet().to(dev); m.load_state_dict(torch.load(ck, map_location=dev))
            mets, pairs = [], []
            for i in range(len(te["lab"])):
                gt, fov = te["lab"][i] > 0, te["fov"][i] > 0
                p_det, p_mc = infer(m, te["img"][i], T, dev)
                P = pair_scores(gt, fov, p_det, p_mc, seed=i)
                pairs.append(P); mets.append({**all_metrics(p_det, gt, fov), **pair_stats(P)})
            save(r, mets, pairs); print(r, json.dumps({k: round(float(np.mean([x[k] for x in mets])), 4) for k in mets[0]}), flush=True)


def run_roads(dev, T):
    d = {"calib": load_roads("calib"), "test": load_roads("test")}
    enc = TopoSAM().to(dev).eval()
    doms = {k: Domain(enc, v, dev, seed=7 + j, flips=1) for j, (k, v) in enumerate(d.items())}
    for fam, runs in ROADS.items():
        for r, kind, ckn in runs:
            ck = Path("runs") / r / ckn
            if (OUT / f"{r}.json").exists() or not ck.exists():
                continue
            if kind == "sam":
                m = TopoSAM().to(dev); m.load_state_dict(torch.load(ck, map_location=dev), strict=False)
            else:
                m = UNet(cin=5 if kind == "unetc" else 3).to(dev); m.load_state_dict(torch.load(ck, map_location=dev))
            mets, pairs = [], []
            for split in ("calib", "test"):
                dd, dom = d[split], doms[split]
                for i in range(dom.N):
                    gt, fov = dd["lab"][i] > 0, dd["fov"][i] > 0
                    if kind == "sam":
                        p_det, p_mc = dom.predict(m, i, T)
                    else:
                        x = dd["img"][i]
                        if kind == "unetc":   # exactly the clicks TopoSAM receives
                            pts, lab = (torch.from_numpy(a)[None].to(dev) for a in dom.eval_prompts[i])
                            x = np.concatenate([x, heatmaps(pts.float(), lab, *x.shape[1:])[0].cpu().numpy()])
                        p_det, p_mc = infer(m, x, T, dev)
                    P = pair_scores(gt, fov, p_det, p_mc, seed=(0 if split == "calib" else 100) + i)
                    pairs.append(P); mets.append({"split": split, **all_metrics(p_det, gt, fov), **pair_stats(P)})
            save(r, mets, pairs)
            tm = [x for x in mets if x["split"] == "test"]
            print(r, json.dumps({k: round(float(np.mean([x[k] for x in tm])), 4) for k in tm[0] if k != "split"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["drive", "roads"], required=True)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these run dirs (smoke tests)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.only:
        DRIVE = {k: [r for r in v if r in a.only] for k, v in DRIVE.items()}
        ROADS = {k: [r for r in v if r[0] in a.only] for k, v in ROADS.items()}
    (run_drive if a.group == "drive" else run_roads)(a.device, a.T if a.group == "drive" else min(a.T, 10))
