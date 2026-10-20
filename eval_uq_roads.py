"""TopoSAM uncertainty on roads: MC-Dropout only vs prompt ensemble (+ MC-Dropout), optionally a decoder ensemble.
Same pair protocol as eval_phase3; reports AUROC(error), and split-conformal + CRC efficiency (100 random 40/40 splits).
    python eval_uq_roads.py --ckpt runs/sam_roads_fs/last.pt [--ens runs/a/last.pt runs/b/last.pt] --T 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_guarantees import as_img, crc_splits
from ccape.data import load_roads
from ccape.sam import Domain, TopoSAM
from ccape.topo_uq import pair_scores
from evaluate import assess, cat, mean_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/sam_roads_fs/last.pt")
    ap.add_argument("--ens", nargs="*", default=[], help="extra checkpoints: decoder deep ensemble")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--tag", default="fs")
    a = ap.parse_args()
    dev = "cuda"
    d = {k: v for k, v in zip(("c", "t"), (load_roads("calib"), load_roads("test")))}
    models = []
    for ck in [a.ckpt] + a.ens:
        m = TopoSAM().to(dev); m.load_state_dict(torch.load(ck, map_location=dev), strict=False); models.append(m)
    doms = {k: Domain(models[0], v, dev, seed=7 + j, flips=1) for j, (k, v) in enumerate(d.items())}
    modes = {"mc_dropout": dict(prompt_ens=False), "prompt_ens+mc": dict(prompt_ens=True)}
    res = {}
    for mode, kw in modes.items():
        imgs = []
        for k in ("c", "t"):
            dom, dd = doms[k], d[k]
            for i in range(dom.N):
                outs = [dom.predict(m, i, a.T // len(models), **kw) for m in models]
                p_det = np.mean([o[0] for o in outs], 0)
                p_mc = np.concatenate([o[1] for o in outs])
                imgs.append(pair_scores(dd["lab"][i] > 0, dd["fov"][i] > 0, p_det, p_mc, seed=100 * len(k) + i))
        rng, sp = np.random.default_rng(0), []
        for _ in range(100):
            p = rng.permutation(len(imgs))
            sp.append(assess(cat([imgs[i] for i in p[:40]]), cat([imgs[i] for i in p[40:]]), "mcbn", (0.1,)))
        ms = mean_std(sp, (0.1,))
        crc = crc_splits(imgs, 40, "mcbn", (0.1,))
        res[mode] = {"auroc_err": ms["auroc_err"], "ece": ms["ece"], "conformal_0.1": ms[0.1], "crc_0.1": crc[0.1]}
        print(a.tag, mode, json.dumps({"auroc": ms["auroc_err"][0], "flagged_conf": ms[0.1]["flagged"][0],
                                       "cov": [ms[0.1]["cov_connected"][0], ms[0.1]["cov_disconnected"][0]],
                                       "crc_risk": crc[0.1]["risk"], "crc_trusted": crc[0.1]["trusted"]}), flush=True)
    Path(f"results/uq_roads_{a.tag}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
