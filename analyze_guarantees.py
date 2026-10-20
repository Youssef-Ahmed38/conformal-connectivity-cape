"""Guarantees on cached pair results (no model inference): image-level CRC + covariate-shift-weighted conformal.

    python analyze_guarantees.py            -> results/guarantees.md, results/guarantees.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ccape.conformal import calibrate, evaluate_sets, predict_sets
from ccape.crc import crc_eval, crc_fit, density_ratio, pair_features, weighted_sets

R = Path("results")
ALPHAS = (0.05, 0.1)
SCORES = ("bn", "mc", "mcbn")


def load_pairs(path):
    """npz with keys '{split}_{i}_{field}' -> {split: [dict per image]}"""
    z = np.load(path)
    d = defaultdict(lambda: defaultdict(dict))
    for k in z.files:
        sp, i, f = re.match(r"(.+)_(\d+)_(\w+)$", k).groups()
        d[sp][int(i)][f] = z[k]
    return {sp: [v[i] for i in sorted(v)] for sp, v in d.items()}


def as_img(P, s):
    return {"s": P[s], "y": P["y"].astype(bool), "det": P["det"]}


def crc_splits(images, n_cal, s, alphas, reps=100, seed=0):
    rng, out = np.random.default_rng(seed), {}
    for al in alphas:
        rs = []
        for _ in range(reps):
            p = rng.permutation(len(images))
            l1, l0 = crc_fit([as_img(images[i], s) for i in p[:n_cal]], al)
            rs.append(crc_eval([as_img(images[i], s) for i in p[n_cal:]], l1, l0))
        out[al] = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        out[al]["risk_std"] = float(np.std([r["risk"] for r in rs]))
    return out


def cat(L):
    return {k: np.concatenate([d[k] for d in L]) for k in L[0]}


def shift_experiment(cal_imgs, tst_imgs, s, alpha=0.1):
    c, t = cat(cal_imgs), cat(tst_imgs)
    Xc = np.concatenate([pair_features(d, i) for i, d in enumerate(cal_imgs)])
    Xt = np.concatenate([pair_features(d, i) for i, d in enumerate(tst_imgs)])
    wc, wt = density_ratio(Xc, Xt)
    yc, yt = c["y"].astype(bool), t["y"].astype(bool)
    un = evaluate_sets(predict_sets(t[s], calibrate(c[s], yc, alpha)), yt, t["det"])
    we = evaluate_sets(weighted_sets(c[s], yc, wc, t[s], wt, alpha), yt, t["det"])
    keys = ("cov_connected", "cov_disconnected", "flagged", "auto_error")
    ess = float(wc.sum() ** 2 / (wc ** 2).sum())
    return {"unweighted": {k: float(un[k]) for k in keys}, "weighted": {k: float(we[k]) for k in keys}, "ess": ess}


def main():
    res = {"crc": {}, "shift": {}}
    jobs = [(f"drive/{m}", R / "phase1" / f"pairs_{m}.npz", "test", 16) for m in ("dice", "cldice", "cape")]
    jobs += [(f"drive-sparse/{m}", R / "phase2" / f"pairs_{m}.npz", "test", 16) for m in ("sp_none", "sp_random", "sp_unc")]
    jobs += [(f"roads/{m}", R / "phase3" / f"pairs_{m}.npz", None, 40)
             for m in ("unet_roads_sp", "toposam_roads_scratch", "toposam_roads_fs")]
    for name, path, split, n_cal in jobs:
        if not path.exists():
            continue
        P = load_pairs(path)
        imgs = P[split] if split else P["roads_calib"] + P["roads_test"]
        res["crc"][name] = {s: crc_splits(imgs, n_cal, s, ALPHAS) for s in SCORES}
        if name.startswith("drive/") or name.startswith("drive-sparse/"):
            res["shift"][name] = {s: shift_experiment(P["calib"], P["test"], s) for s in ("bn", "mcbn")}
        elif name.startswith("roads/"):
            res["shift"][name] = {s: shift_experiment(P["roads_calib"], P["roads_test"], s) for s in ("bn", "mcbn")}
        print(name, "CRC mcbn a=0.1:", json.dumps(res["crc"][name]["mcbn"][0.1]), flush=True)
    (R / "guarantees.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    L = ["## Image-level Conformal Risk Control (provable: E[wrongly auto-accepted fraction per image] ≤ α)", "",
         "Random image splits (DRIVE 16 calib / 4 test, roads 40 / 40), 100 repetitions, loss cap B = 0.5 (never binding: see last column).", "",
         "| Model | Score | α | Realised risk (mean ± sd) | Trusted ↑ | Flagged ↓ | Topology errors flagged ↑ | cap binding |",
         "|---|---|---|---|---|---|---|---|"]
    for name, S in res["crc"].items():
        for s in SCORES:
            for al in ALPHAS:
                c = S[s][al]
                L.append(f"| {name} | {s} | {al} | {c['risk']:.4f} ± {c['risk_std']:.4f} | {c['trusted']:.3f} | "
                         f"{c['flagged']:.3f} | {c['errors_flagged']:.3f} | {c['cap_binding']:.3f} |")
    L += ["", "## Covariate-shift-weighted conformal (calibration from a shifted source, α = 0.1, target ≥ 0.90 per class)", "",
          "DRIVE: calibrate on held-out *training* images → test images. Roads: tile block 40-59 → block 60-119.", "",
          "| Model | Score | Unweighted cov. conn. / disc. | **Weighted** cov. conn. / disc. | Flagged (unw → w) | ESS |",
          "|---|---|---|---|---|---|"]
    for name, S in res["shift"].items():
        for s, v in S.items():
            u, w = v["unweighted"], v["weighted"]
            L.append(f"| {name} | {s} | {u['cov_connected']:.3f} / {u['cov_disconnected']:.3f} | "
                     f"{w['cov_connected']:.3f} / {w['cov_disconnected']:.3f} | {u['flagged']:.3f} → {w['flagged']:.3f} | {v['ess']:.0f} |")
    (R / "guarantees.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
