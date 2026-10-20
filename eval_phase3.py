"""Phase-3 evaluation on Massachusetts Roads (calib 20 tiles / test 60 tiles) and DRIVE test.

Models
  unet_cape              phase-1 U-Net + CAPE trained on DRIVE, no prompts                  (zero-shot on roads)
  sam_vanilla            SAM ViT-B as released, same click prompts, no training             (zero-shot)
  toposam_drive          TopoSAM trained on DRIVE (dense + CAPE)                             (zero-shot on roads)
  unet_roads_sp          U-Net trained on the 40 road tiles with scribbles only + CAPE       (few-shot baseline)
  toposam_roads_scratch  TopoSAM (from SAM) fine-tuned on the same 40 tiles / scribbles      (few-shot, no DRIVE)
  toposam_roads_fs       TopoSAM (from toposam_drive) fine-tuned on the same tiles / scribbles (few-shot, ours)
Conformal protocol: the fixed calib (tiles 40-59) / test (60-119) blocks are consecutive filenames and turned out
NOT exchangeable (spatially grouped), so the main protocol pools the 80 tiles and uses 100 random tile-level
20/60 splits (as for DRIVE); the fixed split is kept as a shift result. Scores are isotonic-recalibrated.
    python eval_phase3.py --T 10
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccape.data import load, load_roads
from ccape.infer import infer
from ccape.metrics import all_metrics
from ccape.model import UNet
from ccape.sam import Domain, TopoSAM
from ccape.topo_uq import pair_scores
from evaluate import assess, cat, mean_std

ALPHAS = (0.1, 0.05)


def run_split(pred, d, seed0):
    pairs, mets = [], []
    for i in range(len(d["lab"])):
        gt, fov = d["lab"][i] > 0, d["fov"][i] > 0
        p_det, p_mc = pred(i)
        mets.append(all_metrics(p_det, gt, fov))
        pairs.append(pair_scores(gt, fov, p_det, p_mc if p_mc is not None else p_det[None], seed=seed0 + i))
    return pairs, {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--out", default="results/phase3")
    a = ap.parse_args()
    dev, out = "cuda", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {"roads_calib": load_roads("calib"), "roads_test": load_roads("test"), "drive_test": load("test")}

    sam = TopoSAM(refine=False).to(dev).eval()
    doms = {k: Domain(sam, d, dev, seed=7 + j, flips=1) for j, (k, d) in enumerate(data.items())}

    models = {}
    for name, ck in (("unet_cape", "runs/cape/best.pt"), ("unet_roads_sp", "runs/unet_roads_sp/best.pt")):
        if Path(ck).exists():
            u = UNet().to(dev); u.load_state_dict(torch.load(ck, map_location=dev)); models[name] = ("unet", u)
    models["sam_vanilla"] = ("sam0", sam)
    for name, ck in (("toposam_drive", "runs/sam_drive/best.pt"), ("toposam_roads_scratch", "runs/sam_roads_scratch/last.pt"),
                     ("toposam_roads_fs", "runs/sam_roads_fs/last.pt")):
        if Path(ck).exists():
            m = TopoSAM().to(dev); m.load_state_dict(torch.load(ck, map_location=dev), strict=False); models[name] = ("sam", m)

    res = {}
    for name, (kind, m) in models.items():
        R, lists = {}, {}
        for k, d in data.items():
            if kind == "unet":
                pred = (lambda d: lambda i: infer(m, d["img"][i], a.T, dev))(d)
            else:
                pred = (lambda dom, T: lambda i: dom.predict(m, i, T))(doms[k], a.T if kind == "sam" else 0)
            lists[k], R[k] = run_split(pred, d, 100 * len(k))
        np.savez_compressed(out / f"pairs_{name}.npz", **{f"{k}_{i}_{f}": v for k, L in lists.items()
                                                          for i, P in enumerate(L) for f, v in P.items()})
        pairs = {k: cat(v) for k, v in lists.items()}
        tp = pairs["roads_test"]
        R["pairs_roads"] = {"n": int(len(tp["y"])), "disconnected_frac": float(1 - tp["y"].mean()),
                            "error": float((tp["det"].astype(bool) != tp["y"]).mean()),
                            "breaks": float(((tp["det"] < 0.5) & tp["y"]).mean()),
                            "merges": float(((tp["det"] > 0.5) & ~tp["y"]).mean())}
        s = "mcbn" if kind != "sam0" else "bn"
        A = assess(pairs["roads_calib"], tp, s, ALPHAS)
        R["conformal_roads"] = {"score": s, **{k: float(v) for k, v in A.items() if not isinstance(v, dict)},
                                **{str(al): {kk: float(vv) for kk, vv in A[al].items()} for al in ALPHAS}}
        pool, rng = lists["roads_calib"] + lists["roads_test"], np.random.default_rng(0)
        sp = []
        for _ in range(a.splits):
            perm = rng.permutation(len(pool))
            sp.append(assess(cat([pool[i] for i in perm[:20]]), cat([pool[i] for i in perm[20:]]), s, ALPHAS))
        R["conformal_roads_split"] = {str(k): v for k, v in mean_std(sp, ALPHAS).items()}
        res[name] = R
        print(name, json.dumps({k: R[k] for k in ("roads_test", "pairs_roads")}), flush=True)
    (out / "phase3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, out / "phase3.md")


def write_md(res, path):
    f = lambda ms: f"{ms[0]:.3f} ± {ms[1]:.3f}"
    L = ["## Phase 3: TopoSAM — segmentation", "",
         "| Model | Data | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Betti-1 err ↓ |", "|---|---|---|---|---|---|"]
    for n, R in res.items():
        for k in ("drive_test", "roads_test"):
            m = R[k]
            L.append(f"| {n} | {k} | {m['dice']:.4f} | {m['cldice']:.4f} | {m['betti0']:.3f} | {m['betti1']:.3f} |")
    L += ["", "## Roads: path-level topology errors (60 test tiles)", "",
          "| Model | Pair topology error ↓ | breaks | false merges |", "|---|---|---|---|"]
    for n, R in res.items():
        p = R["pairs_roads"]
        L.append(f"| {n} | {p['error']:.4f} | {p['breaks']:.4f} | {p['merges']:.4f} |")
    L += ["", "## Roads: uncertainty + conformal — 100 random tile-level 20/60 splits (exchangeable)", "",
          "| Model | Score | AUROC(error) ↑ | ECE iso ↓ | α | Cov. conn. | Cov. disc. | Flagged ↓ | Auto error ↓ | Errors caught ↑ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, R in res.items():
        c, sc = R["conformal_roads_split"], R["conformal_roads"]["score"]
        for al in ALPHAS:
            e = c[str(al)]
            L.append(f"| {n} | {sc} | {f(c['auroc_err'])} | {f(c['ece'])} | {al} | {f(e['cov_connected'])} | "
                     f"{f(e['cov_disconnected'])} | {f(e['flagged'])} | {f(e['auto_error'])} | {f(e['errors_caught'])} |")
    L += ["", "## Roads: same, fixed calib block (tiles 40-59) -> test block (60-119): spatial shift, alpha = 0.1", "",
          "| Model | Cov. conn. | Cov. disc. | Flagged | Auto error |", "|---|---|---|---|---|"]
    for n, R in res.items():
        e = R["conformal_roads"]["0.1"]
        L.append(f"| {n} | {e['cov_connected']:.3f} | {e['cov_disconnected']:.3f} | {e['flagged']:.3f} | {e['auto_error']:.4f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
