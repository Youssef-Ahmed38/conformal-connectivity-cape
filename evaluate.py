"""Phase-1/2 evaluation: pixel/topology metrics + path-level connectivity uncertainty + conformal guarantees.

Two calibration protocols for the conformal sets:
  split   (exchangeable, main): 100 random image-level splits of the 20 DRIVE test images into 10 calib / 10 test.
  holdout (distribution shift): calibrate on the 4 held-out training images, test on all 20 test images.
Scores are recalibrated to P(connected) with isotonic regression fitted on the calibration pairs (monotone,
so conformal sets are unchanged; ECE and the error score become meaningful).

    python evaluate.py --runs dice cldice cape --out results/phase1 [--device cpu]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccape.conformal import Isotonic, auroc, calibrate, ece, error_score, evaluate_sets, predict_sets
from ccape.data import load
from ccape.infer import infer
from ccape.metrics import all_metrics
from ccape.model import UNet
from ccape.topo_uq import pair_scores

SCORES = ("det", "bn", "mc", "mcbn")
SET_KEYS = ("coverage", "cov_connected", "cov_disconnected", "flagged", "auto_error", "errors_caught")


def collect(model, d, T, dev, seed0):
    pairs, mets, probs = [], [], []
    for n, (img, gt, fov) in enumerate(zip(d["img"], d["lab"] > 0, d["fov"] > 0)):
        p_det, p_mc = infer(model, img, T, dev)
        mets.append(all_metrics(p_det, gt, fov))
        pairs.append(pair_scores(gt, fov, p_det, p_mc, seed=seed0 + n))
        probs.append((p_det, p_mc.std(0)))
    return pairs, mets, probs


def cat(ps):
    return {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}


def assess(cal, tst, s, alphas):
    """Fit isotonic + conformal on cal, evaluate on tst (dicts of pair arrays) for score s."""
    iso = Isotonic().fit(cal[s], cal["y"])
    p = iso(tst[s])
    err = tst["det"].astype(bool) != tst["y"]
    out = {"ece_raw": ece(tst[s], tst["y"]), "ece": ece(p, tst["y"]), "auroc_err": auroc(error_score(p, tst["det"]), err)}
    for al in alphas:
        out[al] = evaluate_sets(predict_sets(tst[s], calibrate(cal[s], cal["y"], al)), tst["y"], tst["det"])
    return out


def mean_std(rs, alphas):
    agg = {}
    for k in ("ece_raw", "ece", "auroc_err"):
        v = np.array([r[k] for r in rs]); agg[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    for al in alphas:
        agg[al] = {k: (float(np.nanmean([r[al][k] for r in rs])), float(np.nanstd([r[al][k] for r in rs]))) for k in SET_KEYS}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["dice", "cldice", "cape"])
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.05])
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dev, out = a.device, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal_d, te = load("calib"), load("test")
    res, viz = {}, {}
    for r in a.runs:
        torch.manual_seed(0)
        model = UNet().to(dev)
        model.load_state_dict(torch.load(f"runs/{r}/best.pt", map_location=dev))
        c_list, _, _ = collect(model, cal_d, a.T, dev, 1000)
        t_list, t_mets, t_probs = collect(model, te, a.T, dev, 0)
        np.savez_compressed(out / f"pairs_{r}.npz", **{f"{sp}_{i}_{f}": v for sp, L in (("calib", c_list), ("test", t_list))
                                                       for i, P in enumerate(L) for f, v in P.items()})
        cal, tst = cat(c_list), cat(t_list)
        R = {"pixel_topo": {k: float(np.mean([m[k] for m in t_mets])) for k in t_mets[0]},
             "n_pairs_test": int(len(tst["y"])), "test_disconnected_frac": float(1 - tst["y"].mean()),
             "det_error_rate": float((tst["det"].astype(bool) != tst["y"]).mean()),
             "break_rate": float(((tst["det"] < 0.5) & tst["y"]).mean()),
             "merge_rate": float(((tst["det"] > 0.5) & ~tst["y"]).mean())}
        rng = np.random.default_rng(0)
        for s in SCORES:
            hold = assess(cal, tst, s, a.alphas)
            sp = []
            for _ in range(a.splits):
                perm = rng.permutation(len(t_list))
                sp.append(assess(cat([t_list[i] for i in perm[:10]]), cat([t_list[i] for i in perm[10:]]), s, a.alphas))
            R[s] = {"holdout": {k: (v if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                                for k, v in hold.items()}, "split": mean_std(sp, a.alphas)}
        res[r] = R
        viz[r] = (t_probs[0], t_list[0], calibrate(cat(t_list[10:])["mcbn"], cat(t_list[10:])["y"], 0.1))
        print(r, json.dumps(R["pixel_topo"]), "split a=0.1:", json.dumps(R["mcbn"]["split"][0.1]), flush=True)
    (out / "phase1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    write_md(res, a.alphas, out / "phase1.md")
    figures(te, viz, out)


def write_md(res, alphas, path):
    f = lambda ms: f"{ms[0]:.3f} ± {ms[1]:.3f}"
    L = ["## Pixel & topology metrics (DRIVE test, 20 images)", "",
         "| Model | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Betti-1 err ↓ | Pair topology error ↓ | breaks | false merges |",
         "|---|---|---|---|---|---|---|---|"]
    for r, R in res.items():
        m = R["pixel_topo"]
        L.append(f"| {r} | {m['dice']:.4f} | {m['cldice']:.4f} | {m['betti0']:.3f} | {m['betti1']:.3f} | "
                 f"{R['det_error_rate']:.4f} | {R['break_rate']:.4f} | {R['merge_rate']:.4f} |")
    L += ["", "## Error detection & calibration of path-level connectivity (split protocol, mean ± std over 100 splits)", "",
          "| Model | Score | AUROC(error) ↑ | ECE raw ↓ | ECE after isotonic ↓ |", "|---|---|---|---|---|"]
    for r, R in res.items():
        for s in SCORES:
            S = R[s]["split"]
            L.append(f"| {r} | {s} | {f(S['auroc_err'])} | {f(S['ece_raw'])} | {f(S['ece'])} |")
    for al in alphas:
        L += ["", f"## Conformal connectivity, alpha = {al} (target ≥ {1 - al:.0%} per class) — split protocol", "",
              "| Model | Score | Cov. connected | Cov. disconnected | Flagged for review ↓ | Error of auto decisions ↓ | Topology errors caught ↑ |",
              "|---|---|---|---|---|---|---|"]
        for r, R in res.items():
            for s in SCORES:
                c = R[s]["split"][al]
                L.append(f"| {r} | {s} | {f(c['cov_connected'])} | {f(c['cov_disconnected'])} | {f(c['flagged'])} | "
                         f"{f(c['auto_error'])} | {f(c['errors_caught'])} |")
    L += ["", "## Same, calibrated on held-out *training* images (distribution shift), alpha = 0.1", "",
          "| Model | Score | Cov. connected | Cov. disconnected | Flagged | Auto error | Errors caught |", "|---|---|---|---|---|---|---|"]
    for r, R in res.items():
        for s in ("bn", "mcbn"):
            c = R[s]["holdout"][alphas[0]]
            L.append(f"| {r} | {s} | {c['cov_connected']:.3f} | {c['cov_disconnected']:.3f} | {c['flagged']:.3f} | "
                     f"{c['auto_error']:.4f} | {c['errors_caught']:.3f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(te, viz, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for r, ((p_det, p_std), P, q) in viz.items():
        sets = predict_sets(P["mcbn"], q)
        auto = sets.sum(1) == 1
        wrong = auto & (sets[:, 1] != P["y"])
        fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
        img = te["img"][0].transpose(1, 2, 0); img = (img - img.min()) / (np.ptp(img) + 1e-9)
        ax[0].imshow(img); ax[0].set_title("Input + GT"); ax[0].contour(te["lab"][0], [0.5], colors="w", linewidths=0.3)
        ax[1].imshow(p_std, cmap="magma"); ax[1].set_title("MC-Dropout std (pixel uncertainty)")
        ax[2].imshow(p_det, cmap="gray")
        for m, col, lw in ((auto & ~wrong, "lime", 0.4), (~auto, "red", 1.2), (wrong, "cyan", 1.6)):
            for (y0, x0), (y1, x1) in zip(P["u"][m], P["v"][m]):
                ax[2].plot([x0, x1], [y0, y1], color=col, lw=lw, alpha=0.8)
        ax[2].set_title(f"{r}: green=trusted, red=flagged for review, cyan=missed error")
        for a_ in ax: a_.axis("off")
        plt.tight_layout(); plt.savefig(out / f"fig_{r}.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
