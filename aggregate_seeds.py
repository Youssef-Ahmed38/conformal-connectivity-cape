"""Aggregate cached multi-seed results (results/seeds/*) into results/final.md / final.json.

Statistics
  * every metric: mean ± std over seeds (each seed = mean over test images)
  * method comparisons: paired two-sided Wilcoxon signed-rank test over test images (values averaged over seeds),
    + the fraction of seeds in which the difference has the same sign
  * uncertainty / guarantees per seed (cached pairs): AUROC(error), ECE after isotonic, Mondrian conformal at
    alpha=0.1 (random image splits) and image-level CRC at alpha=0.1, then mean ± std over seeds
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_guarantees import crc_splits
from eval_seeds import DRIVE, ROADS
from evaluate import assess, cat

SD = Path("results/seeds")
METRICS = ("dice", "cldice", "betti0", "betti1", "pair_err", "breaks", "merges")
LOWER = {"betti0", "betti1", "pair_err", "breaks", "merges"}


def load_run(r, split=None):
    if not (SD / f"{r}.json").exists():
        return None, None
    mets = json.loads((SD / f"{r}.json").read_text(encoding="utf-8"))
    z = np.load(SD / f"{r}.npz")
    pairs = defaultdict(dict)
    for k in z.files:
        i, f = re.match(r"(\d+)_(\w+)$", k).groups(); pairs[int(i)][f] = z[k]
    pairs = [pairs[i] for i in sorted(pairs)]
    if split:
        keep = [j for j, m in enumerate(mets) if m.get("split") == split]
        mets, pairs = [mets[j] for j in keep], [pairs[j] for j in keep]
    return mets, pairs


def family(runs, split=None, all_pairs=False):
    out = []
    for r in runs:
        name = r[0] if isinstance(r, tuple) else r
        m, p = load_run(name, split)
        if m is not None:
            pp = load_run(name)[1] if all_pairs else p
            out.append((name, m, p, pp))
    return out


def seed_table(fam):
    per_seed = np.array([[np.mean([m[k] for m in mets]) for k in METRICS] for _, mets, _, _ in fam])
    per_img = np.mean([[[m[k] for k in METRICS] for m in mets] for _, mets, _, _ in fam], 0)   # (images, metrics)
    return per_seed, per_img


def uq(fam, n_cal_crc, reps=50):
    """AUROC/ECE/conformal (random 50/50 image splits) + CRC per seed; mean ± std over seeds."""
    rows = []
    for _, _, _, pairs in fam:
        rng, A = np.random.default_rng(0), []
        for _ in range(reps):
            p = rng.permutation(len(pairs)); h = len(pairs) // 2
            A.append(assess(cat([pairs[i] for i in p[:h]]), cat([pairs[i] for i in p[h:]]), "mcbn", (0.1,)))
        c = crc_splits(pairs, n_cal_crc, "mcbn", (0.1,), reps=reps)[0.1]
        rows.append([np.nanmean([a["auroc_err"] for a in A]), np.nanmean([a["ece"] for a in A]),
                     np.nanmean([a[0.1]["cov_connected"] for a in A]), np.nanmean([a[0.1]["cov_disconnected"] for a in A]),
                     np.nanmean([a[0.1]["flagged"] for a in A]), c["risk"], c["trusted"], c["errors_flagged"]])
    return np.array(rows)


UQ_COLS = ("AUROC(err)", "ECE iso", "conf. cov conn", "conf. cov disc", "conf. flagged", "CRC risk", "CRC trusted", "CRC errors flagged")


def fmt(a):
    return f"{np.mean(a):.4f} ± {np.std(a):.4f}" if len(a) > 1 else f"{a[0]:.4f}"


def compare(A, B, nameA, nameB):
    """paired Wilcoxon over images on seed-averaged per-image values; same-sign seeds."""
    (sa, ia), (sb, ib) = A, B
    out = []
    for j, k in enumerate(METRICS):
        d = ia[:, j] - ib[:, j]
        p = wilcoxon(ia[:, j], ib[:, j]).pvalue if np.any(d != 0) else 1.0
        n = min(len(sa), len(sb))
        better = (sa[:n, j] < sb[:n, j]) if k in LOWER else (sa[:n, j] > sb[:n, j])
        out.append((k, float(np.mean(sa[:, j]) - np.mean(sb[:, j])), float(p), f"{int(better.sum())}/{n}"))
    return out


def section(title, fams, comps, n_cal_crc, split=None, all_pairs=False):
    L = [f"## {title}", "", "| Model | seeds | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    T, U, res = {}, {}, {}
    for name, runs in fams.items():
        fam = family(runs, split, all_pairs)
        if not fam:
            continue
        T[name] = seed_table(fam)
        L.append(f"| {name} | {len(fam)} | " + " | ".join(fmt(T[name][0][:, j]) for j in range(len(METRICS))) + " |")
        U[name] = uq(fam, n_cal_crc)
        res[name] = {"seeds": [f[0] for f in fam], "metrics": {k: T[name][0][:, j].tolist() for j, k in enumerate(METRICS)},
                     "uq": {k: U[name][:, j].tolist() for j, k in enumerate(UQ_COLS)}}
    L += ["", "**Uncertainty + guarantees (score = MC × bottleneck, α = 0.1; mean ± std over seeds)**", "",
          "| Model | " + " | ".join(UQ_COLS) + " |", "|---|" + "---|" * len(UQ_COLS)]
    for name, u in U.items():
        L.append(f"| {name} | " + " | ".join(fmt(u[:, j]) for j in range(len(UQ_COLS))) + " |")
    L += ["", "**Paired comparisons** (Δ = A − B of seed means; Wilcoxon p over images; seeds where A is better)", "",
          "| A vs B | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    res["_comparisons"] = {}
    for a, b in comps:
        if a in T and b in T:
            c = compare(T[a], T[b], a, b)
            res["_comparisons"][f"{a} vs {b}"] = c
            L.append(f"| {a} vs {b} | " + " | ".join(f"{d:+.4f} (p={p:.3g}; {s})" for _, d, p, s in c) + " |")
    return L, res


def main():
    out, R = ["# Final multi-seed results", ""], {}
    p1 = {k: DRIVE[k] for k in ("dice", "cldice", "cape")}
    L, R["phase1"] = section("Phase 1 — DRIVE test (20 images)", p1,
                             [("cape", "dice"), ("cldice", "dice"), ("cape", "cldice")], 16)
    out += L + [""]
    p2 = {k: DRIVE[k] for k in DRIVE if k.startswith("sp_")}
    L, R["phase2"] = section("Phase 2 — label-efficient, DRIVE test (20 images)", p2,
                             [("sp_random", "sp_none"), ("sp_unc", "sp_random"), ("sp_unc_div", "sp_random"),
                              ("sp_unc_div", "sp_unc"), ("sp_unc_div_r48", "sp_unc"), ("sp_unc_div_r48", "sp_random"),
                              ("sp_unc_div_noisy", "sp_unc_div_r48"), ("sp_unc_div_noisy", "sp_none")], 16)
    out += L + [""]
    L, R["phase3"] = section("Phase 3 — roads few-shot (60 test tiles; uncertainty on all 80 tiles)", ROADS,
                             [("toposam_roads_fs", "unet_roads_sp"), ("toposam_roads_fs", "unet_roads_clk"),
                              ("unet_roads_clk", "unet_roads_sp"), ("toposam_roads_fs", "toposam_roads_scratch"),
                              ("toposam_roads_fs_tex", "toposam_roads_fs")], 40, split="test", all_pairs=True)
    out += L
    Path("results/final.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    Path("results/final.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
