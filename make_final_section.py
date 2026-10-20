"""Write results/final_section.md (README-ready) from results/final.json + zero-shot probe JSONs. No hand-copied numbers.
    python make_final_section.py
"""
import json
from pathlib import Path

import numpy as np

R = json.loads(Path("results/final.json").read_text(encoding="utf-8"))
M = ("dice", "cldice", "betti0", "pair_err", "breaks")
NAMES = {"dice": "BCE + Dice", "cldice": "+ clDice", "cape": "+ CAPE", "sp_none": "scribbles + CAPE",
         "sp_random": "+ random clicks", "sp_unc": "+ uncertainty clicks", "sp_unc_div": "+ unc. + diversity (r=24)",
         "sp_unc_div_r12": "+ unc. + diversity (r=12, 1 seed)", "sp_unc_div_r48": "+ unc. + diversity (r=48, selected on val.)",
         "sp_unc_div_noisy": "r=48, **noisy annotator** (2 px jitter, 10 % wrong answers)",
         "unet_roads_sp": "U-Net + CAPE, scribbles", "unet_roads_clk": "U-Net + CAPE, scribbles **+ same clicks as TopoSAM**",
         "toposam_roads_scratch": "TopoSAM from SAM", "toposam_roads_fs": "TopoSAM from DRIVE-TopoSAM",
         "toposam_roads_fs_tex": "TopoSAM from texture/polarity-invariant DRIVE-TopoSAM"}


def ms(v):
    v = np.asarray(v)
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def table(ph, keys, extra=()):
    L = ["| Model | seeds | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Path topology err ↓ | breaks ↓ |" + "".join(f" {e} |" for e in extra),
         "|---|---|---|---|---|---|---|" + "---|" * len(extra)]
    for k in keys:
        if k not in R[ph]:
            continue
        r = R[ph][k]
        row = f"| {NAMES.get(k, k)} | {len(r['seeds'])} | " + " | ".join(ms(r["metrics"][m]) for m in M) + " |"
        for e in extra:
            row += f" {ms(r['uq'][e])} |"
        L.append(row)
    return L


def comps(ph, wanted):
    L = ["| Comparison | metric | Δ (A − B) | Wilcoxon p (images) | seeds A better |", "|---|---|---|---|---|"]
    C = R[ph].get("_comparisons", {})
    for key, metrics in wanted:
        if key not in C:
            continue
        for k, d, p, s in C[key]:
            if k in metrics:
                star = " **" if p < 0.05 else ""
                L.append(f"| {key} | {k} | {d:+.4f} | {p:.2g}{star} | {s} |")
    return L


out = ["## Final results (3 seeds, mean ± std; paired Wilcoxon over test images)", ""]
out += ["### Phase 1 — DRIVE", ""] + table("phase1", ("dice", "cldice", "cape"), ("AUROC(err)", "CRC risk", "CRC trusted", "CRC errors flagged"))
out += [""] + comps("phase1", [("cape vs dice", ("pair_err", "breaks", "cldice", "dice")), ("cldice vs dice", ("pair_err", "cldice")),
                                ("cape vs cldice", ("pair_err", "breaks", "cldice"))])
out += ["", "### Phase 2 — label-efficient (DRIVE)", ""]
out += table("phase2", ("sp_none", "sp_random", "sp_unc", "sp_unc_div", "sp_unc_div_r12", "sp_unc_div_r48", "sp_unc_div_noisy"),
             ("AUROC(err)", "CRC risk", "CRC trusted"))
out += [""] + comps("phase2", [("sp_random vs sp_none", ("cldice", "pair_err")), ("sp_unc vs sp_random", ("cldice", "pair_err", "breaks")),
                                ("sp_unc_div_r48 vs sp_random", ("cldice", "pair_err", "breaks")), ("sp_unc_div_r48 vs sp_unc", ("cldice", "pair_err")),
                                ("sp_unc_div_noisy vs sp_unc_div_r48", ("cldice", "pair_err")), ("sp_unc_div_noisy vs sp_none", ("cldice", "pair_err"))])
out += ["", "### Phase 3 — roads few-shot (40 tiles, scribbles only)", ""]
out += table("phase3", ("unet_roads_sp", "unet_roads_clk", "toposam_roads_scratch", "toposam_roads_fs", "toposam_roads_fs_tex"),
             ("AUROC(err)", "conf. flagged", "CRC risk", "CRC trusted"))
out += [""] + comps("phase3", [("toposam_roads_fs vs unet_roads_clk", ("dice", "cldice", "betti0", "pair_err")),
                                ("toposam_roads_fs vs toposam_roads_scratch", ("dice", "cldice", "betti0")),
                                ("toposam_roads_fs_tex vs toposam_roads_fs", ("dice", "cldice", "betti0", "pair_err")),
                                ("unet_roads_clk vs unet_roads_sp", ("dice", "cldice"))])
zs = [("results/tta_select_sam_drive.json", "TopoSAM-DRIVE"), ("results/tta_select_sam_drive_pol.json", "+ polarity training"),
      ("results/tta_select_sam_drive_tex.json", "+ polarity + RandConv texture training")]
rows = []
for f, n in zs:
    if Path(f).exists():
        z = json.loads(Path(f).read_text(encoding="utf-8"))
        for k in ("fixed inv=False s=1.0", "mean-ens (both polarities, s=0.5)", "min-entropy (per tile)", "min-ent-ens (2 lowest)"):
            if k in z:
                rows.append(f"| {n} | {k} | {z[k]['dice']:.3f} | {z[k]['cldice']:.3f} | {z[k]['betti0']:.2f} |")
if rows:
    out += ["", "### Zero-shot vessels → roads (30 test tiles; no road labels, label-free test-time selection)", "",
            "| Model | test-time rule | Dice | clDice | Betti-0 err |", "|---|---|---|---|---|"] + rows
Path("results/final_section.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
