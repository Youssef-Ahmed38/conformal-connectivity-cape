## Final results (3 seeds, mean ± std; paired Wilcoxon over test images)

### Phase 1 — DRIVE

| Model | seeds | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Path topology err ↓ | breaks ↓ | AUROC(err) | CRC risk | CRC trusted | CRC errors flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE + Dice | 3 | 0.825 ± 0.001 | 0.830 ± 0.001 | 1.043 ± 0.096 | 0.227 ± 0.004 | 0.215 ± 0.005 | 0.827 ± 0.023 | 0.024 ± 0.001 | 0.684 ± 0.003 | 0.595 ± 0.022 |
| + clDice | 3 | 0.820 ± 0.001 | 0.843 ± 0.000 | 0.742 ± 0.013 | 0.215 ± 0.008 | 0.203 ± 0.007 | 0.839 ± 0.047 | 0.034 ± 0.007 | 0.708 ± 0.015 | 0.616 ± 0.013 |
| + CAPE | 3 | 0.818 ± 0.001 | 0.817 ± 0.001 | 1.044 ± 0.024 | 0.179 ± 0.002 | 0.162 ± 0.002 | 0.741 ± 0.016 | 0.024 ± 0.001 | 0.693 ± 0.002 | 0.720 ± 0.017 |

| Comparison | metric | Δ (A − B) | Wilcoxon p (images) | seeds A better |
|---|---|---|---|---|
| cape vs dice | dice | -0.0070 | 1.9e-05 ** | 0/3 |
| cape vs dice | cldice | -0.0124 | 0.0001 ** | 0/3 |
| cape vs dice | pair_err | -0.0479 | 1.9e-06 ** | 3/3 |
| cape vs dice | breaks | -0.0523 | 1.9e-06 ** | 3/3 |
| cldice vs dice | cldice | +0.0134 | 1.9e-06 ** | 3/3 |
| cldice vs dice | pair_err | -0.0112 | 4.8e-05 ** | 2/3 |
| cape vs cldice | cldice | -0.0259 | 1.9e-06 ** | 0/3 |
| cape vs cldice | pair_err | -0.0368 | 1.9e-06 ** | 3/3 |
| cape vs cldice | breaks | -0.0409 | 1.9e-06 ** | 3/3 |

### Phase 2 — label-efficient (DRIVE)

| Model | seeds | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Path topology err ↓ | breaks ↓ | AUROC(err) | CRC risk | CRC trusted |
|---|---|---|---|---|---|---|---|---|---|
| scribbles + CAPE | 3 | 0.636 ± 0.029 | 0.710 ± 0.015 | 1.029 ± 0.079 | 0.470 ± 0.028 | 0.454 ± 0.029 | 0.801 ± 0.020 | 0.022 ± 0.001 | 0.443 ± 0.011 |
| + random clicks | 3 | 0.700 ± 0.002 | 0.797 ± 0.003 | 1.003 ± 0.035 | 0.339 ± 0.017 | 0.324 ± 0.018 | 0.792 ± 0.026 | 0.022 ± 0.000 | 0.552 ± 0.011 |
| + uncertainty clicks | 3 | 0.690 ± 0.003 | 0.791 ± 0.005 | 0.904 ± 0.011 | 0.299 ± 0.016 | 0.271 ± 0.016 | 0.820 ± 0.020 | 0.025 ± 0.003 | 0.532 ± 0.030 |
| + unc. + diversity (r=24) | 3 | 0.701 ± 0.001 | 0.807 ± 0.001 | 0.976 ± 0.027 | 0.304 ± 0.014 | 0.285 ± 0.016 | 0.811 ± 0.009 | 0.023 ± 0.000 | 0.563 ± 0.005 |
| + unc. + diversity (r=12, 1 seed) | 1 | 0.702 | 0.811 | 0.899 | 0.279 | 0.256 | 0.835 | 0.023 | 0.571 |
| + unc. + diversity (r=48, selected on val.) | 3 | 0.700 ± 0.000 | 0.808 ± 0.001 | 0.932 ± 0.033 | 0.311 ± 0.011 | 0.293 ± 0.012 | 0.819 ± 0.029 | 0.022 ± 0.001 | 0.560 ± 0.007 |
| r=48, **noisy annotator** (2 px jitter, 10 % wrong answers) | 3 | 0.619 ± 0.013 | 0.734 ± 0.008 | 1.087 ± 0.035 | 0.445 ± 0.017 | 0.426 ± 0.018 | 0.787 ± 0.018 | 0.022 ± 0.001 | 0.432 ± 0.011 |

| Comparison | metric | Δ (A − B) | Wilcoxon p (images) | seeds A better |
|---|---|---|---|---|
| sp_random vs sp_none | cldice | +0.0866 | 1.9e-06 ** | 3/3 |
| sp_random vs sp_none | pair_err | -0.1304 | 1.9e-06 ** | 3/3 |
| sp_unc vs sp_random | cldice | -0.0061 | 3.6e-05 ** | 0/3 |
| sp_unc vs sp_random | pair_err | -0.0407 | 8.8e-05 ** | 3/3 |
| sp_unc vs sp_random | breaks | -0.0526 | 1.9e-06 ** | 3/3 |
| sp_unc_div_r48 vs sp_random | cldice | +0.0114 | 3.8e-06 ** | 3/3 |
| sp_unc_div_r48 vs sp_random | pair_err | -0.0285 | 3.8e-06 ** | 3/3 |
| sp_unc_div_r48 vs sp_random | breaks | -0.0312 | 3.8e-06 ** | 3/3 |
| sp_unc_div_r48 vs sp_unc | cldice | +0.0174 | 1.9e-06 ** | 3/3 |
| sp_unc_div_r48 vs sp_unc | pair_err | +0.0121 | 0.0056 ** | 0/3 |
| sp_unc_div_noisy vs sp_unc_div_r48 | cldice | -0.0747 | 1.9e-06 ** | 0/3 |
| sp_unc_div_noisy vs sp_unc_div_r48 | pair_err | +0.1340 | 1.9e-06 ** | 0/3 |
| sp_unc_div_noisy vs sp_none | cldice | +0.0232 | 3.8e-06 ** | 3/3 |
| sp_unc_div_noisy vs sp_none | pair_err | -0.0249 | 0.00039 ** | 3/3 |

### Phase 3 — roads few-shot (40 tiles, scribbles only)

| Model | seeds | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Path topology err ↓ | breaks ↓ | AUROC(err) | conf. flagged | CRC risk | CRC trusted |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net + CAPE, scribbles | 3 | 0.554 ± 0.025 | 0.637 ± 0.033 | 0.794 ± 0.163 | 0.252 ± 0.010 | 0.230 ± 0.010 | 0.942 ± 0.001 | 0.319 ± 0.020 | 0.075 ± 0.001 | 0.832 ± 0.009 |
| U-Net + CAPE, scribbles **+ same clicks as TopoSAM** | 3 | 0.566 ± 0.018 | 0.638 ± 0.025 | 0.899 ± 0.263 | 0.224 ± 0.013 | 0.197 ± 0.013 | 0.940 ± 0.001 | 0.314 ± 0.021 | 0.076 ± 0.001 | 0.844 ± 0.005 |
| TopoSAM from SAM | 3 | 0.495 ± 0.007 | 0.624 ± 0.004 | 1.005 ± 0.054 | 0.216 ± 0.002 | 0.153 ± 0.004 | 0.884 ± 0.009 | 0.730 ± 0.182 | 0.053 ± 0.016 | 0.693 ± 0.033 |
| TopoSAM from DRIVE-TopoSAM | 3 | 0.551 ± 0.012 | 0.682 ± 0.019 | 0.619 ± 0.077 | 0.227 ± 0.005 | 0.178 ± 0.009 | 0.887 ± 0.019 | 0.844 ± 0.059 | 0.039 ± 0.001 | 0.720 ± 0.029 |
| TopoSAM from texture/polarity-invariant DRIVE-TopoSAM | 3 | 0.545 ± 0.009 | 0.664 ± 0.013 | 0.691 ± 0.060 | 0.229 ± 0.008 | 0.175 ± 0.010 | 0.879 ± 0.011 | 0.865 ± 0.022 | 0.038 ± 0.001 | 0.707 ± 0.023 |

| Comparison | metric | Δ (A − B) | Wilcoxon p (images) | seeds A better |
|---|---|---|---|---|
| toposam_roads_fs vs unet_roads_clk | dice | -0.0147 | 0.0027 ** | 0/3 |
| toposam_roads_fs vs unet_roads_clk | cldice | +0.0435 | 2.5e-08 ** | 3/3 |
| toposam_roads_fs vs unet_roads_clk | betti0 | -0.2802 | 7.1e-11 ** | 3/3 |
| toposam_roads_fs vs unet_roads_clk | pair_err | +0.0025 | 0.94 | 1/3 |
| toposam_roads_fs vs toposam_roads_scratch | dice | +0.0565 | 2.3e-10 ** | 3/3 |
| toposam_roads_fs vs toposam_roads_scratch | cldice | +0.0580 | 2.2e-09 ** | 3/3 |
| toposam_roads_fs vs toposam_roads_scratch | betti0 | -0.3859 | 1.6e-10 ** | 3/3 |
| toposam_roads_fs_tex vs toposam_roads_fs | dice | -0.0066 | 0.019 ** | 0/3 |
| toposam_roads_fs_tex vs toposam_roads_fs | cldice | -0.0176 | 2.7e-05 ** | 0/3 |
| toposam_roads_fs_tex vs toposam_roads_fs | betti0 | +0.0717 | 4.8e-05 ** | 0/3 |
| toposam_roads_fs_tex vs toposam_roads_fs | pair_err | +0.0026 | 0.58 | 1/3 |
| unet_roads_clk vs unet_roads_sp | dice | +0.0122 | 4.8e-07 ** | 2/3 |
| unet_roads_clk vs unet_roads_sp | cldice | +0.0013 | 0.6 | 2/3 |

### Zero-shot vessels → roads (30 test tiles; no road labels, label-free test-time selection)

| Model | test-time rule | Dice | clDice | Betti-0 err |
|---|---|---|---|---|
| + polarity training | fixed inv=False s=1.0 | 0.205 | 0.187 | 6.77 |
| + polarity training | mean-ens (both polarities, s=0.5) | 0.357 | 0.391 | 1.24 |
| + polarity training | min-entropy (per tile) | 0.220 | 0.197 | 5.85 |
| + polarity training | min-ent-ens (2 lowest) | 0.264 | 0.258 | 3.29 |
| + polarity + RandConv texture training | fixed inv=False s=1.0 | 0.265 | 0.222 | 7.82 |
| + polarity + RandConv texture training | mean-ens (both polarities, s=0.5) | 0.351 | 0.373 | 1.38 |
| + polarity + RandConv texture training | min-entropy (per tile) | 0.277 | 0.234 | 6.36 |
| + polarity + RandConv texture training | min-ent-ens (2 lowest) | 0.306 | 0.276 | 3.83 |
