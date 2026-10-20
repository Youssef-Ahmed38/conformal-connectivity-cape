## Pixel & topology metrics (DRIVE test, 20 images)

| Model | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Betti-1 err ↓ | Pair topology error ↓ | breaks | false merges |
|---|---|---|---|---|---|---|---|
| sp_unc_div | 0.7002 | 0.8073 | 0.944 | 0.353 | 0.2846 | 0.2631 | 0.0215 |

## Error detection & calibration of path-level connectivity (split protocol, mean ± std over 100 splits)

| Model | Score | AUROC(error) ↑ | ECE raw ↓ | ECE after isotonic ↓ |
|---|---|---|---|---|
| sp_unc_div | det | 0.547 ± 0.337 | 0.284 ± 0.014 | 0.194 ± 0.115 |
| sp_unc_div | bn | 0.695 ± 0.177 | 0.254 ± 0.016 | 0.102 ± 0.058 |
| sp_unc_div | mc | 0.625 ± 0.290 | 0.273 ± 0.015 | 0.131 ± 0.095 |
| sp_unc_div | mcbn | 0.831 ± 0.053 | 0.248 ± 0.014 | 0.044 ± 0.019 |

## Conformal connectivity, alpha = 0.1 (target ≥ 90% per class) — split protocol

| Model | Score | Cov. connected | Cov. disconnected | Flagged for review ↓ | Error of auto decisions ↓ | Topology errors caught ↑ |
|---|---|---|---|---|---|---|
| sp_unc_div | det | 1.000 ± 0.000 | 0.964 ± 0.057 | 0.836 ± 0.263 | 0.042 ± 0.002 | 0.975 ± 0.041 |
| sp_unc_div | bn | 1.000 ± 0.000 | 0.904 ± 0.024 | 0.453 ± 0.034 | 0.034 ± 0.006 | 0.934 ± 0.018 |
| sp_unc_div | mc | 1.000 ± 0.000 | 0.902 ± 0.021 | 0.445 ± 0.031 | 0.034 ± 0.005 | 0.933 ± 0.016 |
| sp_unc_div | mcbn | 0.910 ± 0.029 | 0.899 ± 0.021 | 0.301 ± 0.035 | 0.131 ± 0.025 | 0.677 ± 0.068 |

## Conformal connectivity, alpha = 0.05 (target ≥ 95% per class) — split protocol

| Model | Score | Cov. connected | Cov. disconnected | Flagged for review ↓ | Error of auto decisions ↓ | Topology errors caught ↑ |
|---|---|---|---|---|---|---|
| sp_unc_div | det | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | nan ± nan | 1.000 ± 0.000 |
| sp_unc_div | bn | 1.000 ± 0.000 | 0.957 ± 0.015 | 0.563 ± 0.032 | 0.019 ± 0.005 | 0.970 ± 0.011 |
| sp_unc_div | mc | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | nan ± nan | 1.000 ± 0.000 |
| sp_unc_div | mcbn | 0.968 ± 0.030 | 0.948 ± 0.014 | 0.491 ± 0.044 | 0.068 ± 0.041 | 0.873 ± 0.081 |

## Same, calibrated on held-out *training* images (distribution shift), alpha = 0.1

| Model | Score | Cov. connected | Cov. disconnected | Flagged | Auto error | Errors caught |
|---|---|---|---|---|---|---|
| sp_unc_div | bn | 1.000 | 0.926 | 0.503 | 0.0291 | 0.949 |
| sp_unc_div | mcbn | 0.854 | 0.928 | 0.269 | 0.1797 | 0.539 |
