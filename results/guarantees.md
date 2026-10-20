## Image-level Conformal Risk Control (provable: E[wrongly auto-accepted fraction per image] ≤ α)

Random image splits (DRIVE 16 calib / 4 test, roads 40 / 40), 100 repetitions, loss cap B = 0.5 (never binding: see last column).

| Model | Score | α | Realised risk (mean ± sd) | Trusted ↑ | Flagged ↓ | Topology errors flagged ↑ | cap binding |
|---|---|---|---|---|---|---|---|
| drive/dice | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/dice | bn | 0.1 | 0.0190 ± 0.0055 | 0.663 | 0.337 | 0.629 | 0.000 |
| drive/dice | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/dice | mc | 0.1 | 0.0132 ± 0.0041 | 0.626 | 0.374 | 0.772 | 0.000 |
| drive/dice | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/dice | mcbn | 0.1 | 0.0394 ± 0.0111 | 0.721 | 0.279 | 0.532 | 0.000 |
| drive/cldice | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cldice | bn | 0.1 | 0.0183 ± 0.0065 | 0.644 | 0.356 | 0.755 | 0.000 |
| drive/cldice | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cldice | mc | 0.1 | 0.0173 ± 0.0047 | 0.639 | 0.361 | 0.774 | 0.000 |
| drive/cldice | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cldice | mcbn | 0.1 | 0.0351 ± 0.0163 | 0.688 | 0.312 | 0.659 | 0.000 |
| drive/cape | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cape | bn | 0.1 | 0.0195 ± 0.0061 | 0.675 | 0.325 | 0.787 | 0.000 |
| drive/cape | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cape | mc | 0.1 | 0.0201 ± 0.0061 | 0.679 | 0.321 | 0.763 | 0.000 |
| drive/cape | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive/cape | mcbn | 0.1 | 0.0204 ± 0.0061 | 0.680 | 0.320 | 0.762 | 0.000 |
| drive-sparse/sp_none | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_none | bn | 0.1 | 0.0217 ± 0.0076 | 0.444 | 0.556 | 0.870 | 0.000 |
| drive-sparse/sp_none | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_none | mc | 0.1 | 0.0215 ± 0.0074 | 0.435 | 0.565 | 0.890 | 0.000 |
| drive-sparse/sp_none | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_none | mcbn | 0.1 | 0.0222 ± 0.0080 | 0.448 | 0.552 | 0.862 | 0.000 |
| drive-sparse/sp_random | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_random | bn | 0.1 | 0.0194 ± 0.0051 | 0.542 | 0.458 | 0.869 | 0.000 |
| drive-sparse/sp_random | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_random | mc | 0.1 | 0.0196 ± 0.0045 | 0.545 | 0.455 | 0.864 | 0.000 |
| drive-sparse/sp_random | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_random | mcbn | 0.1 | 0.0205 ± 0.0048 | 0.549 | 0.451 | 0.852 | 0.000 |
| drive-sparse/sp_unc | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_unc | bn | 0.1 | 0.0202 ± 0.0060 | 0.538 | 0.462 | 0.918 | 0.000 |
| drive-sparse/sp_unc | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_unc | mc | 0.1 | 0.0214 ± 0.0047 | 0.551 | 0.449 | 0.914 | 0.000 |
| drive-sparse/sp_unc | mcbn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| drive-sparse/sp_unc | mcbn | 0.1 | 0.0217 ± 0.0049 | 0.552 | 0.448 | 0.913 | 0.000 |
| roads/unet_roads_sp | bn | 0.05 | 0.0140 ± 0.0074 | 0.631 | 0.369 | 0.941 | 0.000 |
| roads/unet_roads_sp | bn | 0.1 | 0.0375 ± 0.0130 | 0.777 | 0.223 | 0.606 | 0.000 |
| roads/unet_roads_sp | mc | 0.05 | 0.0120 ± 0.0095 | 0.459 | 0.541 | 0.950 | 0.000 |
| roads/unet_roads_sp | mc | 0.1 | 0.0271 ± 0.0056 | 0.724 | 0.276 | 0.779 | 0.000 |
| roads/unet_roads_sp | mcbn | 0.05 | 0.0253 ± 0.0080 | 0.664 | 0.336 | 0.893 | 0.000 |
| roads/unet_roads_sp | mcbn | 0.1 | 0.0764 ± 0.0159 | 0.849 | 0.151 | 0.412 | 0.000 |
| roads/toposam_roads_scratch | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| roads/toposam_roads_scratch | bn | 0.1 | 0.0376 ± 0.0193 | 0.632 | 0.368 | 0.847 | 0.000 |
| roads/toposam_roads_scratch | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| roads/toposam_roads_scratch | mc | 0.1 | 0.0024 ± 0.0139 | 0.024 | 0.976 | 0.990 | 0.000 |
| roads/toposam_roads_scratch | mcbn | 0.05 | 0.0036 ± 0.0099 | 0.060 | 0.940 | 0.985 | 0.000 |
| roads/toposam_roads_scratch | mcbn | 0.1 | 0.0430 ± 0.0215 | 0.666 | 0.334 | 0.826 | 0.000 |
| roads/toposam_roads_fs | bn | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| roads/toposam_roads_fs | bn | 0.1 | 0.0348 ± 0.0220 | 0.565 | 0.435 | 0.843 | 0.000 |
| roads/toposam_roads_fs | mc | 0.05 | 0.0000 ± 0.0000 | 0.000 | 1.000 | 1.000 | 0.000 |
| roads/toposam_roads_fs | mc | 0.1 | 0.0109 ± 0.0251 | 0.130 | 0.870 | 0.948 | 0.000 |
| roads/toposam_roads_fs | mcbn | 0.05 | 0.0077 ± 0.0106 | 0.179 | 0.821 | 0.966 | 0.000 |
| roads/toposam_roads_fs | mcbn | 0.1 | 0.0409 ± 0.0161 | 0.706 | 0.294 | 0.812 | 0.000 |

## Covariate-shift-weighted conformal (calibration from a shifted source, α = 0.1, target ≥ 0.90 per class)

DRIVE: calibrate on held-out *training* images → test images. Roads: tile block 40-59 → block 60-119.

| Model | Score | Unweighted cov. conn. / disc. | **Weighted** cov. conn. / disc. | Flagged (unw → w) | ESS |
|---|---|---|---|---|---|
| drive/dice | bn | 0.833 / 0.934 | 0.951 / 0.951 | 0.085 → 0.510 | 10870 |
| drive/dice | mcbn | 0.820 / 0.934 | 0.940 / 0.948 | 0.073 → 0.484 | 10870 |
| drive/cldice | bn | 0.818 / 0.942 | 0.946 / 0.945 | 0.088 → 0.460 | 11134 |
| drive/cldice | mcbn | 0.819 / 0.940 | 0.936 / 0.947 | 0.081 → 0.446 | 11134 |
| drive/cape | bn | 0.820 / 0.942 | 0.978 / 0.976 | 0.066 → 0.736 | 7936 |
| drive/cape | mcbn | 0.818 / 0.939 | 0.977 / 0.975 | 0.053 → 0.722 | 7936 |
| drive-sparse/sp_none | bn | 1.000 / 0.910 | 1.000 / 0.970 | 0.594 → 0.821 | 4838 |
| drive-sparse/sp_none | mcbn | 1.000 / 0.908 | 1.000 / 0.969 | 0.588 → 0.806 | 4838 |
| drive-sparse/sp_random | bn | 1.000 / 0.933 | 1.000 / 0.957 | 0.505 → 0.705 | 11439 |
| drive-sparse/sp_random | mcbn | 0.922 / 0.927 | 0.953 / 0.954 | 0.379 → 0.619 | 11439 |
| drive-sparse/sp_unc | bn | 1.000 / 0.953 | 1.000 / 0.971 | 0.537 → 0.727 | 10567 |
| drive-sparse/sp_unc | mcbn | 0.891 / 0.932 | 0.931 / 0.955 | 0.337 → 0.552 | 10567 |
| roads/unet_roads_sp | bn | 0.949 / 0.866 | 0.949 / 0.866 | 0.318 → 0.318 | 39967 |
| roads/unet_roads_sp | mcbn | 0.924 / 0.833 | 0.905 / 0.844 | 0.254 → 0.244 | 39967 |
| roads/toposam_roads_scratch | bn | 0.922 / 1.000 | 0.904 / 1.000 | 0.923 → 0.904 | 40667 |
| roads/toposam_roads_scratch | mcbn | 0.904 / 0.644 | 0.899 / 0.699 | 0.389 → 0.431 | 40667 |
| roads/toposam_roads_fs | bn | 1.000 / 1.000 | 0.902 / 1.000 | 1.000 → 0.895 | 38339 |
| roads/toposam_roads_fs | mcbn | 0.936 / 1.000 | 0.893 / 1.000 | 0.935 → 0.886 | 38339 |
