## Phase 3: TopoSAM — segmentation

| Model | Data | Dice ↑ | clDice ↑ | Betti-0 err ↓ | Betti-1 err ↓ |
|---|---|---|---|---|---|
| unet_cape | drive_test | 0.8177 | 0.8174 | 1.072 | 0.287 |
| unet_cape | roads_test | 0.0463 | 0.0416 | 2.922 | 0.095 |
| unet_roads_sp | drive_test | 0.1367 | 0.1121 | 1.762 | 0.437 |
| unet_roads_sp | roads_test | 0.5196 | 0.5919 | 1.021 | 0.078 |
| sam_vanilla | drive_test | 0.1619 | 0.1677 | 2.973 | 2.401 |
| sam_vanilla | roads_test | 0.2373 | 0.2615 | 1.932 | 1.144 |
| toposam_drive | drive_test | 0.8203 | 0.8173 | 1.190 | 0.320 |
| toposam_drive | roads_test | 0.1926 | 0.1789 | 5.519 | 0.407 |
| toposam_roads_scratch | drive_test | 0.2423 | 0.1893 | 2.115 | 0.519 |
| toposam_roads_scratch | roads_test | 0.4857 | 0.6188 | 1.005 | 0.135 |
| toposam_roads_fs | drive_test | 0.3378 | 0.2843 | 3.003 | 0.944 |
| toposam_roads_fs | roads_test | 0.5343 | 0.6588 | 0.709 | 0.094 |

## Roads: path-level topology errors (60 test tiles)

| Model | Pair topology error ↓ | breaks | false merges |
|---|---|---|---|
| unet_cape | 0.9007 | 0.9007 | 0.0000 |
| unet_roads_sp | 0.2614 | 0.2418 | 0.0196 |
| sam_vanilla | 0.2914 | 0.2356 | 0.0558 |
| toposam_drive | 0.7883 | 0.7862 | 0.0021 |
| toposam_roads_scratch | 0.2202 | 0.1550 | 0.0652 |
| toposam_roads_fs | 0.2212 | 0.1676 | 0.0536 |

## Roads: uncertainty + conformal — 100 random tile-level 20/60 splits (exchangeable)

| Model | Score | AUROC(error) ↑ | ECE iso ↓ | α | Cov. conn. | Cov. disc. | Flagged ↓ | Auto error ↓ | Errors caught ↑ |
|---|---|---|---|---|---|---|---|---|---|
| unet_cape | mcbn | 0.590 ± 0.021 | 0.155 ± 0.212 | 0.1 | 1.000 ± 0.000 | 0.926 ± 0.024 | 0.889 ± 0.018 | 0.053 ± 0.013 | 1.000 ± 0.000 |
| unet_cape | mcbn | 0.590 ± 0.021 | 0.155 ± 0.212 | 0.05 | 1.000 ± 0.000 | 0.946 ± 0.030 | 0.910 ± 0.024 | 0.045 ± 0.017 | 1.000 ± 0.000 |
| unet_roads_sp | mcbn | 0.944 ± 0.005 | 0.023 ± 0.011 | 0.1 | 0.902 ± 0.024 | 0.882 ± 0.059 | 0.303 ± 0.072 | 0.145 ± 0.034 | 0.642 ± 0.065 |
| unet_roads_sp | mcbn | 0.944 ± 0.005 | 0.023 ± 0.011 | 0.05 | 0.953 ± 0.011 | 0.929 ± 0.039 | 0.440 ± 0.083 | 0.090 ± 0.025 | 0.823 ± 0.031 |
| sam_vanilla | bn | 0.877 ± 0.014 | 0.025 ± 0.016 | 0.1 | 0.922 ± 0.030 | 0.894 ± 0.080 | 0.852 ± 0.069 | 0.579 ± 0.151 | 0.730 ± 0.091 |
| sam_vanilla | bn | 0.877 ± 0.014 | 0.025 ± 0.016 | 0.05 | 0.966 ± 0.024 | 0.922 ± 0.068 | 0.918 ± 0.053 | 0.497 ± 0.165 | 0.874 ± 0.077 |
| toposam_drive | mcbn | 0.813 ± 0.047 | 0.083 ± 0.086 | 0.1 | 1.000 ± 0.000 | 0.887 ± 0.047 | 0.751 ± 0.044 | 0.036 ± 0.009 | 0.997 ± 0.000 |
| toposam_drive | mcbn | 0.813 ± 0.047 | 0.083 ± 0.086 | 0.05 | 1.000 ± 0.000 | 0.946 ± 0.026 | 0.819 ± 0.032 | 0.024 ± 0.007 | 0.998 ± 0.000 |
| toposam_roads_scratch | mcbn | 0.855 ± 0.063 | 0.034 ± 0.022 | 0.1 | 0.901 ± 0.026 | 0.990 ± 0.056 | 0.885 ± 0.085 | 0.888 ± 0.124 | 0.589 ± 0.094 |
| toposam_roads_scratch | mcbn | 0.855 ± 0.063 | 0.034 ± 0.022 | 0.05 | 0.980 ± 0.033 | 1.000 ± 0.000 | 0.980 ± 0.033 | 0.928 ± 0.008 | 0.921 ± 0.131 |
| toposam_roads_fs | mcbn | 0.881 ± 0.060 | 0.035 ± 0.022 | 0.1 | 0.899 ± 0.035 | 0.988 ± 0.054 | 0.870 ± 0.112 | 0.836 ± 0.151 | 0.596 ± 0.131 |
| toposam_roads_fs | mcbn | 0.881 ± 0.060 | 0.035 ± 0.022 | 0.05 | 0.993 ± 0.025 | 1.000 ± 0.000 | 0.992 ± 0.026 | 0.886 ± 0.015 | 0.972 ± 0.096 |

## Roads: same, fixed calib block (tiles 40-59) -> test block (60-119): spatial shift, alpha = 0.1

| Model | Cov. conn. | Cov. disc. | Flagged | Auto error |
|---|---|---|---|---|
| unet_cape | 1.000 | 0.901 | 0.876 | 0.0654 |
| unet_roads_sp | 0.924 | 0.833 | 0.254 | 0.1113 |
| sam_vanilla | 0.945 | 0.838 | 0.851 | 0.4258 |
| toposam_drive | 1.000 | 0.812 | 0.681 | 0.0482 |
| toposam_roads_scratch | 0.904 | 0.644 | 0.389 | 0.1914 |
| toposam_roads_fs | 0.936 | 1.000 | 0.935 | 0.8979 |
