# AlphaMotion Benchmark

Release gate: ALL GREEN — 2026-08-15

| metric | value | bar | pass |
|---|---|---|---|
| reencode_fidelity | 0.639 | 0.50 | ✅ |
| follow_score | 0.416 | 0.30 | ✅ |
| amplitude_ratio(info) | 1.170 | 0.00 | ✅ |
| atlas_precision_x | 7.100 | 5.00 | ✅ |
| bridge_excess_nll (sampling ctl -0.52) | 0.967 | 1.50 | ✅ |
| retime_agreement | 0.866 | 0.50 | ✅ |
| synergy_pass_rate | 0.611 | 0.60 | ✅ |

## Synergy ratio by body (12 library clips)

| body | median | min | pass rate |
|---|---|---|---|
| unitree_h1 | 2.03 | 0.01 | 58% |
| booster_t1 | 0.94 | 0.01 | 58% |
| fourier_gr3 | 1.48 | 0.02 | 67% |

Every number is GT-free and reproducible from the packaged artifacts: `alphamotion eval`.