## 组件贡献分析

| Model | Mamba | HIFM | DSCA | AP | AP$_{50}$ | ΔAP |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR (Baseline) | ✗ | ✗ | ✗ | 0.185 | 0.302 | +0.000 |
| RT-DETR + Mamba | ✓ | ✗ | ✗ | 0.188 | 0.305 | +0.003 |
| RT-DETR + HIFM | ✗ | ✓ | ✗ | 0.200 | 0.354 | +0.015 |
| RT-DETR + DSCA | ✗ | ✗ | ✓ | 0.296 | 0.450 | +0.111 |
| RT-DETR + Mamba + HIFM | ✓ | ✓ | ✗ | 0.320 | 0.481 | +0.135 |
| RT-DETR + Mamba + DSCA | ✓ | ✗ | ✓ | 0.320 | 0.481 | +0.135 |
| RT-DETR + HIFM + DSCA | ✗ | ✓ | ✓ | 0.320 | 0.481 | +0.135 |
| RT-DETR + Mamba + HIFM + DSCA | ✓ | ✓ | ✓ | 0.323 | 0.489 | +0.138 |

