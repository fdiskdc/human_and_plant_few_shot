# Ablation

> RGCNFormer (mRModN) 消融实验目录 / RGCNFormer (mRModN) ablation experiments

集中存放所有消融实验脚本及对应的 FLOPs/参数统计工具。
Centralized location for all ablation experiment scripts and the corresponding
FLOPs / parameter statistics utilities.

## 脚本清单 / Script List

| 脚本 / Script | 用途 / Purpose |
|---|---|
| `ablation_3x3_human_mrmodn.py` | 3×3 注意力头矩阵消融 (human, mRModN) / 3×3 attention head matrix ablation |
| `ablation_3x3_v2_human_mrmodn.py` | 3×3 矩阵消融 v2 (改进版) / 3×3 matrix ablation v2 |
| `ablation_mohe_human_mrmodn.py` | MoHE (Mixture of Heads with Experts) 消融 / MoHE ablation |
| `cal_flops_human_mrmodn.py` | mRModN 在 MoHE 配置下的 FLOPs 计算 / FLOPs computation for mRModN under MoHE config |
| `cal_stats_human_mrmodn.py` | 统计指标 (均值/中位数/众数) 计算 / Statistics (mean/median/mode) calculator |

## 使用 / Usage

从项目根目录执行 / Run from project root:

```bash
uv run python ablation/ablation_mohe_human_mrmodn.py
uv run python ablation/cal_flops_human_mrmodn.py
```

## 输出 / Outputs

- 训练曲线、混淆矩阵: `logs_abla/` (gitignored)
- 统计 CSV: 脚本同目录或 `logs_abla/csv/`

## 相关目录 / Related Directories

- `../model/abla_model.py` — 消融模型变体定义 / ablation model variants
- `../logs_abla/` — 消融日志输出 / ablation log outputs
- `../visualization/` — 消融结果可视化 / ablation result visualization
