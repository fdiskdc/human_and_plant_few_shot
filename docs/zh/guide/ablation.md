# 消融实验 / Ablation Studies

## 1. 目的 / Purpose

通过**逐步移除/替换** mRModN 的关键组件，定量评估每个模块对最终性能的贡献。

By **incrementally removing/replacing** key components of mRModN, we quantitatively evaluate the contribution of each module to the final performance.

## 2. 消融脚本总览 / Ablation Script Overview

| 脚本 / Script | 消融对象 / Ablation Target | 输出 / Output |
|---|---|---|
| `ablation_3x3_human_mrmodn.py` | 3×3 矩阵（结构嵌入/池化策略 × 损失） | `output/ablation/3x3.csv` |
| `ablation_3x3_v2_human_mrmodn.py` | 3×3 v2（扩展为 5×5） | `output/ablation/3x3_v2.csv` |
| `ablation_mohe_human_mrmodn.py` | MoHE 单独消融 | `output/ablation/mohe.csv` |
| `cal_flops_human_mrmodn.py` | FLOPs / 参数量 | `output/ablation/flops.csv` |
| `cal_stats_human_mrmodn.py` | 训练/推理耗时统计 | `output/ablation/stats.csv` |

## 3. 3×3 矩阵消融 / 3×3 Matrix Ablation

**文件 / File**: `ablation/ablation_3x3_human_mrmodn.py`

3×3 矩阵的**行**表示池化策略，**列**表示损失组合：

The 3×3 matrix has **rows** for pooling strategy and **columns** for loss combination:

|         | L_cls | L_cls + L_group | L_cls + L_kl |
|---------|-------|-----------------|--------------|
| mean    | ✓     | ✓               | ✓            |
| max     | ✓     | ✓               | ✓            |
| MoHE    | ✓     | ✓               | ✓            |

```bash
python ablation/ablation_3x3_human_mrmodn.py --epochs 20
# 输出: output/ablation/3x3_results.csv
```

**结果解读 / Reading the results**:
- 行内对比 → 不同池化策略的影响
- 列内对比 → 不同损失项的影响
- 右上角 (MoHE + L_cls + L_kl) 应为最佳 / Top-right should be best

## 4. 3×3 v2 扩展 / 3×3 v2 Extension

**文件 / File**: `ablation/ablation_3x3_v2_human_mrmodn.py`

5×5 矩阵，加入更多维度（结构嵌入、GCN 层数、CNN 核组合）：

5×5 matrix adding more dimensions (structure embedding, GCN layers, CNN kernel combos):

| 维度 / Dim | 取值 / Values |
|---|---|
| CNN 核组合 | {(1,3), (1,3,5), (1,3,5,7)} |
| GCN 层数 | {1, 2, 3, 4, 5} |
| 结构嵌入 | {with, without} |
| 池化策略 | {mean, max, MoHE} |
| ABS 采样 | {with, without} |

```bash
python ablation/ablation_3x3_v2_human_mrmodn.py
# 输出: output/ablation/3x3_v2_results.csv
```

## 5. MoHE 消融 / MoHE Ablation

**文件 / File**: `ablation/ablation_mohe_human_mrmodn.py`

逐步消融 MoHE 各子模块：

Step-by-step ablation of MoHE sub-modules:

| 配置 / Config | HQR | HCA | 4 组损失 | F1 预期 |
|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | 0.78 |
| + HQR | ✓ | ✗ | ✗ | 0.80 |
| + HQR + HCA | ✓ | ✓ | ✗ | 0.83 |
| **完整 mRModN** | ✓ | ✓ | ✓ | **0.86** |

## 6. FLOPs 与参数量 / FLOPs & Parameters

**文件 / File**: `ablation/cal_flops_human_mrmodn.py`

```bash
python ablation/cal_flops_human_mrmodn.py
```

**输出示例 / Sample output**:

```
Model: RNA_ClassQuery_Model (full)
Total params: 1.23M
Trainable params: 1.23M
FLOPs (per sample, 1001nt): 4.56 G
GPU memory (batch=32): 3.2 GB
Inference time (1000 samples): 1.05 s
```

**各子模块 FLOPs 占比 / FLOPs breakdown**:

| 子模块 / Sub-module | FLOPs (G) | 占比 / % |
|---|---|---|
| ParallelCNNBlock | 0.45 | 9.9% |
| GCNBlock (3 层) | 1.23 | 27.0% |
| ClassQueryHead | 2.34 | 51.3% |
| HierarchicalClassQueryHeadPooling | 0.54 | 11.8% |

## 7. 训练/推理统计 / Training/Inference Stats

**文件 / File**: `ablation/cal_stats_human_mrmodn.py`

```bash
python ablation/cal_stats_human_mrmodn.py
```

**输出 / Output**:

| 阶段 / Phase | 时间 / Time | 显存 / Memory |
|---|---|---|
| 数据加载 | 0.12 s/epoch | - |
| 前向传播 | 0.45 s/epoch | 2.1 GB |
| 反向传播 | 0.78 s/epoch | 4.5 GB |
| 优化器步进 | 0.08 s/epoch | - |
| **总训练时间** | **1.43 s/epoch** | **峰值 4.8 GB** |
| 推理（1000 样本） | 1.05 s | 1.2 GB |

## 8. 如何运行完整消融 / How to Run All Ablations

```bash
# 1. 3×3 矩阵
python ablation/ablation_3x3_human_mrmodn.py --epochs 20

# 2. 3×3 v2
python ablation/ablation_3x3_v2_human_mrmodn.py

# 3. MoHE 单独消融
python ablation/ablation_mohe_human_mrmodn.py

# 4. FLOPs
python ablation/cal_flops_human_mrmodn.py

# 5. 训练/推理统计
python ablation/cal_stats_human_mrmodn.py
```

## 9. 结果表格 / Results Table Template

实验结果应汇总为如下表格（参考）：

| 方法 / Method | ACC | F1 (macro) | F1 (m6A) | F1 (m5C) | FLOPs (G) |
|---|---|---|---|---|---|
| Baseline (mean pool) | 0.78 | 0.76 | 0.85 | 0.72 | 2.1 |
| + HQR | 0.80 | 0.78 | 0.86 | 0.74 | 2.5 |
| + HQR + HCA | 0.83 | 0.81 | 0.88 | 0.78 | 4.0 |
| **mRModN (full)** | **0.86** | **0.84** | **0.90** | **0.81** | 4.6 |

## 10. 下一步 / Next Steps

- 少样本分析：[少/零样本分析](/guide/fewshot)
- 附录：[附录](/guide/appendix)
