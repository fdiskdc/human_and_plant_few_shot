# Visualization

> RGCNFormer (mRModN) 可视化目录 / RGCNFormer (mRModN) visualization directory

按 **数据集 → 模型** 二级组织所有可视化脚本、Jupyter notebooks、图片和静态资源。
Organized by **dataset → model** for all visualization scripts, Jupyter notebooks,
images, and static assets.

## 目录结构 / Directory Structure

```
visualization/
├── human/                      # human (12 类 m6A/m1A/...) 数据集 / human dataset
│   ├── mrmodn/                 # mRModN 主模型可视化 / mRModN main model viz
│   │   ├── view_data.py        ← view_human.py (原)
│   │   ├── view_total.py       ← view_human_total.py
│   │   ├── view_v3.py          ← view3.py
│   │   ├── attention_comparison.py        ← visualize_attention_comparison.py
│   │   ├── run_attention_comparison.py    ← run_attention_comparison.py
│   │   ├── run_attention_comparison_v2.py ← atten_comp/run_attention_comparison_v2.py
│   │   ├── visualize_attention_comparison_v2.py
│   │   ├── select_representatives.py      ← select_representative_sequences.py
│   │   ├── spatial_motif.py    ← SpatialMotif.py
│   │   └── spatial_motif_nobg.py ← SpatialMotif_nobackground.py
│   ├── modx/                   # modX 模型可视化 / modX model viz
│   │   └── inference_full.py   ← atten_comp/inference_modx_full.py
│   ├── evormd/                 # evoRMD 模型可视化 / evoRMD model viz
│   └── multirm/                # MultiRM 模型可视化 / MultiRM model viz
│
├── plant/                      # plant (m6A/m1A/m5C) 数据集 / plant dataset
│   ├── mrmodn/                 # mRModN
│   ├── modx/
│   ├── evormd/
│   └── multirm/
│
├── multirm/                    # MultiRM benchmark (51nt 滑窗) / MultiRM benchmark
│   ├── mrmodn/
│   ├── modx/
│   ├── evormd/
│   └── multirm/
│
├── ac4c/                       # ac4C 少样本数据集 / ac4C few-shot dataset
│   ├── mrmodn/
│   ├── modx/
│   └── evormd/
│
├── gen3/                       # Gen3 零样本数据集 / Gen3 zero-shot dataset
│   ├── mrmodn/
│   ├── modx/
│   └── evormd/
│
├── tools/                      # 通用工具 (跨数据集) / generic tools (cross-dataset)
│   ├── view_npz.py             ← view_npz.py
│   ├── prepare_umap_data.py    ← prepare_umap_data.py
│   └── prepare_umap_from_npz.py ← prepare_umap_from_npz.py
│
├── notebooks/                  # Jupyter notebooks
│   └── view_gen3.ipynb         ← view_gen3.ipynb (root)
│
├── fig/                        # 主图片 / main figures (PNG)
├── att_fig/                    # 注意力图片 / attention figures (PNG)
└── motif_logo/                 # motif logo (PDF, gitignored)
```

## 使用 / Usage

每个数据集子目录下的脚本独立可运行。从项目根目录执行：
Scripts in each dataset subdirectory run independently. From project root:

```bash
# 示例: human 数据集 mRModN 模型的空间 motif 可视化
uv run python visualization/human/mrmodn/spatial_motif.py

# 工具脚本
uv run python visualization/tools/view_npz.py path/to/data.npz
```

## 添加新可视化 / Adding new visualizations

1. 选择对应的 `数据集/模型/` 子目录 / Pick the right `dataset/model/` subdirectory
2. 命名遵循 `任务_描述.py` (无需重复数据集和模型) / Name as `task_description.py`
3. 更新此 README 表格 / Update this README

## 相关目录 / Related Directories

- `../ablation/` — 消融实验脚本 / ablation experiments
- `../analysis/` — R 绘图脚本和分析报告 / R plot scripts and analysis reports
- `../tests/` — pytest 测试 / pytest tests
