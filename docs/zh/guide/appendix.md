# 附录 / Appendix

## 1. 完整目录结构 / Complete Directory Layout

```
human_and_plant_few_shot/
├── model/                                 # 模型定义 (10 .py) / Model defs
│   ├── mrmodn.py                          # 主网络 / Main network
│   ├── mrmodn_collect_atten.py            # 收集注意力变体
│   ├── mrmodn_multirm.py                  # MultIRM 51nt 变体
│   ├── multirm.py                         # 多任务多修饰变体
│   ├── modx.py                            # 修饰类型消融变体
│   ├── modx_collect_atten.py              # ModX 注意力收集
│   ├── multirm_collect_atten.py           # MultiRM 注意力收集
│   ├── evormd_human.py                    # EvoRMD 集成
│   ├── abla_model.py                      # 消融基线
│   └── __init__.py
│
├── dataset/                               # 数据集加载 (9 .py) / Dataset loaders
│   ├── human.py                           # Mer100Dataset
│   ├── plant.py                           # PlantDataset
│   ├── ac4c.py                            # AC4CDataset
│   ├── multirm.py                         # MultiRMDataset
│   ├── gen3.py                            # Gen3Dataset
│   ├── gen3_zero.py                       # Gen3ZeroDataset
│   ├── human_motif.py
│   ├── human_with_seq.py
│   └── plant_single.py
│
├── utils/                                 # 工具 (20 .py) / Utilities
│   ├── common.py                          # 常量 + 评估 / Constants + eval
│   ├── metrics.py                         # ACC/F1/MCC
│   ├── logging.py                         # 日志 / Logging
│   ├── few_shot.py                        # 少样本核心 / Few-shot core
│   ├── fewshot_analysis_constants.py
│   ├── fewshot_analysis_features.py
│   ├── fewshot_analysis_fewshot.py
│   ├── fewshot_analysis_gen3.py
│   ├── fewshot_analysis_spatial_motif.py
│   ├── fewshot_analysis_utils.py
│   ├── fewshot_analysis_zeroshot.py
│   ├── fewshot_export_helpers.py
│   ├── sliding_window_utils.py            # 滑窗工具
│   ├── train_gen3.py
│   ├── test_gen3_analyse.py
│   ├── rna_visualization.py
│   ├── audit_12loc_structure.py
│   ├── check_m6a_data_integrity.py
│   ├── Zero_structures.py
│   └── __init__.py
│
├── 根目录脚本 / Root scripts (命名规范: <task>_<dataset>_<model>[_<suffix>].py)
│   ├── train_human_mrmodn.py
│   ├── train_plant_mrmodn.py
│   ├── train_multirm_mrmodn.py
│   ├── train_human_multirm.py
│   ├── train_human_modx.py
│   ├── train_human_evormd.py
│   ├── inference_human_mrmodn_full.py
│   ├── inference_human_modx_segmented.py
│   ├── inference_human_evormd_segmented.py
│   ├── inference_multirm_multirm_segmented.py
│   ├── collect_human_mrmodn.py
│   ├── collect_atten_human_mrmodn.py
│   ├── collect_atten_human_modx.py
│   ├── collect_atten_multirm_multirm.py
│   ├── fewshot_ac4c_mrmodn_balance.py
│   ├── fewshot_ac4c_mrmodn_unbalan.py
│   ├── fewshot_plant_mrmodn_3way.py
│   ├── zeroshot_human_mrmodn_analysis.py
│   └── zeroshot_human_mrmodn_extract.py
│
├── ablation/                              # 消融实验 (5 .py) / Ablation
│   ├── ablation_3x3_human_mrmodn.py
│   ├── ablation_3x3_v2_human_mrmodn.py
│   ├── ablation_mohe_human_mrmodn.py
│   ├── cal_flops_human_mrmodn.py
│   ├── cal_stats_human_mrmodn.py
│   └── README.md
│
├── visualization/                         # 可视化 (14 .py) / Viz
│   ├── human/mrmodn/                      # 8 脚本 / scripts
│   ├── human/modx/                        # 1 脚本
│   ├── tools/                             # 3 工具 / tools
│   ├── notebooks/                         # Jupyter
│   ├── fig/ att_fig/ motif_logo/          # 静态图 (gitignored)
│   └── README.md
│
├── analysis/                              # R 脚本 / R scripts
│   ├── plot_zero_fewshot_analysis.R
│   ├── plot_zero_fewshot_export.R
│   └── Rplots.pdf
│
├── tests/                                 # pytest (10 .py) / Tests
│   ├── conftest.py
│   ├── __init__.py
│   ├── test_infrastructure.py
│   ├── test_model_import.py
│   ├── test_dataset_import.py
│   ├── test_config.py
│   ├── test_naming_convention.py
│   ├── test_gen3.py
│   ├── test_multirm_4class.py
│   └── test_multirm_oversampling.py
│
├── docs/                                  # 本教程 (VitePress) / This tutorial
│   ├── .vitepress/config.ts
│   ├── package.json
│   ├── public/logo.svg
│   ├── zh/{index,guide/overview,architecture,quickstart,model,dataset,training,inference,visualization,ablation,fewshot,appendix}.md
│   └── en/{index,guide/overview,architecture,quickstart,model,dataset,training,inference,visualization,ablation,fewshot,appendix}.md
│
├── pytest.ini                             # pytest 配置 / pytest config
├── README.md                              # 项目说明 (双语)
├── npy -> /home/dc/vscode/npyForTrain    # 数据软链接 (不可移动)
├── json/  output/  logs/  logs_abla/     # 训练产物
└── .omo/  .omc/                           # 计划与状态
```

## 2. 命名规范 / Naming Convention

所有根目录 Python 脚本必须遵循：

All root-level Python scripts must follow:

```
<task>_<dataset>_<model>[_<suffix>].py
```

| 段 / Segment | 取值 / Values | 含义 / Meaning |
|---|---|---|
| `<task>` | `train`, `inference`, `collect`, `fewshot`, `zeroshot` | 任务类型 |
| `<dataset>` | `human`, `plant`, `multirm`, `ac4c` | 数据集 |
| `<model>` | `mrmodn`, `modx`, `multirm`, `evormd` | 模型 |
| `<suffix>` | (可选 / optional) `full`, `segmented`, `balance`, `unbalan`, `3way`, `analysis`, `extract` | 细分 |

**示例 / Examples**:
- ✅ `train_human_mrmodn.py`
- ✅ `inference_human_mrmodn_full.py`
- ✅ `fewshot_plant_mrmodn_3way.py`
- ❌ `train.py` (缺少 dataset / model)
- ❌ `main.py` (命名不规范)

命名规范由 `tests/test_naming_convention.py` 自动审计。

Naming convention is auto-audited by `tests/test_naming_convention.py`.

## 3. 常见问题 FAQ

**Q: npy 软链接断了怎么办？**
A: 重新建立：`ln -s /path/to/npyForTrain npy`。注意 `npy` 是只读软链接，**不要移动或删除数据**。

**Q: 训练 Loss 一直 NaN？**
A: 1) 降低学习率到 5e-4；2) 检查数据中是否有非法字符（如 `N` 以外）；3) 关闭混合精度。

**Q: 如何导出 ONNX？**
A: 见 `model/mrmodn.py` 的 `export_onnx()` 方法，或运行后端项目的转换脚本。

**Q: GPU 显存不足？**
A: 1) 减小 `batch_size`；2) 减小 `gcn_hidden_dim`；3) 启用 `torch.cuda.amp`。

**Q: 如何在自定义数据集上训练？**
A: 1) 实现 `torch_geometric.data.Dataset` 子类；2) 创建 `train_<dataset>_<model>.py` 仿照 `train_human_mrmodn.py`。

**Q: ABS 采样器怎么调？**
A: 调整 `beta`（0.999-0.9999，越大越平滑）和初始 F1。`beta=0` 退化为纯随机。

**Q: 测试失败？**
A: 运行 `pytest tests/ -v` 查看具体错误。常见原因：环境缺少 `torch_geometric`、模型导入路径错误。

## 4. 引用格式 / Citation

如果你在研究中使用了 mRModN，请引用：

If you use mRModN in your research, please cite:

```bibtex
@article{mrmodn2025,
  title={mRModN: Mixture of Hierarchical Experts with Self-Adaptive Balanced Sampling for RNA Modification Prediction Across Species},
  author={RGCNFormer Project Team},
  journal={Nucleic Acids Research},
  year={2025},
  note={Manuscript ID: NAR-2025-mRModN}
}
```

## 5. 致谢 / Acknowledgments

- 数据来源 / Data sources: RMBase, MODOMICS, RNAcentral
- 模型架构参考 / Model architecture references: RGCN, Transformer
- 训练框架 / Training framework: PyTorch + PyTorch Geometric
- 文档工具 / Documentation: VitePress

## 6. Web 部署 / Web Deployment

训练好的模型可由 `Cluster_WebAndWx_backend` 加载并对外提供 HTTP API：

Trained models can be loaded by `Cluster_WebAndWx_backend` to expose HTTP APIs:

- 项目主页 / Project home: https://cmb.bnu.edu.cn/rgcnformer/
- API 文档 / API docs: 见后端仓库 README

## 7. 贡献指南 / Contributing

欢迎提交 Pull Request。在提交前请确保：

Pull requests are welcome. Before submitting, please ensure:

1. `pytest tests/ -v` 全部通过
2. 命名规范合规（`test_naming_convention.py`）
3. 添加/更新相应函数的双语 docstring
4. 更新 README.md（如新增文件）

## 8. 许可 / License

本项目仅供学术研究使用。

This project is for academic research only.

## 9. 联系方式 / Contact

- GitHub Issues: 项目仓库 Issues 区
- Email: rgcnformer@cmb.bnu.edu.cn
- Web: https://cmb.bnu.edu.cn/rgcnformer/
