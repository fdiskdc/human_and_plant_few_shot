# Appendix

## 1. Complete Directory Layout

```
human_and_plant_few_shot/
├── model/                                 # Model defs (10 .py)
│   ├── mrmodn.py                          # Main network
│   ├── mrmodn_collect_atten.py            # Collect-attention variant
│   ├── mrmodn_multirm.py                  # MultIRM 51nt variant
│   ├── multirm.py                         # Multi-task multi-mod variant
│   ├── modx.py                            # Mod-type ablation
│   ├── modx_collect_atten.py
│   ├── multirm_collect_atten.py
│   ├── evormd_human.py                    # EvoRMD integration
│   ├── abla_model.py                      # Ablation baseline
│   └── __init__.py
│
├── dataset/                               # Dataset loaders (9 .py)
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
├── utils/                                 # Utilities (20 .py)
│   ├── common.py                          # Constants + eval
│   ├── metrics.py                         # ACC/F1/MCC
│   ├── logging.py
│   ├── few_shot.py                        # Few-shot core
│   ├── fewshot_analysis_constants.py
│   ├── fewshot_analysis_features.py
│   ├── fewshot_analysis_fewshot.py
│   ├── fewshot_analysis_gen3.py
│   ├── fewshot_analysis_spatial_motif.py
│   ├── fewshot_analysis_utils.py
│   ├── fewshot_analysis_zeroshot.py
│   ├── fewshot_export_helpers.py
│   ├── sliding_window_utils.py
│   ├── train_gen3.py
│   ├── test_gen3_analyse.py
│   ├── rna_visualization.py
│   ├── audit_12loc_structure.py
│   ├── check_m6a_data_integrity.py
│   ├── Zero_structures.py
│   └── __init__.py
│
├── Root scripts (naming: <task>_<dataset>_<model>[_<suffix>].py)
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
├── ablation/                              # Ablation (5 .py)
│   ├── ablation_3x3_human_mrmodn.py
│   ├── ablation_3x3_v2_human_mrmodn.py
│   ├── ablation_mohe_human_mrmodn.py
│   ├── cal_flops_human_mrmodn.py
│   ├── cal_stats_human_mrmodn.py
│   └── README.md
│
├── visualization/                         # Viz (14 .py)
│   ├── human/mrmodn/                      # 8 scripts
│   ├── human/modx/                        # 1 script
│   ├── tools/                             # 3 tools
│   ├── notebooks/                         # Jupyter
│   ├── fig/ att_fig/ motif_logo/          # Static figs (gitignored)
│   └── README.md
│
├── analysis/                              # R scripts
│   ├── plot_zero_fewshot_analysis.R
│   ├── plot_zero_fewshot_export.R
│   └── Rplots.pdf
│
├── tests/                                 # pytest (10 .py)
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
├── docs/                                  # This tutorial (VitePress)
│   ├── .vitepress/config.ts
│   ├── package.json
│   ├── public/logo.svg
│   ├── zh/{index,guide/overview,architecture,quickstart,model,dataset,training,inference,visualization,ablation,fewshot,appendix}.md
│   └── en/{index,guide/overview,architecture,quickstart,model,dataset,training,inference,visualization,ablation,fewshot,appendix}.md
│
├── pytest.ini
├── README.md
├── npy -> /home/dc/vscode/npyForTrain    # Data symlink (do not move)
├── json/  output/  logs/  logs_abla/
└── .omo/  .omc/
```

## 2. Naming Convention

All root-level Python scripts must follow:

```
<task>_<dataset>_<model>[_<suffix>].py
```

| Segment | Values | Meaning |
|---|---|---|
| `<task>` | `train`, `inference`, `collect`, `fewshot`, `zeroshot` | Task type |
| `<dataset>` | `human`, `plant`, `multirm`, `ac4c` | Dataset |
| `<model>` | `mrmodn`, `modx`, `multirm`, `evormd` | Model |
| `<suffix>` | optional `full`, `segmented`, `balance`, `unbalan`, `3way`, `analysis`, `extract` | Detail |

**Examples**:
- ✅ `train_human_mrmodn.py`
- ✅ `inference_human_mrmodn_full.py`
- ✅ `fewshot_plant_mrmodn_3way.py`
- ❌ `train.py` (missing dataset/model)
- ❌ `main.py` (non-conforming)

Naming is auto-audited by `tests/test_naming_convention.py`.

## 3. FAQ

**Q: npy symlink broken?**
A: Re-create: `ln -s /path/to/npyForTrain npy`. `npy` is a read-only symlink — **do not move or delete the data**.

**Q: Loss is NaN?**
A: 1) Lower learning rate to 5e-4; 2) check for invalid characters; 3) disable mixed precision.

**Q: How to export ONNX?**
A: See `export_onnx()` in `model/mrmodn.py`, or run the conversion script in the backend repo.

**Q: GPU OOM?**
A: 1) Reduce `batch_size`; 2) reduce `gcn_hidden_dim`; 3) enable `torch.cuda.amp`.

**Q: How to train on a custom dataset?**
A: 1) Implement a `torch_geometric.data.Dataset` subclass; 2) create `train_<dataset>_<model>.py` after `train_human_mrmodn.py`.

**Q: How to tune ABS?**
A: Adjust `beta` (0.999-0.9999, larger = smoother) and initial F1. `beta=0` falls back to uniform random.

**Q: Tests failing?**
A: Run `pytest tests/ -v` for details. Common causes: missing `torch_geometric`, wrong model import path.

## 4. Citation

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

## 5. Acknowledgments

- Data sources: RMBase, MODOMICS, RNAcentral
- Model architecture references: RGCN, Transformer
- Training framework: PyTorch + PyTorch Geometric
- Documentation: VitePress

## 6. Web Deployment

Trained models can be loaded by `Cluster_WebAndWx_backend` to expose HTTP APIs:

- Project home: https://cmb.bnu.edu.cn/rgcnformer/
- API docs: see backend repo README

## 7. Contributing

Pull requests are welcome. Before submitting, please ensure:

1. `pytest tests/ -v` all pass
2. Naming convention compliance (`test_naming_convention.py`)
3. Bilingual docstrings added/updated for new/changed functions
4. README.md updated (when adding files)

## 8. License

This project is for academic research only.

## 9. Contact

- GitHub Issues: project repo Issues tab
- Email: rgcnformer@cmb.bnu.edu.cn
- Web: https://cmb.bnu.edu.cn/rgcnformer/
