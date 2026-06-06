# Project Overview

## 1. Research Background

RNA modifications are a central topic in epitranscriptomics. To date, more than **170 distinct chemical modifications** have been identified on various RNA molecules. Among them, 12 well-characterized types — `m6A`, `m5C`, `Ψ`, `ac4C`, `m1A`, `m2G`, etc. — are the most abundant on mRNA and the most deeply studied. These modifications are involved in splicing, nuclear export, translation and degradation, and are closely linked to cancer, neurodevelopmental disorders and metabolic diseases.

## 2. Motivation

Traditional RNA modification detection methods (`iCLIP`, `miCLIP`, `SCARLET`, `MAZTER-seq`) are accurate but have notable limitations:

- **Costly**: Each experiment costs thousands of RMB in reagents and sequencing
- **Low-throughput**: Each run detects a single modification or site
- **Antibody-dependent**: Antibody specificity causes false positives/negatives
- **Limited single-site resolution**

Deep learning enables high-throughput, low-cost prediction from RNA sequence alone, providing a powerful complement to wet-lab assays.

## 3. Position of mRModN

**mRModN (Mixture of Hierarchical Experts with self-adaptive balanced sampling for RNA modification prediction across species)** is the core method of this project. Its key innovations are:

| Module | Full Name | Purpose |
|---|---|---|
| **M2D** | Multi-view Motif Discovery | Multi-scale CNN for local k-mer patterns |
| **MoHE** | Mixture of Hierarchical Experts | Hierarchical experts with query routing |
| **ABS** | Adaptive Balanced Sampler | Adaptive class-balanced sampling |
| **A2P** | Anchor-to-Positive Pooling | Multi-anchor attention pooling |

## 4. Supported 12 Modifications

| Index | Modification | Meaning |
|---|---|---|
| 0 | m6A | N6-methyladenosine |
| 1 | m5C | 5-methylcytosine |
| 2 | m1A | N1-methyladenosine |
| 3 | m2G | N2-methylguanosine |
| 4 | Ψ | Pseudouridine |
| 5 | ac4C | N4-acetylcytosine |
| 6 | m7G | N7-methylguanosine |
| 7 | m3C | 3-methylcytosine |
| 8 | I | Inosine (A-to-I editing) |
| 9 | s2U | 2-thiouridine |
| 10 | D | Dihydrouridine |
| 11 | Nm | 2'-O-methylation |

## 5. Repository Organization

This repository `human_and_plant_few_shot` is the **core training & inference codebase** of the entire RGCNFormer system:

- `model/` — 10 model files (main + 4 variants + ablation)
- `dataset/` — 9 dataset loaders
- `utils/` — 20 utility scripts (including few-shot analysis)
- 20 root-level `train_*.py` / `inference_*.py` / `fewshot_*.py` / `zeroshot_*.py` / `collect_*.py` scripts
- `ablation/` — 5 ablation scripts
- `visualization/` — 14 visualization scripts
- `tests/` — 10 pytest test cases

## 6. Relation to Other Projects

```
rgcnformer_sum (this repo)
    │  train and export ONNX
    ▼
Cluster_WebAndWx_backend  ←── HTTP API entry
    │
    ├──→ Cluster_WebAndWx_WxFrontend (WeChat MiniApp)
    └──→ RGCNFormer_WebAndWx_WebFrontend (Web Frontend)
```

- **Cluster_WebAndWx_backend**: loads `epoch_040.pt` and ONNX models exported from this project, exposes HTTP APIs
- **Cluster_WebAndWx_WxFrontend**: WeChat MiniApp using the model indirectly via the backend
- **RGCNFormer_WebAndWx_WebFrontend**: Web frontend using the model indirectly via the backend

## 7. Next Steps

- Continue to [Algorithm Architecture](/en/guide/architecture) to understand the four innovation modules
- Jump to [Quickstart](/en/guide/quickstart) to run a minimal training pipeline
