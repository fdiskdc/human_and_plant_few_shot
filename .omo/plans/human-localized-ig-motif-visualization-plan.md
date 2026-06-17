# Human Localized IG Motif Pipeline Plan

## 当前目标

脚本 `ipynb/motif/human_localized_ig_motif.py` 用于解释 human 12-class RNA modification 模型中，各修饰类别的高 IG 序列窗口，并与 STREME de novo motif 对比。目标是回答：模型对每种 RNA 修饰最关注什么局部序列模式，这些模式是否与 STREME 从 IG top 窗口中发现的 de novo motif 一致。

## 当前数据流

### 1. Dataset 加载

```python
dataset = Mer100Dataset(
    mode='train', data_dir=..., cache_dir=...,
    use_human3=True, use_cache=True, preload_cache=True,
)
```

- 复用 `Mer100Dataset` 的真实 `edge_index`（LinearFold 结构边）和 `full_labels`（1001loc.npy 位点标签）。
- `use_human3=True` 时 `mode` 不切分样本，只影响结构缓存文件名；脚本使用全量 human 样本，不做 train/test split。
- one-hot 编码使用脚本内置的 T/U 等价映射（`bytes_to_onehot`），T 和 U 共用第 4 通道。

### 2. 事件收集

```python
events = collect_events_from_dataset(dataset, indices, target_mods)
```

- 从 `dataset.full_labels` 中提取 `(sample_idx, site_pos, class_idx)` 事件。
- 多标签处理：同一条 1001nt 序列可包含多个修饰位点和多个修饰类别，脚本按 `(sample_idx, site_pos, class_idx)` 拆分为独立事件。
- 仅保留 `25 <= site_pos <= 975` 的事件（保证 51nt 窗口不越界）。

### 3. Localized IG 计算

```python
score_51, pred_prob, logit = compute_localized_ig(
    model, inp, edge_index, batch_vec, class_idx, site_pos,
    ig_steps=args.ig_steps, device=device)
```

- IG target：单个修饰类别 logit `logits_12[:, class_idx]`，不使用 softmax。
- baseline：仅清零目标位点中心 51nt（`site_pos-25 : site_pos+26`），其余位置保持真实输入。
- 输出 `score_51`：51 维向量，每位置 `abs(ig_attr).sum(base_channel)`。

### 4. 窗口提取

```python
windows = extract_windows(score_51, seq_str, site_pos, motif_w=8, top_k=5)
```

- 在 51nt IG 分数上滑动 8nt 窗口，选 top-k 非重叠高分窗口。
- 输出写入 `windows.tsv`，包含 `motif_start/motif_end`、`motif_start_51/motif_end_51`、`window_score`、`seq_8nt`、`seq_51nt`。

### 5. 正样本 FASTA

- 将选中的 8nt 窗口序列写入 `positives.fasta`，作为 STREME `--p` 输入。
- 所有序列统一使用 RNA 字母 A/C/G/U（T 自动转 U）。

### 6. 负样本采样

```python
neg_seqs, neg_hdrs, neg_records = generate_negatives_matched_offsets(
    records, neg_zero_seq_path, motif_w=8, negative_ratio=1.0, seed=666)
```

- 负样本来源：`npy/zero/zero_seq.npy`（zero 修饰背景序列）。
- 匹配策略：按 positive 窗口的 `motif_start_51` 相对 offset，从随机 zero 样本的中心 51nt 同一位置抽取等长片段。
- 输出 `negatives.fasta`（STREME `--n` 输入）和 `negative_windows.tsv`。

### 7. STREME de novo motif

```python
run_streme(streme_out, fasta_path, negative_fasta=negative_fasta_path,
           minw=5, maxw=15, rna=True)
```

- 使用显式负样本（非 shuffled positives）作为背景。
- `--rna` 模式，字母表 A/C/G/U。
- 输出 `streme_out/streme.txt`。

### 8. 模型侧 DBSCAN motif

```python
cm, ig_sc = cal_consensus_motif_2(
    fasta_seqs, [r['window_score'] for r in records],
    eps=cluster_eps, min_samples=args.cluster_min_samples,
    random_state=args.seed)
```

- 对 IG top 窗口的 one-hot 序列做 UMAP/PCA 降维后跑 DBSCAN 聚类。
- 每簇生成共识 PWM，导出为 `model_motifs.meme`。
- 导出后调用 `rewrite_meme_alphabet_to_rna` 将 `ALPHABET= ACGT` 改写为 `ALPHABET= ACGU`。
- 聚类参数来源：`DEFAULT_CLUSTER_EPS_BY_MOD` 使用 `offset=16` 计算（如 `m6A: 2.0/16=0.125`）；`--cluster_eps` 可覆盖所有修饰；`--cluster_min_samples` 默认 10。

### 9. Tomtom 验证

```python
run_tomtom_validation(meme_p, streme_txt, tomtom_out)
```

- 比较模型侧 motif 与 STREME de novo motif 的相似性。
- 输出 `tomtom_out/tomtom.tsv`。

### 10. Logo 绘制与 p-value 过滤

```python
p_values = best_tomtom_pvalues_by_query(tomtom_rows, n_motifs)
cm_plot, ig_plot, p_plot, keep_idx = filter_motifs_with_pvalues(cm, ig_sc, p_values)
draw_motif_logos(cm_plot, ig_plot, mod_name, ..., p_values=p_plot)
```

- 仅绘制有 Tomtom p-value 的模型 motif；过滤掉 `p=NA` 的 motif。
- 若所有 motif 均无 p-value，删除旧 logo 文件（`remove_logo_files`）。
- 输出 `*_motif_motif.png` 和 `*_motif_motif.pdf`。

### 11. 全局 Excel 汇总

```python
write_tomtom_summary_excel(output_dir, all_match_rows, all_streme_rows, all_model_rows)
```

- 所有修饰跑完后生成 `tomtom_summary.xlsx`，包含三个 sheet：
  - `tomtom_matches`：模型 motif 与 STREME motif 的 Tomtom 匹配详情。
  - `streme_motifs`：各修饰的 STREME de novo motif 汇总。
  - `model_motifs`：各修饰的模型侧 DBSCAN motif 汇总。

## 输出文件

每个修饰目录（如 `output/human_localized_ig/m6A/`）下：

| 文件 | 说明 |
|------|------|
| `windows.tsv` | positive IG 窗口元数据（sample_idx, site_pos, motif_start/end, window_score, seq_8nt, seq_51nt 等） |
| `positives.fasta` | IG top-k 8nt 窗口序列，RNA A/C/G/U |
| `negatives.fasta` | zero 背景匹配 offset 负样本序列 |
| `negative_windows.tsv` | 负样本窗口元数据 |
| `errors.tsv` | IG 计算失败的事件记录（若有） |
| `streme_out/streme.txt` | STREME de novo motif 输出 |
| `model_motifs.meme` | 模型侧 DBSCAN 共识 motif（MEME 格式，ALPHABET=ACGU） |
| `treex_consensus_motifs.npz` | 模型侧 PWM、IG 分数、聚类参数 |
| `tomtom_out/tomtom.tsv` | Tomtom 匹配结果 |
| `*_motif_motif.png` | motif logo（仅含 p-value 的 motif） |
| `*_motif_motif.pdf` | motif logo PDF 版 |
| `results.txt` | 该修饰的 Tomtom 匹配摘要 |

全局输出：

| 文件 | 说明 |
|------|------|
| `run_config.json` | 运行配置（含 data_usage=all_human, sequence_alphabet, cluster_min_samples 等） |
| `tomtom_summary.xlsx` | 所有修饰的 STREME/model/Tomtom 汇总（三个 sheet） |

## 参数与默认值

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | 模型配置 JSON（如 `json/human.json`） |
| `--checkpoint` | 必填 | 模型 checkpoint 路径 |
| `--output_dir` | `ipynb/motif/output` | 输出根目录 |
| `--mods` | 全部 12 类 | 逗号分隔的修饰名列表（如 `m6A,Am,Cm`） |
| `--sample_size` | 5000 | 每修饰最大事件采样数（0 或负数表示不限） |
| `--motif_w` | 8 | motif 窗口宽度（nt） |
| `--top_k` | 5 | 每事件选 top-k 个非重叠窗口 |
| `--ig_steps` | 32 | Integrated Gradients 积分步数 |
| `--seed` | 666 | 随机种子 |
| `--device` | `cuda` | 推理设备 |
| `--skip_external_tools` | False | 跳过 STREME/Tomtom（仅生成 IG 窗口文件） |

### 负样本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--negative_dir` | `npy/zero` | zero_seq.npy 所在目录 |
| `--negative_mode` | `matched_offsets` | 负样本采样策略（当前仅支持 matched_offsets） |
| `--negative_ratio` | 1.0 | 负/正样本比例 |
| `--negative_seed` | 666 | 负样本随机种子 |
| `--allow_no_negative` | False | zero_seq.npy 缺失时是否继续 |

### STREME 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--streme_minw` | 5 | STREME 最小 motif 宽度 |
| `--streme_maxw` | 15 | STREME 最大 motif 宽度 |

### 聚类参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cluster_eps` | None（使用 per-mod 默认） | 覆盖所有修饰的 DBSCAN eps |
| `--cluster_min_samples` | 10 | DBSCAN min_samples |

`DEFAULT_CLUSTER_EPS_BY_MOD` 使用 `offset=16` 计算，各修饰默认 eps：

```python
offset = 16
Am=1.8/offset, Atol=1.7/offset, Cm=1.8/offset, Gm=1.8/offset,
Tm=2.1/offset, Y=2.1/offset, ac4C=1.9/offset, m1A=2.0/offset,
m5C=1.9/offset, m6A=2.0/offset, m6Am=2.1/offset, m7G=2.0/offset
```

### 遗留/保留参数

以下参数当前不是核心生效路径，保留用于兼容但不描述为已实现能力：

- `--positive_only`：当前未在主流程中生效。
- `--alignment_r_script`：R 脚本对齐路径，当前流程不调用。
- `--window_51`：窗口大小参数，当前硬编码为 51。
- `--batch_size`：当前 IG 计算为单样本前向。
- `--baseline`：baseline 策略，当前固定为 51nt 窗口清零。

## 当前已知问题 / 下一步

1. **DBSCAN noise 过滤**：`cal_consensus_motif_2` 应进一步过滤 DBSCAN `label=-1` 的 noise 点，除非所有点都是 noise 才 fallback 成单簇。当前实现可能将 noise 点混入共识 PWM。

2. **旧输出目录不可直接引用**：`all_mods_v2` 等目录中的旧输出由旧参数生成，不能直接代表当前脚本默认参数。解释结果前应使用当前参数重新运行。

3. **Atol/AtoI eps 评估**：需要基于当前 `offset=16` 和 noise 过滤后重新评估 Atol/AtoI 的最佳 eps 值。

4. **层级可视化（未来方向）**：M2D/A2P/MoHE 层级可视化（heatmap、中间层 gradient*activation attribution、event-level NPZ）未实现，可作为未来扩展方向。当前脚本不包含 `--visualize_layers`、`--save_event_figures` 等参数。

## 验收与测试

### 静态检查

```bash
python -m py_compile ipynb/motif/human_localized_ig_motif.py
python -m py_compile ipynb/motif/motif_utils.py
```

确认文档不再声称已实现 `--visualize_layers`、`--save_event_figures`、M2D/A2P/MoHE heatmap、event-level NPZ 等不存在功能。

### 小样本 smoke test（仅 IG 窗口）

```bash
python ipynb/motif/human_localized_ig_motif.py \
  --config json/human.json \
  --checkpoint <checkpoint.pt> \
  --output_dir ipynb/motif/output/smoke_test \
  --mods Atol --sample_size 20 --top_k 2 --ig_steps 4 \
  --skip_external_tools --batch_size 1 --device cuda
```

验收：
- `windows.tsv` 存在且非空。
- `positives.fasta` 中序列为 RNA `U` 字母表（不含 T）。
- `run_config.json` 记录 `data_usage=all_human`、`sequence_alphabet=RNA_ACGU_T_normalized_to_U`。

### 小样本 smoke test（含 STREME/Tomtom）

```bash
python ipynb/motif/human_localized_ig_motif.py \
  --config json/human.json \
  --checkpoint <checkpoint.pt> \
  --output_dir ipynb/motif/output/smoke_test_full \
  --mods Atol --sample_size 20 --top_k 2 --ig_steps 4 \
  --batch_size 1 --device cuda
```

验收：
- `streme_out/streme.txt` 存在。
- `model_motifs.meme` 的 `ALPHABET=` 行为 `ACGU`。
- `tomtom_out/tomtom.tsv` 存在。
- `tomtom_summary.xlsx` 存在且包含 `tomtom_matches`、`streme_motifs`、`model_motifs` 三个 sheet。
- PNG/PDF logo 中不出现 `p=NA`。
- 若 Tomtom 没有任何 p-value，旧 logo 文件应已被删除。

## 默认假设

- 脚本是本文档的唯一事实来源；旧输出目录仅作为历史结果参考。
- 本计划仅描述当前已实现的数据流和 CLI；未实现的功能不列入核心参数表。
- 层级可视化（M2D/A2P/MoHE heatmap）作为未来方向保留，当前不展开实现细节。
