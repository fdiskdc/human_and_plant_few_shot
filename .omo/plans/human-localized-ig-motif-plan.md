# Human Localized IG Motif Analysis Plan

## 目标

在 `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif` 下新增一个脚本，调用 `/home/dc/vscode/vscode20260424/rgcnformer_sum/model/main_model.py` 的 `RNA_ClassQuery_Model`，基于 human 数据集完成 motif 分析。

核心方法采用 localized Integrated Gradients：

- 模型输入仍为完整 1001nt human RNA 序列。
- 对每个真实修饰位点，以该位点为中心取 51nt 局部区域。
- IG baseline 只替换该 51nt 区域，其他 950nt 保持真实输入不变。
- IG target 为对应修饰类别的序列级 logit。
- 在 51nt attribution score 内使用 8nt 滑窗选 top-k motif windows。
- 后续复用旧 motif 流程：STREME 发现 motif，模型 consensus motif 导出 MEME，再用 Tomtom 比较得到 p-value。

## 新增脚本位置

建议新增：

`/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/human_localized_ig_motif.py`

输出目录默认：

`/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig`

## 实施路径约定

当前计划文件位于：

`/home/dc/vscode/vscode20260406/rgcnformer_sum/.omo/plans/`

但脚本、模型、配置和输出目标均指向：

`/home/dc/vscode/vscode20260424/rgcnformer_sum/`

实施前必须确认本次修改目标仓库。如果在 `20260406/rgcnformer_sum` 中实施，应将脚本路径、`PROJECT_ROOT`、输出目录和 checkpoint 路径统一改为当前仓库；如果在 `20260424/rgcnformer_sum` 中实施，则当前计划只作为说明文档和参考。

旧项目 `/home/dc/vscode/vscode20260406/DtreeInMultiRM` 只作为 motif 工具函数参考，不作为运行时必需 import 路径。第一版建议把必要函数复制或重写到新脚本/本仓库 `motif_utils.py`，避免跨项目路径依赖。

## 主要依赖文件

- 模型定义：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/model/main_model.py`

- human 数据集：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/dataset/human_with_seq.py`

- human 基础标签映射：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/dataset/human.py`

- 模型配置：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/json/human.json`

- 参考 checkpoint：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt`

- 旧 motif 参考逻辑：
  `/home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/draw_motif_all.py`

- 旧 motif 工具函数参考：
  `/home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/utils.py`

- 旧 8nt 滑窗和 consensus motif 参考：
  `/home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/util_cm_treex_local_fixed_pwm.py`

## 数据单位

以修饰事件 event 为最小单位：

```text
event = (sample_idx, site_pos, class_idx)
```

其中：

- `sample_idx`: human 数据集中样本索引。
- `site_pos`: `y_site` 中真实修饰位点。
- `class_idx`: 模型类别索引，范围 `0-11`。
- `mod_id`: human 位点标签，等于 `class_idx + 1`。

同一条 1001nt 序列上如果有多个修饰位点，应拆成多个 event。不同修饰类别分别计算 IG、分别汇总 motif。

### sample_idx 保真要求

`sample_idx` 必须是 human 数据集中原始样本索引，而不是 DataLoader batch 内下标。实现时需要满足以下任一方式：

- 直接按 dataset index 顺序枚举样本，`shuffle=False`。
- 在 dataset wrapper 的 `__getitem__` 中给 `Data` 增加 `sample_idx=torch.LongTensor([idx])`。
- event 收集阶段保存 `(idx, site_pos, class_idx)`，后续 IG 只通过该 index 重新读取样本。

不得从 batch 顺序反推原始 index。

### 坐标体系

所有输出 TSV/NPZ 中的区间字段统一采用 **0-based half-open** 坐标：

```text
start 包含，end 不包含
```

因此：

- `window51_start = site_pos - 25`
- `window51_end = site_pos + 26`
- `motif_start = window51_start + motif_start_51`
- `motif_end = window51_start + motif_end_51_exclusive`
- `seq_51nt = seq_str[window51_start:window51_end]`，长度必须为 51。
- `seq_8nt = seq_str[motif_start:motif_end]`，长度必须为 8。

旧 `highest_x` 返回的 `end_idx` 是 inclusive。若复刻该函数，写入 TSV 前必须转换为 half-open：

```python
motif_end_51_exclusive = old_end_idx_inclusive + 1
```

可视化高亮区必须使用同一套 half-open 坐标。

## Localized IG 设计

对每个 event：

1. 读取完整输入：

```python
input_full = batch.x
```

注意 PyG DataLoader 默认不会给 `[B, 1001, 4]` 的 `batch.x`。通常形状为：

```text
batch.x:     [B * 1001, 4]
batch.batch: [B * 1001]
```

第一版推荐 `batch_size=1`，单 event 计算时显式使用：

```text
input_full: [1001, 4]
edge_index: [2, E]
batch_vec:  [1001]，全 0
```

如果包装给 Captum，可在 wrapper 内把输入 reshape 回模型需要的格式：

```python
def forward(x_flat_or_seq):
    x = x_flat_or_seq.view(1001, 4)
    out = model(x, edge_index, batch_vec)
    logits_12 = out[0] if isinstance(out, tuple) else out
    return logits_12[:, class_idx] if logits_12.dim() == 2 else logits_12[class_idx]
```

2. 计算 51nt 边界：

```python
site_start = site_pos - 25
site_end = site_pos + 26
```

边界不足 51nt 的 event 默认跳过，第一版不做 padding，避免引入人工碱基。

3. 构造 localized baseline：

```python
baseline_full = input_full.clone()
baseline_full[site_start:site_end, :] = 0.0
```

其他位置保持真实输入不变。

4. 对目标类别 logit 计算 IG：

```python
target = logits_12[:, class_idx]
```

积分路径含义：

```text
只有真实修饰位点周围 51nt 从 baseline 变化为真实序列；
完整 1001nt 上下文中的其他位置始终保持真实输入。
```

5. 聚合 attribution 到位点分数：

```python
site_scores = abs(ig_attr).sum(axis=-1)
```

之后只取：

```python
score_51 = site_scores[site_start:site_end]
```

`score_51.shape` 必须为 `[51]`。如果 Captum 输入使用 flatten 形式，得到 attribution 后必须先 reshape 为 `[1001, 4]` 再聚合。

### 数据目录加载注意

`json/human.json` 中 human 数据目录为 `npy/human3`，实际包含：

```text
npy/human3/seq.npy
npy/human3/1001loc.npy
npy/human3/12loc.npy
npy/human3/4loc.npy
```

但 `Mer100DatasetWithSeq(use_human3=True)` 继承的 `human.py` 逻辑默认查找 `human3`、`../human3`，否则回退到旧绝对路径，不会自动使用 `Config.human_data_dir`。实现时必须补一个显式数据目录方案：

- 推荐：新增轻量 wrapper，直接从 `--data_dir` 读取四个 npy，并复用 one-hot、edge_index/cache 逻辑。
- 或修改/继承 `Mer100DatasetWithSeq`，让 `use_human3=True` 时优先使用传入的 `data_dir`。

不能依赖当前工作目录下恰好存在 `human3`。

## 8nt 窗口选择

复刻旧逻辑中的 `highest_x`：

- 输入：51nt attribution score。
- 窗口宽度：`motif_w = 8`。
- 每个窗口分数：窗口内 attribution score 求和。
- 取 top-k 个窗口。
- 选择一个窗口后，用 `p = 1` 排除窗口左右 padding 区域，减少近邻重复窗口。

如果 51nt 内可选非重叠窗口不足 `top_k`，只输出实际可选窗口数，并在日志中记录。窗口分数使用同一份 `score_51`，不得重新计算或使用 attention 替代。

每个选中窗口输出：

```text
sample_idx
site_pos
class_idx
mod_name
window51_start
window51_end
motif_start
motif_end
window_score
seq_8nt
seq_51nt
motif_start_51
motif_end_51
rank
pred_prob
logit
```

metadata 建议保存到：

`/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig/<mod_name>/windows.tsv`

## STREME 与 Tomtom

每个修饰类别独立运行：

```text
/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig/<mod_name>/
```

每类输出：

```text
positives.fasta
windows.tsv
streme_out/
aligned_seqs.txt
model_motifs.meme
treex_consensus_motifs.npz
tomtom_out/tomtom.tsv
results.txt
```

STREME 参数建议第一版显式使用 8nt：

```text
--minw 8
--maxw 8
```

### FASTA 与字母表约定

`positives.fasta` 第一版写入 selected raw 8nt window，即 `seq_8nt`，不是 51nt 序列，也不是 aligned 序列。

human 序列使用 RNA 字母 `A/C/G/U`，旧 MEME 导出函数使用 DNA 字母 `A/C/G/T`。第一版统一采用 DNA 字母表：

- 写 FASTA/MEME 前将 `U` 转为 `T`。
- `run_streme(..., dna=True)` 或 CLI 中显式传 `--dna`。
- `model_motifs.meme` 使用 `ALPHABET= ACGT`。

如果后续改为 RNA alphabet，STREME、MEME 导出、Tomtom target/query 必须同时切换，不能混用 `U` 和 `T`。

### alignment 与 consensus

R alignment 只用于 consensus motif 聚类/展示，不用于 STREME 输入。若 alignment 输出包含 gap `-`：

- `cal_consensus_motif_2` 可以按旧逻辑把 gap 编码为均匀分布。
- 导出到 MEME 前的 PWM 必须保证每行是 4 列且归一化。
- 不得把带 gap 的 aligned sequence 直接写入 `positives.fasta` 给 STREME。

Tomtom 比较仍保持旧逻辑：

```text
query:  model_motifs.meme
target: streme_out/streme.txt
```

p-value 来自 Tomtom，而不是只看 STREME 输出。

## 多修饰位点策略

同一序列多个修饰位点时：

```text
sample i:
  pos 120 -> m6A
  pos 310 -> m5C
  pos 700 -> m6A
```

应拆成：

```text
(i, 120, m6A)
(i, 310, m5C)
(i, 700, m6A)
```

计算策略：

- `(i, 120, m6A)` 使用 `target=logit_m6A`，只打开 `pos120` 周围 51nt。
- `(i, 700, m6A)` 使用 `target=logit_m6A`，只打开 `pos700` 周围 51nt。
- `(i, 310, m5C)` 使用 `target=logit_m5C`，只打开 `pos310` 周围 51nt。

同一类别同一序列的多个位点可以分别计算 localized IG，因为 baseline 打开的 51nt 区域不同，得到的是不同局部区域在完整上下文下对该 class logit 的贡献。

## 样本筛选

建议第一版支持以下参数：

```text
--mods m6A,m5C,m1A
--sample_size 5000
--pred_threshold 0.5
--positive_only
--seed 666
--skip_external_tools
```

默认筛选：

```text
y_site 中存在对应 mod_id
```

可选进一步筛选：

```text
sigmoid(logits_12[class_idx]) >= pred_threshold
```

注意 human 是多标签任务，概率应使用 `sigmoid`，不要使用 `softmax`。

`--sample_size` 定义为每个 mod 最多处理的 event 数，不是样本数，也不是输出 window 数。若同一样本包含多个同类位点，每个位点都算一个 event。采样应在 event 级别执行，并使用 `--seed` 固定顺序。

建议第一版增加以下过滤项：

- `--skip_n_windows`：如果 51nt 或 8nt window 含 `N`，跳过该 window。
- `--min_events_per_mod`：默认 10；低于该数量仍可输出 TSV，但跳过 STREME/Tomtom 并记录 warning。
- `--positive_only`：只要求真实标签存在；若同时提供 `--pred_threshold`，再要求对应 sigmoid 概率达标。
- `--skip_external_tools`：只生成 IG、windows、FASTA、NPZ/TSV，不调用 R/STREME/Tomtom，用于 smoke test。

## CLI 建议

```bash
python /home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/human_localized_ig_motif.py \
  --config /home/dc/vscode/vscode20260424/rgcnformer_sum/json/human.json \
  --checkpoint /home/dc/vscode/vscode20260424/rgcnformer_sum/logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
  --output_dir /home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig \
  --mods m6A,m5C,m1A \
  --sample_size 5000 \
  --window_51 51 \
  --motif_w 8 \
  --top_k 5 \
  --ig_steps 32 \
  --baseline zero \
  --data_dir /home/dc/vscode/vscode20260424/rgcnformer_sum/npy/human3 \
  --seed 666 \
  --batch_size 1 \
  --device cuda
```

`--batch_size 1` 是第一版建议值，因为 localized IG 是按 event 改 baseline，批处理逻辑更复杂。后续可优化为同类别同窗口事件批量计算。

## 实现步骤

1. 在 `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/human_localized_ig_motif.py` 建立脚本骨架。

2. 实现项目根路径注入：

```python
PROJECT_ROOT = "/home/dc/vscode/vscode20260424/rgcnformer_sum"
sys.path.insert(0, PROJECT_ROOT)
```

3. 加载配置、数据集、DataLoader：

- `/home/dc/vscode/vscode20260424/rgcnformer_sum/json/human.json`
- `/home/dc/vscode/vscode20260424/rgcnformer_sum/dataset/human_with_seq.py`
- 显式使用 `--data_dir /home/dc/vscode/vscode20260424/rgcnformer_sum/npy/human3`，不要依赖 `human.py` 的默认 `human3` 搜索路径。

配置解析建议第一版避免直接调用 `utils.load_config()`，因为它会创建 timestamp log/checkpoint 目录并写入 config。可改用无副作用 JSON 解析，只提取 `model` 和 `training.random_seed` 字段；若沿用 `load_config()`，需要接受它会产生日志目录这一副作用。

4. 加载 `/home/dc/vscode/vscode20260424/rgcnformer_sum/model/main_model.py` 中的 `RNA_ClassQuery_Model`，并加载 checkpoint。

5. 实现 localized IG：

- 优先使用 Captum `IntegratedGradients`。
- Captum wrapper 必须固定 `edge_index` 和 `batch_vec`，只把 `x` 作为可微输入。
- wrapper 返回标量或 `[B]` target logit；层级模型输出 tuple 时取 `out[0]` 作为 `logits_12`。
- 如果 Captum 不可用，可实现简单积分版本：
  - 在 baseline 和 input 之间线性插值。
  - 对每个 alpha 前向传播。
  - 对目标 class logit 反传。
  - 平均梯度乘以 `(input - baseline)`。

6. 实现 `highest_x` 本地函数，避免直接依赖旧项目路径。

7. 实现 event 收集与 51nt/8nt 窗口抽取。

8. 保存运行元数据：

- `run_config.json`：CLI 参数、项目根、config 路径、checkpoint 路径、checkpoint epoch、随机种子。
- `events.tsv`：所有候选 event 及过滤原因。
- `errors.tsv`：单 event 失败原因，不中断整个类别。

9. 每个修饰类别写出：

- `positives.fasta`
- `windows.tsv`

10. 迁移或本地实现以下 motif 工具：

- `run_streme`
- `run_tomtom_validation`
- `export_to_meme`
- `cal_consensus_motif_2` 相关最小依赖

第一版可直接从旧文件复制必要函数到新脚本，后续再拆成 `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/motif_utils.py`。

11. 验证最小闭环：

```bash
python /home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/human_localized_ig_motif.py \
  --config /home/dc/vscode/vscode20260424/rgcnformer_sum/json/human.json \
  --checkpoint /home/dc/vscode/vscode20260424/rgcnformer_sum/logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
  --output_dir /home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/debug_localized_ig \
  --mods m6A \
  --sample_size 20 \
  --top_k 2 \
  --ig_steps 8 \
  --skip_external_tools \
  --batch_size 1 \
  --device cuda
```

## 验证指标

最小验证：

- 脚本能成功加载 human 数据集。
- 模型 checkpoint 能成功加载。
- 单个修饰类别能产生 `positives.fasta` 和 `windows.tsv`。
- `windows.tsv` 中所有 `seq_8nt` 长度为 8。
- `seq_8nt` 坐标落在对应 `seq_51nt` 内。
- 每条 event 的 `site_pos` 在 `y_site` 中确实等于对应 `mod_id`。
- `window51_end - window51_start == 51`。
- `motif_end - motif_start == 8`。
- `seq_8nt == seq_51nt[motif_start_51:motif_end_51]`。
- `score_51` 只由 localized IG 的输入 attribution 聚合得到。
- 同一 `--seed` 下 event 采样和 `windows.tsv` 顺序可复现。

完整验证：

- STREME 成功生成：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig/<mod_name>/streme_out/streme.txt`

- Tomtom 成功生成：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/output/human_localized_ig/<mod_name>/tomtom_out/tomtom.tsv`

- `results.txt` 能展示 Tomtom p-value/e-value/q-value。

单元/函数级验证：

- `highest_x` 返回窗口从 inclusive 转换到 half-open 后长度为 `motif_w`。
- 边界位点 `site_pos < 25` 或 `site_pos > 975` 被跳过。
- 含多个修饰位点的样本会生成多个 event。
- 层级模型 forward tuple 能正确解析 `logits_12`。

## 风险与注意事项

- localized IG 计算成本高，第一版应支持 `--sample_size`、`--mods`、`--ig_steps` 控制规模。
- `main_model.py` 是 1001nt 模型，不应直接喂 51nt 输入。
- 边界不足 51nt 的修饰位点第一版建议跳过。
- 多标签任务概率使用 `sigmoid`，不要使用 `softmax`。
- STREME/Tomtom 依赖 MEME Suite 和 conda 环境，默认可沿用旧环境名 `meme_env`。
- 如果 R alignment 脚本继续复用旧文件，需要显式传入：
  `/home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/alignment_local.R`
- `utils.load_config()` 有创建日志目录副作用，分析脚本如需无副作用运行应自行解析 JSON。
- `Captum IntegratedGradients` 对 PyG batch 输入不透明，推荐第一版每次只处理一个 event，并固定图结构输入。
- `U/T` 字母表混用会导致 MEME Suite 结果不可解释，必须在 FASTA/MEME/STREME/Tomtom 中保持一致。

## 后续扩展

- 增加 `--score_source attention` 作为快速对照。
- 增加 `--baseline uniform` 和 `--baseline background`。
- 增加 random-window 和 site-centered baseline motif 对照。
- 将 motif 通用函数拆到：
  `/home/dc/vscode/vscode20260424/rgcnformer_sum/ipynb/motif/motif_utils.py`
- 支持同类别 event 批量 localized IG，加速大规模运行。
