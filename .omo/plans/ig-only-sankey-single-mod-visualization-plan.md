# IG-only Sankey Single-mod Visualization Plan

## 目标

在现有 localized IG motif 流程基础上，新增“单修饰单图”的 IG-only Sankey 可视化。每个修饰类别输出一张独立桑基图，用 IG top-k motif window 的贡献量作为唯一流量来源，展示：

```text
Input tensor -> M2D (cnn_block + gcn_block) -> Attention -> Attention pooling -> Motif
```

这里的 M2D 定义为当前模型中的 `cnn_block + gcn_block`。桑基图表达的是“IG 证据汇总路径”，不是神经网络内部真实守恒的信息流。

## 强约束

- 不修改 `ipynb/motif/human_localized_ig_motif.py`。
- 新功能必须基于原脚本复制出一个新脚本后实现。
- 原脚本只作为稳定基线和 diff 参考。
- 新脚本需要保留原 localized IG、STREME、Tomtom、motif logo 主流程，Sankey 只是附加输出。

建议新增脚本：

```text
ipynb/motif/human_localized_ig_motif_ig_sankey.py
```

建议复制来源：

```text
ipynb/motif/human_localized_ig_motif.py
```

## 图语义

IG-only Sankey 使用 `windows.tsv`/内存中的 `records` 作为数据基础。当前 `records` 已包含：

- `mod_name`
- `sample_idx`
- `site_pos`
- `class_idx`
- `window_score`
- `seq_8nt`
- `motif_start_51`
- `motif_end_51`
- `rank`
- `pred_prob`
- `logit`

桑基图边权只使用：

```text
flow_weight = sum(window_score)
```

不使用 attention weight，不使用 hidden-state norm，不使用 `IG * attention`。`Attention` 和 `Attention pooling` 只作为模型结构叙事节点。

## 单修饰单图结构

每个 `mod_name` 独立输出一张 Sankey。

基础链：

```text
Input tensor -> M2D
M2D -> Attention
Attention -> Attention pooling
Attention pooling -> Motif leaf nodes
```

前三条边的权重相同：

```text
total_flow = sum(flow_weight over visible motif leaves)
```

最后一段按 motif leaf 分流：

```text
Attention pooling -> Motif_1: sum(window_score assigned to Motif_1)
Attention pooling -> Motif_2: sum(window_score assigned to Motif_2)
...
```

## Motif leaf 方案

第一版支持三种 leaf 分组模式，优先级从高到低：

### 1. 用户 motif 映射表

适合用户自行拼接 motif 后回填。

新增参数：

```text
--sankey_motif_map <path>
```

建议 TSV 字段：

```text
mod_name sample_idx site_pos rank motif_id motif_label
```

匹配键：

```text
(mod_name, sample_idx, site_pos, rank)
```

如果同一窗口在映射表中有 `motif_label`，叶子节点显示 `motif_label`；否则显示 `motif_id`。

### 2. 按 `seq_8nt` 聚合

无映射表时默认 fallback。叶子节点：

```text
Motif window: <seq_8nt>
```

优点是完全依赖现有 `windows.tsv`，不需要额外聚类结果，适合 smoke test。

### 3. 按模型 DBSCAN motif 聚合

后续增强项。若要按 `Model_Motif_1/2/...` 分组，需要 `cal_consensus_motif_2` 返回每条窗口所属 cluster label，当前脚本没有保存该映射。第一版不建议强做，除非同步改造 motif_utils 或在新脚本内重跑聚类并记录 labels。

## 输出文件

每个修饰目录下新增：

```text
<mod_dir>/<mod_name>_ig_only_sankey.html
<mod_dir>/<mod_name>_ig_only_sankey.tsv
```

可选 PNG 导出：

```text
<mod_dir>/<mod_name>_ig_only_sankey.png
```

PNG 需要 `plotly[kaleido]` 或 `kaleido` 可用；如果不可用，只输出 HTML 和 TSV，不中断主流程。

全局输出可选：

```text
<output_dir>/ig_only_sankey_summary.tsv
```

用于汇总每个修饰的 `total_flow`、leaf 数、窗口数。

## TSV 设计

`<mod_name>_ig_only_sankey.tsv` 建议字段：

```text
mod_name
source
target
flow_weight
raw_window_count
mean_window_score
mean_pred_prob
leaf_group
leaf_label
leaf_seq
```

其中前三段结构边可设置：

```text
leaf_group = "__pipeline__"
leaf_label = ""
leaf_seq = ""
```

叶子边记录具体 motif 分组统计。

## 新增参数

建议在新脚本 argparse 中加入：

```text
--enable_ig_sankey
--sankey_group_by {motif_map,seq_8nt}
--sankey_motif_map <path>
--sankey_top_n <int>
--sankey_min_flow <float>
--sankey_export_png
```

默认建议：

```text
--enable_ig_sankey: False
--sankey_group_by: seq_8nt
--sankey_top_n: 20
--sankey_min_flow: 0.0
--sankey_export_png: False
```

若传入 `--sankey_motif_map`，可以自动切换到 `motif_map`，但最好仍允许用户显式指定。

## 绘图实现建议

优先使用 Plotly：

```python
import plotly.graph_objects as go
```

理由：

- Sankey 原生支持较好。
- HTML hover 适合查看 `flow_weight`、窗口数、平均概率。
- 不影响现有 matplotlib motif logo 逻辑。

新增 helper 建议：

```python
def load_sankey_motif_map(path):
    ...

def assign_sankey_leaf(record, motif_map, group_by):
    ...

def build_ig_only_sankey_edges(records, mod_name, motif_map=None,
                               group_by='seq_8nt', top_n=20,
                               min_flow=0.0):
    ...

def write_sankey_edges_tsv(path, edge_rows):
    ...

def draw_ig_only_sankey(edge_rows, output_html, output_png=None,
                        title=None):
    ...
```

核心逻辑应尽量只依赖 `records`，不要重新 forward，不要重新计算 IG。

## Top-N 与长尾处理

如果 leaf 太多，桑基图会很乱。第一版建议：

- 按 leaf `sum(window_score)` 降序保留 top-N。
- 剩余 leaf 合并为 `Other selected motif windows`。
- `Other` 的 flow 是被合并 leaf 的总和。

注意：前三段 `total_flow` 应等于 top-N + Other 的总和，而不是所有原始 leaf 的总和被截断后丢失。

## 主流程插入位置

在新脚本中，每个修饰完成 `records` 收集并写出 `windows.tsv` 后即可绘制 Sankey：

```text
records collected
write_windows_tsv(...)
if args.enable_ig_sankey:
    build edges from records
    write sankey tsv
    draw html/png
```

建议插入在正负样本 FASTA、STREME、Tomtom 之前。这样即使 `--skip_external_tools` 或外部工具失败，Sankey 仍可输出。

## 运行示例

仅生成 IG 窗口和 Sankey：

```bash
python ipynb/motif/human_localized_ig_motif_ig_sankey.py \
  --config json/human.json \
  --checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
  --output_dir ipynb/motif/output/debug_ig_sankey \
  --mods m6A \
  --sample_size 20 \
  --top_k 2 \
  --ig_steps 8 \
  --skip_external_tools \
  --enable_ig_sankey \
  --sankey_group_by seq_8nt \
  --device cuda
```

使用用户 motif 映射表：

```bash
python ipynb/motif/human_localized_ig_motif_ig_sankey.py \
  --config json/human.json \
  --checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
  --output_dir ipynb/motif/output/debug_ig_sankey_map \
  --mods m6A \
  --sample_size 100 \
  --top_k 3 \
  --ig_steps 16 \
  --skip_external_tools \
  --enable_ig_sankey \
  --sankey_group_by motif_map \
  --sankey_motif_map ipynb/motif/motif_maps/m6A_sankey_map.tsv \
  --device cuda
```

## 验收标准

### 文件安全

- `git diff -- ipynb/motif/human_localized_ig_motif.py` 为空。
- 新增脚本存在：`ipynb/motif/human_localized_ig_motif_ig_sankey.py`。

### 静态检查

```bash
python -m py_compile ipynb/motif/human_localized_ig_motif_ig_sankey.py
```

### smoke test

运行单修饰小样本后，确认：

- `<output_dir>/m6A/windows.tsv` 存在且非空。
- `<output_dir>/m6A/m6A_ig_only_sankey.html` 存在。
- `<output_dir>/m6A/m6A_ig_only_sankey.tsv` 存在。
- Sankey TSV 中每个 `mod_name` 只有当前修饰。
- `Input tensor -> M2D`、`M2D -> Attention`、`Attention -> Attention pooling` 三条结构边权重相等。
- 所有 `Attention pooling -> leaf` 的 flow 之和等于结构边权重。

### 可解释性检查

- 图标题或 HTML hover 中明确写明 `IG-only`。
- 图注不出现“attention weight as flow”这类表述。
- M2D 节点标注为 `M2D (CNN + RGCN)` 或 `M2D (cnn_block + gcn_block)`。

## 后续增强

1. 支持从模型 DBSCAN 聚类结果生成 leaf label。
2. 增加 `selected_flow / total_51nt_ig_flow` 覆盖率，需要额外保存每个 event 的 51nt IG 总量。
3. 增加跨修饰总览图，但不替代单修饰单图。
4. 在 HTML 中加入 leaf hover 的代表序列、窗口数、平均 logit、平均 pred_prob。
