# Motif Apply Converged Training Plan

## 目标

在当前仓库新增一套 `ipynb/motif_apply/` 工作流，用于先从统一 human 预训练权重初始化，在 gen3 或 plant 数据上继续训练至收敛，然后使用收敛后的权重绘制 motif 和 Sankey。

原始 `ipynb/motif/*.py` 文件必须保留，不作为本次实现目标的运行入口。新增代码、训练产物和绘图输出均放在 `ipynb/motif_apply/` 下。

用户已确认的硬约束：

- 允许留出 10% validation 用于 early stopping。
- plant 只训练和绘制 `Y,m5C,m6A`，对应 class index `[5, 8, 9]`。
- 原 `ipynb/motif/*.py` 保留不动。
- 绘制 motif 和 Sankey 时仍然使用 train 数据，不使用 validation 数据。
- 原始初始化权重统一使用：
  `logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt`

## 推荐最终目录

```text
ipynb/motif_apply/
  motif_utils.py
  apply_train.py
  gen3_localized_ig_motif.py
  gen3_localized_ig_motif_ig_sankey.py
  plant_localized_ig_motif.py
  plant_localized_ig_motif_ig_sankey.py
  checkpoints/
    gen3/
      best_converged.pt
      last.pt
      train_config.json
      train_history.tsv
    plant/
      best_converged.pt
      last.pt
      train_config.json
      train_history.tsv
  output/
    gen3/
      localized_ig/
      ig_attn_stage_sankey/
    plant/
      localized_ig/
      ig_attn_stage_sankey/
```

`motif_utils.py` 建议从 `ipynb/motif/motif_utils.py` 复制一份，使 `motif_apply` 下的脚本可以自包含运行。不要改原文件。

## 需要新增或复制的脚本

从当前已有迁移脚本复制到 `ipynb/motif_apply/`：

- `ipynb/motif/gen3_localized_ig_motif.py`
- `ipynb/motif/gen3_localized_ig_motif_ig_sankey.py`
- `ipynb/motif/plant_localized_ig_motif.py`
- `ipynb/motif/plant_localized_ig_motif_ig_sankey.py`

复制后只修改 `motif_apply` 下的新文件。

新增共享训练模块：

- `ipynb/motif_apply/apply_train.py`

四个绘图脚本都调用 `apply_train.py`，避免同一数据集被 motif 和 Sankey 脚本重复训练。

## 共享训练模块设计

### 公开接口

建议实现：

```python
def ensure_converged_checkpoint(
    dataset_name,
    config_path,
    init_checkpoint,
    output_root,
    force_retrain=False,
    max_epochs=None,
    patience=10,
    min_delta=1e-4,
    val_ratio=0.1,
    seed=None,
    batch_size=None,
    device=None,
):
    ...
```

返回：

```python
{
    "checkpoint_path": ".../best_converged.pt",
    "train_indices": [...],
    "val_indices": [...],
    "dataset": dataset,
    "config": cfg,
    "history_path": ".../train_history.tsv",
}
```

注意：绘图脚本后续只使用返回的 `train_indices` 收集 event，不使用 `val_indices`。

### 训练 checkpoint 复用规则

如果存在：

```text
ipynb/motif_apply/checkpoints/{dataset_name}/best_converged.pt
```

且没有传 `--force_retrain`，直接复用该 checkpoint，不再训练。

如果传 `--force_retrain`，或 checkpoint 不存在，则从固定初始权重重新训练：

```text
logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt
```

### 训练/验证划分

使用全量目标数据集做 deterministic split：

- train: 90%
- validation: 10%
- seed 默认使用配置中的 `training.random_seed`，如果 CLI 传入 `--seed` 则优先使用 CLI。

gen3 可复用 `multi_label_disjoint_split` 的思想，但比例需要改为 0.9/0.1。若直接调用现有函数，应确认其输出稳定并能处理目标数据集。

plant 只考虑 class `[5,8,9]`。划分必须确保 validation 中尽量包含这三类正样本：

- 先对每个有效类按样本索引收集正样本。
- 每类抽取 10% 进入 validation。
- 合并去重得到 `val_indices`。
- `train_indices = all_indices - val_indices`。
- 若某类样本太少，至少保留 1 个 validation 正样本，除非该类总样本数为 0。

### 收敛标准

默认 early stopping：

- `max_epochs`: 使用 config 中 `training.num_epochs`，CLI 可覆盖。
- `patience=10`
- `min_delta=1e-4`
- gen3 监控 validation macro F1，若实现成本高，可退化为 validation loss，但必须在 `train_config.json` 中记录。
- plant 监控有效类 `[5,8,9]` 的 validation macro F1。

每个 epoch 写入：

```text
epoch	train_loss	val_loss	val_macro_f1	learning_rate	is_best
```

到：

```text
ipynb/motif_apply/checkpoints/{dataset_name}/train_history.tsv
```

### 模型初始化

必须使用与 checkpoint 匹配的模型结构配置：

- gen3: 默认 `json/3gen.json`
- plant: 默认 `json/plant.json`

两者当前结构与 human checkpoint 的 64/128 版本一致；不要使用 `json/plant_single.json`，它是 128/256 结构，不能和指定 checkpoint 直接匹配。

加载方式：

```python
ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
```

训练保存方式建议兼容现有工具：

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "dataset_name": dataset_name,
    "init_checkpoint": init_checkpoint,
    "train_indices": train_indices,
    "val_indices": val_indices,
    "best_metric": best_metric,
    "config": config_dict,
}, best_path)
```

## gen3 训练细节

数据集：

```python
Gen3Dataset(
    mode="train",
    data_dir=cfg["data"].get("gen3_data_dir", "npy/3gen"),
    cache_dir=cfg["data"].get("gen3_cache_dir", cfg["data"].get("cache_dir", "npy/cache")),
    use_cache=True,
    preload_cache=True,
)
```

训练 loss：

- 12-class BCEWithLogitsLoss。
- hierarchical 开启时同时计算 4-class loss，沿用 `utils.common.train_epoch` 的行为。
- 如果数据集中有 `y_site` 且 config 中 `use_attention_supervision=true`，可以启用 attention supervision。

验证指标：

- 使用 all 12 classes macro F1。
- 可复用 `get_all_predictions` + `evaluate_unbalance`，或实现轻量 macro F1，避免引入太多日志输出。

绘图默认修饰：

- gen3 绘制 12 类全量：`','.join(MOD_NAMES.values())`。

## plant 训练细节

数据集：

```python
PlantDataset(
    plant_dir=cfg["data"].get("plant_data_dir", "npy/plant"),
    cache_dir=cfg["data"].get("plant_cache_dir", cfg["data"].get("cache_dir", "npy/cache")),
    use_cache=True,
    preload_cache=True,
)
```

有效类别：

```python
PLANT_VALID_CLASSES = [5, 8, 9]  # Y, m5C, m6A
PLANT_VALID_GROUPS = [3, 1, 0]   # U, C, A; 如果按 group index 顺序可写成 [0, 1, 3]
```

训练 loss：

- 模型保持 12-class head，不 prune head。
- 只对 logits[:, [5,8,9]] 与 batch.y[:, [5,8,9]] 计算 12-class 分类 loss。
- hierarchical 开启且 batch 有 `y_4class` 时，只对 group `[0,1,3]` 计算 4-class loss。
- 不使用 `train_plant.py` 中的 few-shot/pruned 训练逻辑。

注意：当前 `dataset/plant.py` 的 `Data` 返回里没有 `y_4class/y_site`，但类对象本身加载了 `full_labels/y_4class`。如果要做 hierarchical group loss 或 attention supervision，需要在 `motif_apply` 的训练模块中二选一：

1. 优先修改 `dataset/plant.py`，让 `__getitem__` 返回 `y_4class` 和 `y_site`，但这会影响原数据集文件。
2. 更稳妥：在 `apply_train.py` 中构建一个轻量 wrapper，调用 `PlantDataset[idx]` 后补充：
   - `data.y_4class = torch.FloatTensor(dataset.y_4class[idx]).unsqueeze(0)`
   - `data.y_site = torch.LongTensor(dataset.full_labels[idx])`

推荐方案 2，避免改原数据集。

绘图默认修饰：

```python
PLANT_DEFAULT_MODS = "Y,m5C,m6A"
```

## 四个 motif_apply 绘图脚本的改动

四个脚本都需要新增 CLI：

```text
--init_checkpoint
--trained_checkpoint
--force_retrain
--max_train_epochs
--early_stop_patience
--early_stop_min_delta
--val_ratio
--train_batch_size
```

默认：

```text
--init_checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt
--trained_checkpoint None
--val_ratio 0.1
--early_stop_patience 10
--early_stop_min_delta 1e-4
```

运行流程：

1. 如果 `--trained_checkpoint` 给定：跳过训练，直接用它绘图。
2. 否则调用 `ensure_converged_checkpoint(...)`，得到 `best_converged.pt` 和 `train_indices`。
3. 后续 `load_model(cfg, checkpoint_path, device)` 使用收敛 checkpoint。
4. event 收集只使用 `train_indices`：

```python
indices = train_indices
all_events = collect_events_from_dataset(dataset, indices, target_mods)
```

5. `run_config.json` 中记录：
   - `init_checkpoint`
   - `trained_checkpoint`
   - `training_mode = "train_90_val_10_early_stop"`
   - `motif_event_split = "train_only"`
   - `n_train`
   - `n_val`
   - `train_history_path`

## 默认输出目录

非 Sankey：

```text
ipynb/motif_apply/output/gen3/localized_ig
ipynb/motif_apply/output/plant/localized_ig
```

Sankey：

```text
ipynb/motif_apply/output/gen3/ig_attn_stage_sankey
ipynb/motif_apply/output/plant/ig_attn_stage_sankey
```

每个输出目录下仍保持现有 per-mod 子目录结构，例如：

```text
ipynb/motif_apply/output/plant/localized_ig/m6A/
ipynb/motif_apply/output/plant/ig_attn_stage_sankey/m6A/
```

## launch.json 更新建议

保留当前 `ipynb/motif` 的启动项。新增 4 个 `motif_apply` 启动项：

- `motif_apply gen3 localized IG`
- `motif_apply plant localized IG`
- `motif_apply gen3 IG Sankey`
- `motif_apply plant IG Sankey`

共同参数：

```text
--init_checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt
--early_stop_patience 10
--val_ratio 0.1
```

Sankey 入口保持：

```text
--enable_ig_sankey
--sankey_mode ig_attention_stage
--sankey_group_by motif_cluster
```

## 验证步骤

实施后必须运行：

```bash
python -m py_compile ipynb/motif_apply/apply_train.py
python -m py_compile ipynb/motif_apply/gen3_localized_ig_motif.py
python -m py_compile ipynb/motif_apply/gen3_localized_ig_motif_ig_sankey.py
python -m py_compile ipynb/motif_apply/plant_localized_ig_motif.py
python -m py_compile ipynb/motif_apply/plant_localized_ig_motif_ig_sankey.py
```

建议 smoke test：

```bash
python ipynb/motif_apply/gen3_localized_ig_motif.py \
  --mods m6A --sample_size 2 --top_k 1 --ig_steps 2 \
  --max_train_epochs 1 --skip_external_tools --allow_no_negative

python ipynb/motif_apply/plant_localized_ig_motif_ig_sankey.py \
  --mods m6A --sample_size 2 --top_k 1 --ig_steps 2 \
  --max_train_epochs 1 --skip_external_tools --allow_no_negative \
  --enable_ig_sankey --sankey_mode ig_attention_stage
```

如果 smoke test 因缺少依赖或数据缓存失败，需要记录具体失败原因，不要静默跳过。

## 不要做的事

- 不要修改原 `ipynb/motif/*.py` 作为新工作流入口。
- 不要使用 `json/plant_single.json` 加载指定 human checkpoint。
- 不要复用 `train_plant.py` 的 pruned/few-shot head 作为本次 plant 全量训练逻辑。
- 不要在 validation 数据上绘制 motif 或 Sankey。
- 不要把训练输出写到 `ipynb/motif/output/`；所有新输出必须在 `ipynb/motif_apply/output/`。

