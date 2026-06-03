"""
3x3_2.py - 人类到植物零样本3x3迁移 v2 (单任务) / Human-to-Plant Zero-Shot 3x3 v2 (Single Task)

3x3.py 的 v2 版本:每个 m6A 单任务,带 --p_neg {zero,plant} 命令行选项选择负样本来源 (零背景或植物其他修饰)。
V2 of 3x3.py: each m6A single task, with --p_neg {zero,plant} CLI option to choose negative source.

功能模块 / Modules:
- 3 个独立模型训练 / 3 independent model training
- --p_neg 负样本选择 / Negative sample selection
- 1:1 平衡 / 1:1 balance
- 评估指标 / Evaluation metrics
- main: 主入口 / Main entry point

输入 / Inputs:
- json/human_plant_3x3_v2.json: 训练配置 / Training config
- human3/seq.npy, plant3/seq.npy, zero/seq.npy: 人类/植物/零数据 / Human/plant/zero data
- 命令行参数 / CLI: --config, --gpu, --p_neg {zero,plant}

输出 / Outputs:
- checkpoints/3x3_v2_*.pt: 3 个模型 / 3 models
- logs/3x3_v2_*/results.json: 评估结果 / Evaluation results

数据流 / Data Flow:
1. 加载数据 / Load data
2. 根据 --p_neg 选择负样本 / Choose negatives based on --p_neg
3. 训练 3 个独立二分类模型 / Train 3 independent binary models
4. 零样本评估 / Zero-shot evaluation
5. 保存结果 / Save results

相关文件 / Related Files:
- 调用 / Calls: model.main_model, dataset.{human,plant,plant_single}
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python 3x3_2.py --config json/human_plant_3x3_v2.json --p_neg plant --gpu 0

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
import copy
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from model.main_model import RNA_ClassQuery_Model
from dataset.human import Mer100Dataset
from dataset.plant_single import PlantSingleDataset
from utils.common import (
    GROUP_TO_INDEX,
    INDEX_TO_NUCLEOTIDE,
    load_config,
    MOD_NAMES,
    get_smoothed_pos_weights,
)
from utils.metrics import get_all_predictions


CONFIG_PATH = "json/human2.json"
DEFAULT_TEST_EVERY_N_ITERATIONS = 1000
DEFAULT_NUM_EPOCHS = 10
OUTPUT_DIR = "output"
TARGET_CLASSES = [
    # {
    #     "index": 5,
    #     "name": "Y",
    #     "test_every_n_iterations": 100,
    #     "num_epochs": 2,
    # },
    # {
    #     "index": 8,
    #     "name": "m5C",
    #     "test_every_n_iterations": 100,
    #     "num_epochs": 10,
    # },
    {
        "index": 9,
        "name": "m6A",
        "test_every_n_iterations": 200,
        "num_epochs": 1,
        "human_exclusive_only": True,
    },
]
FIG_DPI = 300
RANDOM_SEED = 666
DEFAULT_BATCH_SIZE = 32
TRAIN_RATIO = 0.7
COMMON_ATTENTION_MASK_KEYS = ["attn_mask_A", "attn_mask_C", "attn_mask_G", "attn_mask_U", "attn_mask_N"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="3x3 binary training for human-to-plant zero-shot transfer."
    )
    parser.add_argument(
        "--p_neg",
        choices=["zero", "plant"],
        default="zero",
        help="Negative source for plant tasks: 'zero' samples or 'plant' other-modification samples.",
    )
    return parser.parse_args()


def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def load_human_data(Config):
    print(f"\nLoading human dataset from {Config.data.human_data_dir}...")
    dataset = Mer100Dataset(
        mode="train",
        data_dir=Config.data.human_data_dir,
        cache_dir=Config.data.cache_dir,
        use_human3=True,
        use_cache=True,
    )
    print(f"Human dataset loaded: {len(dataset)} samples")

    cache_stats = dataset.get_cache_stats()
    if not cache_stats["batch_cache"].get("exists", False):
        print("Precomputing secondary structures...")
        dataset.precompute_all_structures(
            batch_size=100, num_workers=None, show_progress=True
        )
    else:
        print(f"Batch cache exists: {cache_stats['batch_cache']['path']}")

    return dataset


def load_plant_full_dataset(Config):
    print(f"\nLoading full plant+zero dataset...")
    full_dataset = PlantSingleDataset(
        plant_dir=Config.data.plant_data_dir,
        zero_dir=getattr(Config.data, "zero_data_dir", "npy/zero"),
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True,
    )
    return full_dataset


# -----------------------------------------------------------------------------
# Common utility functions for balanced binary splitting
# -----------------------------------------------------------------------------

def balanced_binary_split(pos_indices, neg_indices, train_ratio=0.7, seed=RANDOM_SEED):
    """
    Split positive and negative indices into balanced train and test sets.

    Each split (train, test) will have equal number of positive and negative samples.

    Args:
        pos_indices: list of all positive sample indices
        neg_indices: list of all negative sample indices
        train_ratio: fraction of data to use for training
        seed: random seed for reproducibility

    Returns:
        tuple: (train_indices, test_indices)
            - train_indices: balanced training set with equal pos/neg
            - test_indices: balanced test set with equal pos/neg
    """
    np.random.seed(seed)
    random.seed(seed)

    pos_indices = np.array(pos_indices)
    neg_indices = np.array(neg_indices)

    # Shuffle both sets
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # Split positive into train/test
    n_pos_total = len(pos_indices)
    n_pos_train = int(n_pos_total * train_ratio)
    pos_train = pos_indices[:n_pos_train]
    pos_test = pos_indices[n_pos_train:]

    # Split negative into train/test - match the count from positive
    n_neg_total = len(neg_indices)
    n_neg_train = n_pos_train
    n_neg_test = len(pos_test)

    # Sample negatives for train first, then sample test negatives from the
    # remaining pool when possible. Fall back to replacement when the pool is
    # too small so both splits are always defined.
    if n_neg_train > n_neg_total:
        warnings.warn(
            f"Not enough negative samples: need {n_neg_train}, have {n_neg_total}. "
            f"Using sampling with replacement."
        )
        neg_train = np.random.choice(neg_indices, size=n_neg_train, replace=True)
        neg_indices_remaining = neg_indices
    else:
        neg_train = neg_indices[:n_neg_train]
        neg_indices_remaining = neg_indices[n_neg_train:]

    if n_neg_test > len(neg_indices_remaining):
        warnings.warn(
            f"Not enough negative samples for test: need {n_neg_test}, have {len(neg_indices_remaining)}. "
            f"Using sampling with replacement."
        )
        neg_test = np.random.choice(neg_indices_remaining, size=n_neg_test, replace=True)
    else:
        neg_test = neg_indices_remaining[:n_neg_test]

    # Combine
    train_indices = np.concatenate([pos_train, neg_train]).tolist()
    test_indices = np.concatenate([pos_test, neg_test]).tolist()

    # Shuffle again
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    print(f"  Balanced split: train={len(train_indices)} ({len(pos_train)} pos, {len(neg_train)} neg), "
          f"test={len(test_indices)} ({len(pos_test)} pos, {len(neg_test)} neg)")

    return train_indices, test_indices


def split_indices(indices, train_ratio=0.7, seed=RANDOM_SEED):
    """Deterministically split a list of indices into train/test."""
    np.random.seed(seed)
    indices = np.array(indices)
    np.random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    return indices[:split_point].tolist(), indices[split_point:].tolist()


def get_group_index_for_class(target_class):
    """Map a 12-class modification index to its 4-class group index."""
    nucleotide = INDEX_TO_NUCLEOTIDE[target_class]
    return GROUP_TO_INDEX[nucleotide]


class DataFieldNormalizer(torch.utils.data.Dataset):
    """Wrap a dataset so every PyG Data sample exposes the same field set."""

    def __init__(self, dataset, default_is_plant=False):
        self.dataset = dataset
        self.default_is_plant = default_is_plant

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if not hasattr(data, "is_plant"):
            data.is_plant = bool(self.default_is_plant)
        if not hasattr(data, "real_idx"):
            data.real_idx = int(idx)

        seq_len = int(data.x.size(0))
        zero_mask = torch.zeros(seq_len, dtype=torch.float32)
        for key in COMMON_ATTENTION_MASK_KEYS:
            if not hasattr(data, key):
                setattr(data, key, zero_mask.clone())

        return data


def get_human_positive_indices(human_dataset, target_class, exclusive_only=False):
    """Return human positive indices for a target class with optional exclusivity."""
    if hasattr(human_dataset, "y_12class"):
        all_targets = np.asarray(human_dataset.y_12class).astype(np.int8, copy=False)
    else:
        all_targets = []
        for i in range(len(human_dataset)):
            data = human_dataset[i]
            all_targets.append(data.y[0].cpu().numpy())
        all_targets = np.asarray(all_targets, dtype=np.int8)

    all_labels = all_targets[:, target_class]
    pos_mask = all_labels == 1

    if exclusive_only:
        pos_mask &= all_targets.sum(axis=1) == 1

    return np.where(pos_mask)[0].tolist()


def get_plant_positive_indices(plant_full_dataset, target_class):
    """Return plant positive indices for a target class."""
    return plant_full_dataset.get_plant_indices_by_class(target_class)


def get_plant_other_modification_negative_indices(plant_full_dataset, target_class):
    """
    Return plant negative indices for a target class.

    Negatives are restricted to plant samples that carry at least one other
    modification while not containing the target modification itself.
    """
    plant_labels = np.asarray(plant_full_dataset.plant_y12).astype(np.int8, copy=False)
    neg_mask = (plant_labels[:, target_class] == 0) & (plant_labels.sum(axis=1) > 0)
    return np.where(neg_mask)[0].tolist()


def build_shared_zero_splits(plant_full_dataset, train_count, test_count, seed=RANDOM_SEED):
    """
    Sample a shared ordered zero pool for one task.

    The returned train/test lists are ordered so human and plant can each take
    a prefix while still drawing from the same zero sampling sequence.
    """
    zero_train_pool, zero_test_pool = plant_full_dataset.get_zero_split(
        test_ratio=1 - TRAIN_RATIO, seed=seed
    )
    rng = np.random.default_rng(seed)

    def sample_pool(pool, sample_count, split_name):
        pool = np.asarray(pool)
        if sample_count <= len(pool):
            sampled = rng.permutation(pool)[:sample_count]
        else:
            warnings.warn(
                f"Not enough zero samples for {split_name}: need {sample_count}, "
                f"have {len(pool)}. Using sampling with replacement."
            )
            sampled = rng.choice(pool, size=sample_count, replace=True)
        return sampled.tolist()

    return {
        "train": sample_pool(zero_train_pool, train_count, "train"),
        "test": sample_pool(zero_test_pool, test_count, "test"),
    }


# -----------------------------------------------------------------------------
# Human per-class loader builder
# -----------------------------------------------------------------------------

def build_human_binary_loaders(
    human_dataset,
    plant_full_dataset,
    target_class,
    train_ratio=0.7,
    batch_size=DEFAULT_BATCH_SIZE,
    exclusive_only=False,
    shared_zero_splits=None,
):
    """
    Build balanced binary train/test loaders for a specific target class on human data.

    Positive: all samples where y[:, target_class] == 1
    Negative: all samples where y[:, target_class] == 0
    Train and test are each 1:1 balanced.

    Args:
        human_dataset: full Mer100Dataset
        target_class: class index (5, 8, 9)
        train_ratio: train split ratio
        batch_size: batch size

    Returns:
        dict: {'train_loader': ..., 'test_loader': ...}
    """
    pos_indices = get_human_positive_indices(
        human_dataset, target_class, exclusive_only=exclusive_only
    )

    print(f"\nBuilding human binary loader for class {target_class}:")
    print(f"  Total positives: {len(pos_indices)}")
    if exclusive_only:
        print("  Positive filter: exclusive-only")

    pos_train, pos_test = split_indices(pos_indices, train_ratio=train_ratio, seed=RANDOM_SEED)
    n_neg_train = len(pos_train)
    n_neg_test = len(pos_test)

    if shared_zero_splits is None:
        raise ValueError("shared_zero_splits must be provided for human binary loaders")

    neg_train = shared_zero_splits["train"][:n_neg_train]
    neg_test = shared_zero_splits["test"][:n_neg_test]

    print(
        f"  Balanced split: train={len(pos_train) + len(neg_train)} "
        f"({len(pos_train)} pos, {len(neg_train)} neg), "
        f"test={len(pos_test) + len(neg_test)} ({len(pos_test)} pos, {len(neg_test)} neg)"
    )

    train_subset = ConcatDataset(
        [
            DataFieldNormalizer(Subset(human_dataset, pos_train), default_is_plant=False),
            DataFieldNormalizer(Subset(plant_full_dataset, neg_train), default_is_plant=False),
        ]
    )
    test_subset = ConcatDataset(
        [
            DataFieldNormalizer(Subset(human_dataset, pos_test), default_is_plant=False),
            DataFieldNormalizer(Subset(plant_full_dataset, neg_test), default_is_plant=False),
        ]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_pos_indices': pos_train,
        'test_pos_indices': pos_test,
        'train_neg_indices': neg_train,
        'test_neg_indices': neg_test,
    }


# -----------------------------------------------------------------------------
# Plant per-class loader builder
# -----------------------------------------------------------------------------

def build_plant_binary_loaders(
    plant_full_dataset,
    target_class,
    train_ratio=0.7,
    batch_size=DEFAULT_BATCH_SIZE,
    p_neg="plant",
    shared_zero_splits=None,
):
    """
    Build balanced binary train/test loaders for a specific target class on plant data.

    Positive: plant samples where target_class == 1 (from get_plant_indices_by_class)
    Negative:
        - p_neg='plant': plant samples with other modifications, excluding the target class
        - p_neg='zero': all-zero samples from the zero pool

    Args:
        plant_full_dataset: full PlantSingleDataset
        target_class: class index (5, 8, 9)
        train_ratio: train split ratio for positives
        batch_size: batch size

    Returns:
        dict: {'train_loader': ..., 'test_loader': ...}
    """
    # Get positive plant indices for this class
    pos_indices = get_plant_positive_indices(plant_full_dataset, target_class)
    print(f"\nBuilding plant binary loader for class {target_class}:")
    print(f"  Total plant positives: {len(pos_indices)}")

    pos_train, pos_test = split_indices(pos_indices, train_ratio=train_ratio, seed=RANDOM_SEED)

    if p_neg == "zero":
        if shared_zero_splits is None:
            raise ValueError("shared_zero_splits must be provided when p_neg='zero'")

        neg_train = shared_zero_splits["train"][:len(pos_train)]
        neg_test = shared_zero_splits["test"][:len(pos_test)]
        print(
            f"  Negative source: zero "
            f"(train={len(neg_train)}, test={len(neg_test)})"
        )
    elif p_neg == "plant":
        neg_indices = get_plant_other_modification_negative_indices(
            plant_full_dataset, target_class
        )
        print(f"  Total plant other-modification negatives: {len(neg_indices)}")

        train_indices, test_indices = balanced_binary_split(
            pos_indices, neg_indices, train_ratio=train_ratio, seed=RANDOM_SEED
        )

        pos_set = set(pos_indices)
        pos_train = [idx for idx in train_indices if idx in pos_set]
        neg_train = [idx for idx in train_indices if idx not in pos_set]
        pos_test = [idx for idx in test_indices if idx in pos_set]
        neg_test = [idx for idx in test_indices if idx not in pos_set]
    else:
        raise ValueError(f"Unsupported p_neg value: {p_neg}")

    if len(pos_train) != len(neg_train) or len(pos_test) != len(neg_test):
        raise ValueError(
            f"Plant splits are not balanced for class {target_class}: "
            f"train=({len(pos_train)} pos, {len(neg_train)} neg), "
            f"test=({len(pos_test)} pos, {len(neg_test)} neg)"
        )

    train_subset = ConcatDataset(
        [
            DataFieldNormalizer(Subset(plant_full_dataset, pos_train), default_is_plant=True),
            DataFieldNormalizer(Subset(plant_full_dataset, neg_train), default_is_plant=True),
        ]
    )
    test_subset = ConcatDataset(
        [
            DataFieldNormalizer(Subset(plant_full_dataset, pos_test), default_is_plant=True),
            DataFieldNormalizer(Subset(plant_full_dataset, neg_test), default_is_plant=True),
        ]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_pos_indices': pos_train,
        'test_pos_indices': pos_test,
        'train_neg_indices': neg_train,
        'test_neg_indices': neg_test,
    }


# -----------------------------------------------------------------------------
# Binary evaluation
# -----------------------------------------------------------------------------

def extract_task_logit(model_output, use_hierarchical):
    """Extract the single active task logit from a pruned one-head model."""
    if use_hierarchical and isinstance(model_output, tuple):
        logits = model_output[0]
    else:
        logits = model_output if not isinstance(model_output, tuple) else model_output[0]

    if logits.dim() == 1:
        return logits
    if logits.size(-1) != 1:
        raise ValueError(f"Expected a single active classification head, got logits shape {tuple(logits.shape)}")
    return logits.squeeze(-1)


def evaluate_binary_task(model, test_loader, target_class, device, use_hierarchical):
    """
    Evaluate binary classification task for a single target class.

    Args:
        model: model (can output 12 dims, we only extract target_class)
        test_loader: test loader
        target_class: target class index to evaluate
        device: device
        use_hierarchical: whether model uses hierarchical output

    Returns:
        auc: AUC score, 0.5 if only one class present
        y_true: numpy array of true binary labels
        y_prob: numpy array of predicted probabilities
    """
    model.eval()
    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        for batch in test_loader:
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.batch)
            logit = extract_task_logit(out, use_hierarchical)
            # Get probability from logit
            prob = torch.sigmoid(logit)

            # Get true label - extract target class from 12-dim label
            true = batch.y[:, target_class].view(-1).cpu().numpy()
            prob = prob.cpu().numpy()

            all_y_true.extend(true)
            all_y_prob.extend(prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    # Compute AUC with safety fallback
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(all_y_true)) < 2:
            print(f"    Warning: only one class present in test set, AUC=0.5")
            return 0.5, all_y_true, all_y_prob
        auc = roc_auc_score(all_y_true, all_y_prob)
    except Exception as e:
        print(f"    Warning: AUC computation failed: {e}, returning 0.5")
        auc = 0.5

    return auc, all_y_true, all_y_prob


# -----------------------------------------------------------------------------
# Single binary task training
# -----------------------------------------------------------------------------

class BinaryTask:
    """Container for a single binary classification task."""
    def __init__(self, task_info, human_loaders, plant_loaders):
        from typing import Optional
        self.class_index = task_info['index']
        self.class_name = task_info['name']
        self.test_every_n_iterations = task_info.get('test_every_n_iterations', DEFAULT_TEST_EVERY_N_ITERATIONS)
        self.num_epochs = task_info.get('num_epochs', DEFAULT_NUM_EPOCHS)
        self.batch_size = task_info.get('batch_size', DEFAULT_BATCH_SIZE)
        self.human_loaders = human_loaders
        self.plant_loaders = plant_loaders
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.criterion: Optional[nn.Module] = None


def create_model_for_task(Config, task_class_index):
    """Create a fresh one-head model for a binary task."""
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=Config.num_classes,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=Config.use_hierarchical,
        use_layer_norm=Config.use_layer_norm,
    ).to(Config.device)

    if Config.use_hierarchical:
        model.prune_heads(
            valid_class_indices=[task_class_index],
            valid_group_indices=[get_group_index_for_class(task_class_index)],
        )
    else:
        model.prune_heads(valid_class_indices=[task_class_index])

    return model


def train_binary_task(task: BinaryTask, Config):
    """Train a single binary task, record AUCs at intervals.
    Uses per-task configuration: test_every_n_iterations, num_epochs from task."""
    device = Config.device
    use_hierarchical = Config.use_hierarchical
    use_amp = Config.use_amp
    test_every_n_iterations = task.test_every_n_iterations
    num_epochs = task.num_epochs

    # Get the loader
    train_loader = task.human_loaders['train_loader']
    human_test_loader = task.human_loaders['test_loader']
    plant_test_loader = task.plant_loaders['test_loader']

    # Create fresh model, optimizer, criterion
    task.model = create_model_for_task(Config, task.class_index)
    task.criterion = nn.BCEWithLogitsLoss()  # Single class, no pos weight needed for balanced data

    task.optimizer = optim.AdamW(
        task.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    task.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        task.optimizer, T_max=num_epochs, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in task.model.parameters())
    print(f"\n  Model created: {total_params:,} parameters")
    print(f"  Training: {len(train_loader)} batches per epoch, {num_epochs} epochs")
    print(f"  Evaluation every {test_every_n_iterations} iterations")

    records = []
    global_step = 0

    print(f"\n  {'=' * 50}")
    print(f"  Starting training for {task.class_name} (class {task.class_index})")
    print(f"  {'=' * 50}")

    for epoch in range(1, num_epochs + 1):
        print(f"\n  Epoch {epoch}/{num_epochs}")
        task.scheduler.step()

        pbar = tqdm(train_loader, desc=f"  {task.class_name} Epoch {epoch}", leave=False)
        for batch in pbar:
            # Train step - binary loss only on target class
            task.model.train()

            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)
            batch = batch.to(device)
            task.optimizer.zero_grad()

            # Forward pass
            if use_amp and torch.cuda.is_available():
                scaler = torch.amp.GradScaler('cuda')
                with torch.amp.autocast('cuda'):
                    out = task.model(batch.x, batch.edge_index, batch.batch)
                    logit = extract_task_logit(out, use_hierarchical)
                    target = batch.y[:, task.class_index].view(-1)
                    loss = task.criterion(logit, target)
                scaler.scale(loss).backward()
                scaler.step(task.optimizer)
                scaler.update()
            else:
                out = task.model(batch.x, batch.edge_index, batch.batch)
                logit = extract_task_logit(out, use_hierarchical)
                target = batch.y[:, task.class_index].view(-1)
                loss = task.criterion(logit, target)
                loss.backward()
                task.optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            # Evaluation
            if global_step % test_every_n_iterations == 0:
                task.model.eval()
                human_auc, _, _ = evaluate_binary_task(
                    task.model, human_test_loader, task.class_index, device, use_hierarchical
                )
                plant_auc, _, _ = evaluate_binary_task(
                    task.model, plant_test_loader, task.class_index, device, use_hierarchical
                )

                record = {
                    "task_name": task.class_name,
                    "class_index": task.class_index,
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    f"human_auc": human_auc,
                    f"plant_auc": plant_auc,
                }
                records.append(record)

                print(f"\n    [Step {global_step}] "
                      f"Human AUC={human_auc:.4f}, Plant AUC={plant_auc:.4f}")

    print(f"\n  Training complete for {task.class_name}, total {global_step} iterations, {len(records)} evaluations")
    return records


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    setup_output_dir()
    args = parse_args()

    global Config, config_dict
    Config, config_dict = load_config(CONFIG_PATH)

    Config.random_seed = RANDOM_SEED
    Config.batch_size = DEFAULT_BATCH_SIZE
    Config.use_hierarchical = config_dict.get("model", {}).get("use_hierarchical", True)

    print(f"\n{'=' * 60}")
    print("3x3 Independent Binary Tasks: Human-to-Plant Zero-Shot Transfer")
    print(f"{'=' * 60}")
    print(f"Device: {Config.device}")
    print(f"Random seed: {Config.random_seed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Plant negative source (--p_neg): {args.p_neg}")
    print(f"Target tasks:")
    for t in TARGET_CLASSES:
        print(f"  - {t['name']} (index={t['index']}, test_every={t.get('test_every_n_iterations', DEFAULT_TEST_EVERY_N_ITERATIONS)}, epochs={t.get('num_epochs', DEFAULT_NUM_EPOCHS)})")

    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load data
    human_dataset = load_human_data(Config)
    plant_full_dataset = load_plant_full_dataset(Config)

    # Build tasks - each with its own human/plant loaders
    tasks = []
    for task_info in TARGET_CLASSES:
        print(f"\n{'=' * 60}")
        print(f"Building task: {task_info['name']} (class {task_info['index']})")
        print(f"{'=' * 60}")

        exclusive_only = task_info.get("human_exclusive_only", False)
        human_pos_indices = get_human_positive_indices(
            human_dataset, task_info["index"], exclusive_only=exclusive_only
        )
        plant_pos_indices = get_plant_positive_indices(plant_full_dataset, task_info["index"])

        raw_human_pos_indices = None
        if exclusive_only:
            raw_human_pos_indices = get_human_positive_indices(
                human_dataset, task_info["index"], exclusive_only=False
            )

        human_pos_train, human_pos_test = split_indices(
            human_pos_indices, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
        )
        plant_pos_train, plant_pos_test = split_indices(
            plant_pos_indices, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
        )

        shared_zero_splits = build_shared_zero_splits(
            plant_full_dataset,
            train_count=max(len(human_pos_train), len(plant_pos_train)),
            test_count=max(len(human_pos_test), len(plant_pos_test)),
            seed=RANDOM_SEED + task_info["index"],
        )

        print(
            f"Shared zero prefixes for {task_info['name']}: "
            f"train={len(shared_zero_splits['train'])}, "
            f"test={len(shared_zero_splits['test'])}"
        )
        print(f"Task summary [{task_info['name']}]:")
        print(
            f"  human positives used: total={len(human_pos_indices)}, "
            f"train={len(human_pos_train)}, test={len(human_pos_test)}"
        )
        print(
            f"  plant positives used: total={len(plant_pos_indices)}, "
            f"train={len(plant_pos_train)}, test={len(plant_pos_test)}"
        )
        print(
            f"  shared zero used: train={len(shared_zero_splits['train'])}, "
            f"test={len(shared_zero_splits['test'])}"
        )
        print(f"  plant negative source: {args.p_neg}")
        if raw_human_pos_indices is not None:
            print(
                f"  m6A exclusive filter: kept={len(human_pos_indices)} / "
                f"raw={len(raw_human_pos_indices)}, "
                f"removed={len(raw_human_pos_indices) - len(human_pos_indices)}"
            )

        # Build human loaders for this task - use per-task batch size if specified
        task_batch_size = task_info.get('batch_size', DEFAULT_BATCH_SIZE)
        human_loaders = build_human_binary_loaders(
            human_dataset,
            plant_full_dataset,
            task_info['index'],
            train_ratio=TRAIN_RATIO,
            batch_size=task_batch_size,
            exclusive_only=exclusive_only,
            shared_zero_splits=shared_zero_splits,
        )

        # Build plant loaders for this task
        plant_loaders = build_plant_binary_loaders(
            plant_full_dataset,
            task_info['index'],
            train_ratio=TRAIN_RATIO,
            batch_size=task_batch_size,
            p_neg=args.p_neg,
            shared_zero_splits=shared_zero_splits,
        )

        task = BinaryTask(task_info, human_loaders, plant_loaders)
        tasks.append(task)

    print(f"\n{'=' * 60}")
    print(f"All {len(tasks)} tasks built, starting training...")
    print(f"{'=' * 60}")

    # Train each task independently and collect all records
    all_records = []
    for task in tasks:
        task_records = train_binary_task(task, Config)
        # Reshape records to final format with full column names
        for rec in task_records:
            full_rec = {
                "task_name": rec["task_name"],
                "class_index": rec["class_index"],
                "global_step": rec["global_step"],
                "epoch": rec["epoch"],
                "loss": rec["loss"],
                f"human_{rec['task_name']}_auc": rec["human_auc"],
                f"plant_{rec['task_name']}_auc": rec["plant_auc"],
            }
            all_records.append(full_rec)

    # Save combined records
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(OUTPUT_DIR, "3x3_binary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"All training complete!")
    print(f"Combined metrics saved: {csv_path}")
    print(f"{'=' * 60}")

    # Display final results
    print("\nFinal AUC results:")
    for task in tasks:
        # Get last evaluation for this task
        task_records = [r for r in all_records if r['task_name'] == task.class_name]
        if task_records:
            last = task_records[-1]
            print(f"  {task.class_name}: "
                  f"Human={last[f'human_{task.class_name}_auc']:.4f}, "
                  f"Plant={last[f'plant_{task.class_name}_auc']:.4f}")

    # Generate 3x3 correlation plots
    generate_3x3_plots(all_records)

    print(f"\n{'=' * 60}")
    print("All outputs saved to:", OUTPUT_DIR)
    print(f"  - 3x3_binary_metrics.csv")
    print(f"  - 3x3_auc_matrix.png")
    print(f"  - 3x3_auc_matrix.pdf")
    print(f"  - 3x3_correlation.csv")
    print(f"{'=' * 60}")


def generate_3x3_plots(all_records):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib import rcParams
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    font_candidates = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]
    selected_font = next((font for font in font_candidates if font in available_fonts or font == "serif"), "serif")

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.size"] = 18

    morandi_colors = ["#b39bb0", "#b0c4a4", "#d4b8a0", "#9bb0c8", "#c8b8d4", "#d4c8b8"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.08, wspace=0.25, hspace=0.3
    )

    # Convert records to DataFrame
    df = pd.DataFrame(all_records)
    task_names = [t['name'] for t in TARGET_CLASSES]

    correlation_results = []

    for i, human_class in enumerate(task_names):
        for j, plant_class in enumerate(task_names):
            ax = axes[i, j]
            col_human = f"human_{human_class}_auc"
            col_plant = f"plant_{plant_class}_auc"

            # Get data points - for each evaluation step of human_class
            # we have the corresponding plant_class AUC from the same step
            # Filter records for this human task
            # In our refactored version, each row has all three AUCs
            if col_human in df.columns and col_plant in df.columns:
                x = df[col_human].to_numpy(dtype=float)
                y = df[col_plant].to_numpy(dtype=float)
            else:
                # Column not present, skip
                x = np.array([])
                y = np.array([])

            if len(x) == 0:
                continue

            valid_mask = np.isfinite(x) & np.isfinite(y)
            x = x[valid_mask]
            y = y[valid_mask]

            if len(x) == 0:
                continue

            color = morandi_colors[(i * 3 + j) % len(morandi_colors)]

            ax.scatter(
                x, y, c=color, s=50, alpha=0.7, edgecolors="white", linewidth=0.5
            )

            has_variation = len(np.unique(x)) > 1 and len(np.unique(y)) > 1
            if len(x) > 2 and has_variation:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(
                        x_line,
                        p(x_line),
                        color="#666666",
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                    )
                except np.linalg.LinAlgError:
                    continue

                spearman_r, spearman_p = spearmanr(x, y)
                pearson_r, pearson_p = pearsonr(x, y)

                sig_str = ""
                if spearman_p < 0.001:
                    sig_str = "***"
                elif spearman_p < 0.01:
                    sig_str = "**"
                elif spearman_p < 0.05:
                    sig_str = "*"

                label = (
                    f"Spearman r={spearman_r:.3f}{sig_str}\nPearson r={pearson_r:.3f}"
                )
                ax.text(
                    0.05,
                    0.95,
                    label,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    fontfamily=selected_font,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"
                    ),
                )

                correlation_results.append(
                    {
                        "human_class": human_class,
                        "plant_class": plant_class,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                    }
                )

            if i == 2:
                ax.set_xlabel(
                    f"Plant {plant_class}", fontsize=18, fontfamily=selected_font
                )
            if j == 0:
                ax.set_ylabel(
                    f"Human {human_class}", fontsize=18, fontfamily=selected_font
                )

            ax.tick_params(labelsize=14)
            ax.set_xlim([0.3, 1.0])
            ax.set_ylim([0.3, 1.0])
            ax.grid(True, alpha=0.3, linestyle="--")

    corr_csv_path = os.path.join(OUTPUT_DIR, "3x3_correlation.csv")
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Correlation CSV saved: {corr_csv_path}")

    png_path = os.path.join(OUTPUT_DIR, "3x3_auc_matrix.png")
    pdf_path = os.path.join(OUTPUT_DIR, "3x3_auc_matrix.pdf")

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"PNG saved: {png_path}")
    print(f"PDF saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
