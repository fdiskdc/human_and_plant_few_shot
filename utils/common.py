"""
Common utilities and constants for RNA Multi-label Classification Training

This module contains:
- Hierarchical classification constants
- Configuration loading
- Model checkpointing
- Helper functions
"""

import os
import json
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from torch_geometric.loader import DataLoader

# ============================================================================
# Hierarchical Classification Constants
# ============================================================================

# 12类修饰名称映射 (模型索引 -> 修饰名称)
# 与 human_make_npy.py 中的 MOD_TO_INDEX 一致 (mod_index - 1 转换后)
MOD_NAMES = {
    0: 'Am',     1: 'Atol',   2: 'Cm',      # 索引 0-2
    3: 'Gm',     4: 'Tm',     5: 'Y',       # 索引 3-5
    6: 'ac4C',   7: 'm1A',    8: 'm5C',     # 索引 6-8
    9: 'm6A',    10: 'm6Am',  11: 'm7G'     # 索引 9-11
}

# 4-Class Group Names (Nucleotide Groups)
NUCLEOTIDE_GROUP_NAMES = ['A', 'C', 'G', 'U']

# Model Index (0-11) to Nucleotide Group Name
INDEX_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',          # Am, Atol
    2: 'C',                  # Cm
    3: 'G',                  # Gm
    4: 'U', 5: 'U',          # Tm, Y
    6: 'C',                  # ac4C
    7: 'A',                  # m1A
    8: 'C',                  # m5C
    9: 'A', 10: 'A',         # m6A, m6Am
    11: 'G'                  # m7G
}

# Group Name to original IDs (for reference)
NUCLEOTIDE_GROUPS = {
    'A': [1, 2, 8, 10, 11],
    'C': [3, 7, 9],
    'G': [4, 12],
    'U': [5, 6]
}

# Group to 4-Class Index Mapping (Order for y_4class)
GROUP_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# Index to Group Mapping (reverse of above)
INDEX_TO_GROUP = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Number of classes per group (for hierarchical derivation)
GROUP_SIZES = {
    'A': 5,  # Am(0), Atol(1), m1A(7), m6A(9), m6Am(10)
    'C': 3,  # Cm(2), ac4C(6), m5C(8)
    'G': 2,  # Gm(3), m7G(11)
    'U': 2   # Tm(4), Y(5)
}

# Mapping from group to the indices (0-11) of classes in that group
GROUP_TO_CLASS_INDICES = {
    'A': [0, 1, 7, 9, 10],    # Am, Atol, m1A, m6A, m6Am
    'C': [2, 6, 8],           # Cm, ac4C, m5C
    'G': [3, 11],             # Gm, m7G
    'U': [4, 5]               # Tm, Y
}

# ============================================================================
# Plant Dataset Constants
# ============================================================================

# Plant dataset only has 3 valid classes (Y, m5C, m6A)
# These are the class indices (0-11) that should be evaluated for Plant data
PLANT_VALID_CLASS_INDICES = [5, 8, 9]  # Y (class 5), m5C (class 8), m6A (class 9)


# ============================================================================
# Helper Functions
# ============================================================================

def get_center_nucleotide(sequence: str) -> str:
    """
    Get the center nucleotide (index 500) from a sequence.

    Args:
        sequence: RNA sequence string (length 1001)

    Returns:
        Nucleotide character ('A', 'C', 'G', or 'U')
    """
    if len(sequence) >= 501:
        return sequence[500].upper()
    return 'A'  # Default fallback


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = 'model.json') -> Tuple:
    """
    Load configuration from JSON file and create timestamped experiment folder.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Config object and config_dict
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create timestamped experiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config_dict.get('experiment_name', 'rna_classification')
    exp_folder = f"{exp_name}_{timestamp}"

    # Update paths with timestamped folder
    base_log_dir = config_dict.get('paths', {}).get('log_dir', './logs')

    log_dir = os.path.join(base_log_dir, exp_folder)
    # Save checkpoints inside the log folder
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration to experiment folder for reproducibility
    config_save_path = os.path.join(log_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Define Config class
    class Config:
        """Global configuration class"""
        pass

    # Set paths - support both old and new config format
    data_cfg = config_dict.get('data', {})
    
    # For backward compatibility, if 'data' section doesn't exist, use old format
    if data_cfg:
        # New format with separate data paths
        Config.data_dir = data_cfg.get('human_data_dir', './human3')
        Config.human_data_dir = data_cfg.get('human_data_dir', './human3')
        Config.plant_data_dir = data_cfg.get('plant_data_dir', './plant')
        Config.cache_dir = data_cfg.get('cache_dir', './cache')
    else:
        # Old format (backward compatibility)
        Config.data_dir = config_dict.get('data_dir', './human3')
        Config.human_data_dir = Config.data_dir
        Config.plant_data_dir = './plant'
        Config.cache_dir = './cache'
    
    Config.checkpoint_dir = checkpoint_dir
    Config.log_dir = log_dir
    Config.experiment_name = exp_name
    Config.timestamp = timestamp
    
    # Create Config.data object for easier access
    class DataConfig:
        pass
    DataConfig.human_data_dir = Config.human_data_dir
    DataConfig.plant_data_dir = Config.plant_data_dir
    DataConfig.cache_dir = Config.cache_dir
    Config.data = DataConfig

    # Set model parameters
    model_cfg = config_dict.get('model', {})
    Config.cnn_hidden_dim = model_cfg.get('cnn_hidden_dim', 64)
    Config.cnn_kernel_sizes = tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7]))
    Config.cnn_dropout = model_cfg.get('cnn_dropout', 0.1)
    Config.gcn_hidden_dim = model_cfg.get('gcn_hidden_dim', 128)
    Config.gcn_out_channels = model_cfg.get('gcn_out_channels', 128)
    Config.gcn_num_layers = model_cfg.get('gcn_num_layers', 3)
    Config.gcn_dropout = model_cfg.get('gcn_dropout', 0.3)
    Config.num_classes = model_cfg.get('num_classes', 12)
    Config.num_attn_heads = model_cfg.get('num_attn_heads', 4)
    Config.attn_dropout = model_cfg.get('attn_dropout', 0.1)
    Config.use_simple_pooling = model_cfg.get('use_simple_pooling', False)
    Config.use_hierarchical = model_cfg.get('use_hierarchical', False)
    Config.use_layer_norm = model_cfg.get('use_layer_norm', True)

    # Set training parameters
    train_cfg = config_dict.get('training', {})
    Config.batch_size = train_cfg.get('batch_size', 32)
    Config.num_epochs = train_cfg.get('num_epochs', 100)
    Config.learning_rate = train_cfg.get('learning_rate', 1e-3)
    Config.weight_decay = train_cfg.get('weight_decay', 1e-4)
    Config.warmup_epochs = train_cfg.get('warmup_epochs', 5)
    Config.train_ratio = train_cfg.get('train_ratio', 0.7)
    Config.random_seed = train_cfg.get('random_seed', 42)
    Config.eval_threshold = train_cfg.get('eval_threshold', 0.5)
    Config.test_interval = train_cfg.get('test_interval', 1)
    Config.save_every_epoch = train_cfg.get('save_every_epoch', True)
    Config.save_best_only = train_cfg.get('save_best_only', False)

    # Dynamic sampler parameters
    Config.use_dynamic_sampler = train_cfg.get('use_dynamic_sampler', True)
    Config.balance_ratio = train_cfg.get('balance_ratio', 0.3)

    # Device
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return Config, config_dict


# ============================================================================
# Model Checkpointing
# ============================================================================

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict, filepath: str,
                   logger: Optional[logging.Logger] = None,
                   config_dict: Optional[Dict] = None):
    """
    Save model checkpoint with model parameters.

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        logger: Optional logger instance
        config_dict: Optional configuration dictionary to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config_dict
    }
    torch.save(checkpoint, filepath)

    msg = f"Checkpoint saved to {filepath}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   filepath: str, device: torch.device) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        filepath: Path to checkpoint file
        device: Device to load checkpoint onto

    Returns:
        Dictionary containing epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False
) -> float:
    """
    Train for one epoch with TQDM progress monitoring.

    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Optional logger instance
        use_hierarchical: If True, use multi-task learning (4-class + 12-class)

    Returns:
        Average loss for the epoch
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=True)
    for batch in pbar:
        # Ensure labels are tensor before moving to device
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float32)

        batch = batch.to(device)

        # Ensure labels are on same device
        batch.y = batch.y.to(device)

        # Forward pass
        optimizer.zero_grad()

        if use_hierarchical:
            # Multi-task learning: get both 12-class and 4-class logits
            from .common import GROUP_TO_CLASS_INDICES

            logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)

            # Generate 4-class labels from 12-class labels
            # y_4class[g] = 1 if any class in group g has modification
            y_12 = batch.y  # (Batch, 12)
            y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

            # For each group (A=0, C=1, G=2, U=3)
            for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                class_indices = GROUP_TO_CLASS_INDICES[group_name]
                y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

            # Calculate losses for both tasks
            # Note: criterion has pos_weight for 12 classes, so we can only use it for 12-class loss
            # For 4-class loss, we use BCEWithLogitsLoss without pos_weight
            loss_12 = criterion(logits_12, y_12)
            loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
            loss = loss_12 + loss_4

            total_loss_12 += loss_12.item()
            total_loss_4 += loss_4.item()
        else:
            # Single-task learning: only 12-class
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if use_hierarchical:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / num_batches
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg = f"Train loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})"
    else:
        msg = f"Train loss: {avg_loss:.4f}"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss


def test_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    phase: str = "test",
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False
) -> float:
    """
    Test for one epoch with TQDM progress monitoring.

    Args:
        model: The model to evaluate
        dataloader: Test/validation dataloader
        criterion: Loss function
        device: Device to run evaluation on
        phase: Phase name ('test' or 'val')
        logger: Optional logger instance
        use_hierarchical: If True, use multi-task learning (4-class + 12-class)

    Returns:
        Average loss for the epoch
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    with torch.no_grad():
        for batch in pbar:
            # Ensure labels are tensor before moving to device
            if not isinstance(batch.y, torch.Tensor):
                batch.y = torch.tensor(batch.y, dtype=torch.float32)

            batch = batch.to(device)

            # Ensure labels are on same device
            batch.y = batch.y.to(device)

            # Forward pass
            if use_hierarchical:
                # Multi-task learning: get both 12-class and 4-class logits
                from .common import GROUP_TO_CLASS_INDICES

                logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)

                # Generate 4-class labels from 12-class labels
                y_12 = batch.y  # (Batch, 12)
                y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                # For each group (A=0, C=1, G=2, U=3)
                for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                    class_indices = GROUP_TO_CLASS_INDICES[group_name]
                    y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                # Calculate losses for both tasks
                # Note: criterion has pos_weight for 12 classes, so we can only use it for 12-class loss
                # For 4-class loss, we use BCEWithLogitsLoss without pos_weight
                loss_12 = criterion(logits_12, y_12)
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                loss = loss_12 + loss_4

                total_loss_12 += loss_12.item()
                total_loss_4 += loss_4.item()
            else:
                # Single-task learning: only 12-class
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)

            total_loss += loss.item()
            num_batches += 1

            if use_hierarchical:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})"
    else:
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f}"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss