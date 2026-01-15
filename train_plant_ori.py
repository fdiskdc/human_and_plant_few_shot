import os
import random
import json
import numpy as np
import torch
import argparse
import copy
from datetime import datetime

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as PyGBatch
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

from model.main_model import RNA_ClassQuery_Model
from dataset.plant import PlantDataset
from utils import (
    setup_logging, setup_tensorboard,
    save_checkpoint, load_config,
    evaluate_plant_unbalance, evaluate_plant_balanceb,
    get_all_predictions
)
from utils.common import GROUP_TO_CLASS_INDICES, INDEX_TO_GROUP
from utils.logging import print_few_shot_results

# Import few-shot utilities
from utils.few_shot import (
    prepare_training_batch
)

# [REMOVED] reset_hierarchical_head function is removed to preserve pre-trained knowledge.

def sample_support_set(full_dataset, train_indices, k, valid_classes, random_seed=None):
    if k == 0 or k == '0':
        return []

    if random_seed is not None:
        np.random.seed(random_seed)

    y_12class = full_dataset.y_12class
    support_indices = []

    for c in valid_classes:
        c_pos_in_train = [idx for idx in train_indices if y_12class[idx, c] == 1]
        if len(c_pos_in_train) >= k:
            selected = np.random.choice(c_pos_in_train, k, replace=False)
            support_indices.extend(selected.tolist())
        else:
            support_indices.extend(c_pos_in_train)

    return support_indices

def train_few_shot(model, support_data_list, valid_classes, config, device, logger,
                   k_shot, support_masks=None, original_state_dict=None, eval_data_list=None):
    
    # [CRITICAL CHANGE] Do NOT reset the head. 
    # We want to start from the pre-trained weights which are already good for Class 8 & 9.
    
    # Split Strategy: Different hyperparameters for Low vs High Shot
    
    if k_shot <= 10:
        # === Low Shot (1-10) ===
        # Strategy: Linear Probing (Frozen Backbone)
        # Goal: Adapt the classifier boundary for Class 5 without destroying Class 8/9 features.
        logger.info(f"Shot {k_shot} Strategy: Linear Probing (Frozen Backbone, Keep Pre-trained Head)")
        
        # Freeze Backbone
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze Head
        for param in model.class_query_head.parameters():
            param.requires_grad = True
            
        trainable_params = model.class_query_head.parameters()
        
        # Hyperparameters
        # Use moderate LR to tweak existing weights, not random ones
        ft_lr = 1e-3      
        ft_epochs = 50
        
    else:
        # === High Shot (50-100) ===
        # Strategy: Full Fine-tuning (Unfrozen Backbone)
        # Goal: Optimize features and classifier with sufficient data.
        logger.info(f"Shot {k_shot} Strategy: Full Fine-tuning (Unfrozen Backbone)")
        
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
            
        trainable_params = model.parameters()
        
        # Hyperparameters
        # Very low LR to preserve pre-trained knowledge
        ft_lr = 5e-5      
        ft_epochs = 50
    
    optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Data Augmentation Factor
    aug_factor = 4 if k_shot <= 10 else 2

    # Prepare Batch
    batch_list = prepare_training_batch(
        support_data_list, support_masks if support_masks else [], aug_factor, k_shot,
        shuffle=True, logger=None
    )
    
    model.train()
    
    mini_batch_size = 64
    if len(batch_list) > 0:
        num_mini_batches = (len(batch_list) + mini_batch_size - 1) // mini_batch_size
    else:
        num_mini_batches = 0
    
    # Training Loop
    for epoch in range(ft_epochs):
        epoch_loss = 0.0
        random.shuffle(batch_list)
        batches_processed = 0
        
        for i in range(num_mini_batches):
            batch_slice = batch_list[i*mini_batch_size : (i+1)*mini_batch_size]
            if not batch_slice: continue
            
            batch = PyGBatch.from_data_list(batch_slice).to(device)
            
            optimizer.zero_grad()
            
            if config.use_hierarchical:
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Loss Calculation: Only on valid classes [5, 8, 9]
            loss = criterion(logits[:, valid_classes], batch.y[:, valid_classes])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
            
        if batches_processed > 0 and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / batches_processed
            logger.info(f"Shot {k_shot} Ep {epoch+1}/{ft_epochs}: Loss {avg_loss:.4f}")

    return model

def train_few_shot_pruned(model, support_data_list, valid_class_indices, valid_group_indices, config, device, logger, k_shot):
    """
    Train using Dynamic Head Pruning with Surgical Freezing and Weight Interpolation.
    """
    # 0. Save original state for weight interpolation (The "Anchor")
    original_head_state = copy.deepcopy(model.class_query_head.state_dict())

    # 1. Prune the model heads
    model.prune_heads(valid_class_indices, valid_group_indices)

    # 2. Advanced Freeze/Unfreeze Strategy
    trainable_params = []

    if k_shot <= 10:
        logger.info(f"Shot {k_shot}: Low Shot Strategy - Surgical Freezing (Frozen Backbone & MLPs, Train Projections Only)")

        # Freeze EVERYTHING first
        for param in model.parameters():
            param.requires_grad = False

        # Only Unfreeze the Output Projections in the Head
        # We keep 'group_queries' and 'group_mlps' FROZEN to preserve derivation logic
        for name, param in model.class_query_head.named_parameters():
            if "output_proj" in name: # Only train the final classifiers
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        ft_lr = 1e-3 # Higher LR is safe for linear probing
        ft_epochs = 50
        aug_factor = 4

        # Enable Weight Interpolation for Low Shot
        use_weight_interpolation = True
        interpolation_alpha = 0.5 # Blend 50% original weights back in to prevent overfitting

    else:
        logger.info(f"Shot {k_shot}: High Shot Strategy - Full Fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
            trainable_params.append(param)

        ft_lr = 5e-5 # Lower LR for full fine-tuning
        ft_epochs = 50
        aug_factor = 2
        use_weight_interpolation = False

    # 3. Optimizer
    optimizer = optim.AdamW(trainable_params, lr=ft_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Data Preparation
    batch_list = prepare_training_batch(support_data_list, [], aug_factor, k_shot, shuffle=True)

    model.train()
    mini_batch_size = 64
    if len(batch_list) > 0:
        num_mini_batches = (len(batch_list) + mini_batch_size - 1) // mini_batch_size
    else:
        num_mini_batches = 0

    # 5. Training Loop
    for epoch in range(ft_epochs):
        epoch_loss = 0.0
        random.shuffle(batch_list)
        batches_processed = 0

        for i in range(num_mini_batches):
            batch_slice = batch_list[i*mini_batch_size : (i+1)*mini_batch_size]
            if not batch_slice: continue

            batch = PyGBatch.from_data_list(batch_slice).to(device)
            optimizer.zero_grad()

            logits_class, logits_group = model(batch.x, batch.edge_index, batch.batch)
            target_class = batch.y[:, valid_class_indices]

            loss_class = criterion(logits_class, target_class)

            if config.use_hierarchical and hasattr(batch, 'y_4class'):
                 target_group = batch.y_4class[:, valid_group_indices]
                 loss_group = criterion(logits_group, target_group)
                 loss = loss_class + loss_group
            else:
                 loss = loss_class

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0 and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / batches_processed
            logger.info(f"Shot {k_shot} Ep {epoch+1}/{ft_epochs}: Loss {avg_loss:.4f}")

    # 6. Weight Interpolation (Safety Net for Low Shots)
    if use_weight_interpolation:
        logger.info(f"Applying Weight Interpolation (alpha={interpolation_alpha})...")
        with torch.no_grad():
            current_head_state = model.class_query_head.state_dict()
            for name, param in current_head_state.items():
                if name in original_head_state:
                    # w_new = (1 - alpha) * w_trained + alpha * w_original
                    # pulling weights back towards the "Human" prior
                    param.copy_((1 - interpolation_alpha) * param + interpolation_alpha * original_head_state[name])

    return model

def main(config_path='json/plant.json', checkpoint_path=None):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name + '_plant_fewshot_pruned')
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_plant_fewshot_pruned')

    # Load Dataset
    full_dataset = PlantDataset(
        plant_dir=Config.data.plant_data_dir,
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )
    
    # Precompute Check
    cache_stats = full_dataset.get_cache_stats()
    if not cache_stats['batch_cache'].get('exists', False):
         full_dataset.precompute_all_structures(batch_size=100)

    # Define Plant-Specific Indices
    # Classes: 5(Y), 8(m5C), 9(m6A)
    plant_class_indices = [5, 8, 9]
    # Groups: 0(A), 1(C), 3(U) (Note: G is index 2 and is unused)
    plant_group_indices = [0, 1, 3]

    # Valid classes: 5(Y), 8(m5C), 9(m6A)
    valid_classes = [5, 8, 9]
    
    y_12class = full_dataset.y_12class
    train_indices = []
    test_indices = []
    np.random.seed(Config.random_seed)
    
    for c in valid_classes:
        c_pos_indices = np.where(y_12class[:, c] == 1)[0]
        np.random.shuffle(c_pos_indices)
        
        split_idx = int(0.1 * len(c_pos_indices))
        train_indices.extend(c_pos_indices[:split_idx].tolist())
        test_indices.extend(c_pos_indices[split_idx:].tolist())
        
    train_indices = list(set(train_indices))
    test_indices = list(set(test_indices))
    test_indices = [i for i in test_indices if i not in train_indices]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Train Pool: {len(train_indices)}, Test Set: {len(test_indices)}")

    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=12,  # Init with 12
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=True,  # Must be True for this strategy
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    # Load Pre-trained Weights (CHECKPOINT IS MANDATORY)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded Human Pre-trained Weights from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found! Pruning requires a pre-trained model.")
    
    original_state_dict = copy.deepcopy(model.state_dict())

    shot_counts = [0, 1, 3, 5, 7, 10, 50, 100]
    all_results = {}
    
    for k in shot_counts:
        logger.info(f"--- Running {k}-Shot Learning (Pruned) ---")

        # Reset model to 12-class state
        model.load_state_dict(original_state_dict,strict=False)
        # Clear any pruning buffers from previous iterations if necessary
        if hasattr(model.class_query_head, 'valid_class_indices'):
            del model.class_query_head.valid_class_indices
            del model.class_query_head.valid_group_indices

        if k == 0:
            # Zero-shot: Just prune and eval
            model.prune_heads(plant_class_indices, plant_group_indices)
        else:
            # Sample Support Set
            support_indices = sample_support_set(
                full_dataset, train_indices, k, plant_class_indices,
                random_seed=Config.random_seed + k
            )
            support_data_list = [full_dataset[i] for i in support_indices]

            # Train (Pruning happens inside)
            model = train_few_shot_pruned(
                model, support_data_list, plant_class_indices, plant_group_indices,
                Config, Config.device, logger, k
            )

        # Inference & Evaluation
        # Note: Model output is now (Batch, 3)
        model.eval()
        all_y_true = []
        all_y_prob = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(Config.device)
                logits_3, _ = model(batch.x, batch.edge_index, batch.batch)
                probs_3 = torch.sigmoid(logits_3)

                # Get True Labels for the 3 specific classes
                y_true_12 = batch.y
                y_true_3 = y_true_12[:, plant_class_indices]

                all_y_true.append(y_true_3.cpu())
                all_y_prob.append(probs_3.cpu())

        y_true = torch.cat(all_y_true, dim=0).numpy()
        y_prob = torch.cat(all_y_prob, dim=0).numpy()

        # Calculate Metrics (Need to adjust evaluate function or just calc manual for 3 classes)
        # Re-using evaluate_plant_unbalance but we need to trick it or adapt it.
        # Since y_true is now (N, 3), we need to map it back to (N, 12) sparse for the evaluator,
        # OR just calculate F1 directly here.
        # For simplicity, let's map back to 12-dim zero-filled for compatibility.

        N = y_true.shape[0]
        y_true_12_sparse = np.zeros((N, 12))
        y_prob_12_sparse = np.zeros((N, 12))

        y_true_12_sparse[:, plant_class_indices] = y_true
        y_prob_12_sparse[:, plant_class_indices] = y_prob

        # Generate y_4class from y_true_12_sparse (same logic as get_all_predictions)
        y_4class = np.zeros((N, 4), dtype=np.float32)
        for group_idx in range(4):
            nucleotide = INDEX_TO_GROUP[group_idx]  # 'A', 'C', 'G', or 'U'
            class_indices = GROUP_TO_CLASS_INDICES[nucleotide]
            y_4class[:, group_idx] = y_true_12_sparse[:, class_indices].max(axis=1)

        # Now we can use the existing evaluator
        metrics_unbalance = evaluate_plant_unbalance(
            y_true_12_sparse, y_prob_12_sparse, Config.device, y_4class,
            random_seed=Config.random_seed
        )
        metrics_balanceb = evaluate_plant_balanceb(
            y_true_12_sparse, y_prob_12_sparse, y_4class, Config.device,
            random_seed=Config.random_seed
        )

        logger.info(f"{k}-Shot Unbalance Macro F1: {metrics_unbalance.get('group_plant_opt_macro_f1', 0):.4f}")
        all_results[k] = {'unbalance': metrics_unbalance, 'balanceb': metrics_balanceb}

        if tb_writer:
            tb_writer.add_scalar('few_shot/macro_f1', metrics_unbalance.get('group_plant_opt_macro_f1', 0), k)

    print_few_shot_results(all_results, 0, logger)

    results_path = os.path.join(Config.checkpoint_dir, 'plant_fewshot_pruned_results.json')
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = {}
    for k, v in all_results.items():
        serializable_results[str(k)] = convert_to_serializable(v)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    if tb_writer: tb_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='json/plant.json')
    parser.add_argument('--checkpoint', default="logs/rna_classification_20260111_111223/checkpoints/epoch_010.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)