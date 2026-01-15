"""
Plant 3-Way Multi-class Classification using Rectified Prototypical Networks

This script implements metric-based few-shot learning using PROTOTYPE RECTIFICATION
(TIP-style blending of pre-trained priors and few-shot empirical prototypes):

1. NO gradient-based fine-tuning - model is a FIXED feature extractor
2. Classification based on feature distance (Cosine Similarity) to class prototypes
3. PROTOTYPE RECTIFICATION: p_rect = alpha * p_prior + (1-alpha) * p_empirical
   - p_prior: Pre-trained class query embeddings (0-shot knowledge)
   - p_empirical: Mean of support set embeddings (few-shot knowledge)
   - alpha: Blending coefficient (default 0.7, bias towards pre-trained)
4. Temperature-scaled softmax for probability calibration

Key Innovation - Prototype Rectification:
- Standard prototypical networks discard pre-trained classification head weights
- This approach PRESERVES the 0-shot knowledge and refines it with few-shot data
- Addresses the issue where more shots can lead to WORSE performance
- Formula: p_rect = 0.7 * p_prior + 0.3 * p_empirical (recommended)

Plant Class Mappings:
- Class 5 (Y) -> Label 0
- Class 8 (m5C) -> Label 1
- Class 9 (m6A) -> Label 2

Dataset Structure:
- Plant: plant/seq.npy, plant/12loc.npy, plant/4loc.npy, plant/1001loc.npy
"""

import os
import random
import json
import numpy as np
import torch
import torch_geometric
import argparse
import copy
from datetime import datetime
from prettytable import PrettyTable
import torch.nn.functional as F

# Set GPU to use first device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as PyGBatch, Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from model.main_model import RNA_ClassQuery_Model
from dataset.plant_single import PlantSingleDataset
from utils import (
    setup_logging, setup_tensorboard,
    save_checkpoint, load_config,
    get_all_predictions
)
from utils.common import GROUP_TO_CLASS_INDICES, INDEX_TO_GROUP
from utils.logging import print_few_shot_results

# Import few-shot utilities
from utils.few_shot import prepare_training_batch
import threading
import queue


# ============================================================================
# Feature Extractor Wrapper
# ============================================================================

class FeatureExtractor(nn.Module):
    """
    Wrapper to extract embeddings from the pre-trained RNA_ClassQuery_Model.

    This intercepts the forward pass and returns the node features after the
    GCN backbone, BEFORE the classification head. These embeddings are used
    for computing prototypes and distances.
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        # The model has cnn_block and gcn_block as feature extractors
        # The class_query_head is used for classification
        self.cnn_block = original_model.cnn_block
        self.gcn_block = original_model.gcn_block

    def forward(self, x, edge_index, batch=None):
        """
        Extract features from the backbone (CNN + GCN).

        Args:
            x: Node features (Batch, Seq_Len, 4) or (Total_Nodes, 4)
            edge_index: Edge indices (2, Num_Edges)
            batch: Batch vector (Total_Nodes,)

        Returns:
            features: Node embeddings (Total_Nodes, Hidden_Dim)
        """
        # Handle Data/Batch objects
        if isinstance(x, Data) or isinstance(x, PyGBatch):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        # Ensure correct shape
        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            assert seq_len == 1001, f"Expected sequence length 1001, got {seq_len}"
            if batch is None:
                batch = torch.arange(
                    batch_size, device=x.device
                ).repeat_interleave(seq_len)
        elif x.dim() == 2:
            assert batch is not None, "batch vector must be provided for PyG format input"
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Forward through CNN and GCN (feature extraction only)
        node_features = self.cnn_block(x, batch)
        node_features = self.gcn_block(node_features, edge_index)

        return node_features


# ============================================================================
# Pre-trained Prototype Extraction
# ============================================================================

def get_pretrained_prototypes(model, target_classes, device):
    """
    Extracts class query embeddings from the pre-trained classification head
    to serve as Prior Prototypes (0-Shot knowledge).

    The pre-trained class queries contain rich class priors learned from
    large-scale data. These serve as excellent initialization points
    that can be refined with few-shot support data.

    Args:
        model: The pre-trained RNA_ClassQuery_Model (frozen)
        target_classes: List of target class indices [5, 8, 9]
        device: Device to run on

    Returns:
        dict: {class_id: prior_prototype_tensor} for each target class
    """
    prior_prototypes = {}

    # Access the class query head
    if hasattr(model, 'class_query_head'):
        head = model.class_query_head

        # Handle different head types
        if hasattr(head, 'class_queries'):
            # Standard ClassQueryHead or ClassQueryHeadPooling
            all_queries = head.class_queries.data  # (12, Hidden_Dim)
        elif hasattr(head, 'group_queries') and hasattr(head, '_derive_class_queries'):
            # HierarchicalClassQueryHeadPooling - need to derive class queries from group queries
            all_queries = head._derive_class_queries()  # (12, Hidden_Dim)
        else:
            raise ValueError(f"Unknown class query head type: {type(head)}")

        # Extract and normalize queries for target classes
        with torch.no_grad():
            for cls in target_classes:
                # Extract weight for specific class
                w = all_queries[cls].unsqueeze(0).to(device)  # (1, Hidden_Dim)

                # L2 normalize to create unit vector prototype
                w = F.normalize(w, p=2, dim=1)

                prior_prototypes[cls] = w

    return prior_prototypes


# ============================================================================
# Prototype Computation with Rectification
# ============================================================================

def compute_rectified_prototypes(feature_extractor, support_data_list, target_classes,
                                  device, prior_prototypes=None, alpha=0.7, return_support_mean=False):
    """
    Computes rectified prototype vectors by blending pre-trained priors
    with empirical support set prototypes.

    Formula: p_rect = alpha * p_prior + (1 - alpha) * p_empirical
    - alpha=1.0: Pure 0-shot (use only pre-trained weights)
    - alpha=0.0: Pure prototypical network (use only support means)
    - alpha=0.7: Recommended - bias towards pre-trained knowledge with few-shot refinement

    NEW: Feature Centering (CL2N)
    - Computes the global mean of all support samples (regardless of class)
    - This support_mean is used to center both prototypes and query features
    - Reduces the "hubness" problem in high-dimensional spaces

    Args:
        feature_extractor: Frozen feature extractor model
        support_data_list: List of support Data objects
        target_classes: List of target class indices [5, 8, 9]
        device: Device to run on
        prior_prototypes: Dict of pre-trained class query prototypes (from get_pretrained_prototypes)
        alpha: Blending coefficient [0, 1], default 0.7
        return_support_mean: If True, return (prototypes, support_mean) for CL2N centering

    Returns:
        dict: {class_id: rectified_prototype_tensor} for each target class
        tuple: (prototypes, support_mean) if return_support_mean=True
    """
    feature_extractor.eval()
    prototypes = {}

    # First, compute empirical prototypes from support set (standard protonet approach)
    class_samples = {c: [] for c in target_classes}
    for data in support_data_list:
        # data.y is (1, 12), find which class is 1
        label = torch.argmax(data.y).item()
        if label in target_classes:
            class_samples[label].append(data)

    # Store all support embeddings for computing global mean (CL2N)
    all_support_embeddings = []

    with torch.no_grad():
        for cls in target_classes:
            samples = class_samples[cls]
            if not samples:
                print(f"Warning: No samples found for class {cls}")
                # Fallback to prior prototype if available
                if prior_prototypes and cls in prior_prototypes:
                    prototypes[cls] = prior_prototypes[cls]
                continue

            # Create a batch for this class
            batch = PyGBatch.from_data_list(samples).to(device)

            # Get Embeddings from feature extractor
            # Shape: (Total_Nodes, Hidden_Dim)
            node_embeddings = feature_extractor(batch.x, batch.edge_index, batch.batch)

            # Global mean pooling to get graph-level embeddings
            num_samples = len(samples)
            embeddings_list = []

            for i in range(num_samples):
                # Get nodes belonging to sample i
                mask = batch.batch == i
                sample_nodes = node_embeddings[mask]  # (Num_Nodes_i, Hidden_Dim)

                # Mean pooling over nodes
                graph_emb = torch.mean(sample_nodes, dim=0, keepdim=True)  # (1, Hidden_Dim)
                embeddings_list.append(graph_emb)

            # Stack all embeddings for this class
            # Shape: (Num_Samples, Hidden_Dim)
            embeddings = torch.cat(embeddings_list, dim=0)

            # Normalize embeddings (Crucial for Cosine Similarity)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Store for global mean computation (CL2N)
            all_support_embeddings.append(embeddings)

            # Calculate Empirical Prototype (mean of support embeddings)
            # Shape: (1, Hidden_Dim)
            p_empirical = torch.mean(embeddings, dim=0, keepdim=True)

            # Re-normalize the empirical prototype
            p_empirical = F.normalize(p_empirical, p=2, dim=1)

            # ====================================================================
            # PROTOTYPE RECTIFICATION (TIP-style blending)
            # ====================================================================
            if prior_prototypes and cls in prior_prototypes:
                p_prior = prior_prototypes[cls]

                # Blend prior and empirical prototypes
                # Both are already normalized, so we can directly interpolate
                p_rect = alpha * p_prior + (1 - alpha) * p_empirical

                # Re-normalize the rectified prototype
                p_rect = F.normalize(p_rect, p=2, dim=1)

                prototypes[cls] = p_rect
            else:
                # No prior available, use empirical prototype
                prototypes[cls] = p_empirical

    # ========================================================================
    # FEATURE CENTERING (CL2N): Compute Global Support Mean
    # ========================================================================
    support_mean = None
    if return_support_mean and len(all_support_embeddings) > 0:
        # Concatenate all support embeddings across all classes
        all_embeddings = torch.cat(all_support_embeddings, dim=0)  # (Total_Support, Hidden_Dim)

        # Compute global mean (mean of all support samples regardless of class)
        support_mean = torch.mean(all_embeddings, dim=0, keepdim=True)  # (1, Hidden_Dim)

    if return_support_mean:
        return prototypes, support_mean
    return prototypes


# Backward compatibility alias
def compute_prototypes(feature_extractor, support_data_list, target_classes, device):
    """
    Legacy function for backward compatibility.
    Uses standard prototypical network approach (alpha=0, no prior).
    """
    return compute_rectified_prototypes(
        feature_extractor, support_data_list, target_classes,
        device, prior_prototypes=None, alpha=0.0
    )


# ============================================================================
# Metric-Based Prediction
# ============================================================================

def predict_with_prototypes(feature_extractor, batch, prototypes, target_classes, device, temperature=10.0):
    """
    Predicts classes based on Cosine Similarity to prototypes.

    For each query sample:
    1. Extract features using the frozen feature extractor
    2. Normalize the query embedding
    3. Compute cosine similarity to each class prototype
    4. Apply temperature-scaled softmax to get probabilities

    Args:
        feature_extractor: Frozen feature extractor model
        batch: Query batch Data object
        prototypes: Dict of {class_id: prototype_tensor}
        target_classes: List of target class indices [5, 8, 9]
        device: Device to run on
        temperature: Temperature scaling for softmax (higher = sharper predictions)

    Returns:
        probs: (Batch_Size, 3) probability matrix
        similarities: (Batch_Size, 3) raw similarity scores
    """
    feature_extractor.eval()
    with torch.no_grad():
        # 1. Get Query Features (graph-level via mean pooling)
        node_features = feature_extractor(batch.x, batch.edge_index, batch.batch)

        # Global mean pooling for each graph in the batch
        batch_size = batch.batch.max().item() + 1
        query_feats_list = []

        for i in range(batch_size):
            mask = batch.batch == i
            sample_nodes = node_features[mask]
            graph_emb = torch.mean(sample_nodes, dim=0, keepdim=True)
            query_feats_list.append(graph_emb)

        query_feats = torch.cat(query_feats_list, dim=0)  # (Batch_Size, Hidden_Dim)
        query_feats = F.normalize(query_feats, p=2, dim=1)

        # 2. Stack Prototypes in consistent order [5, 8, 9]
        proto_stack = torch.cat([prototypes[c] for c in target_classes], dim=0)  # (3, Hidden_Dim)

        # 3. Calculate Cosine Similarity
        # (Batch_Size, Hidden_Dim) @ (Hidden_Dim, 3) -> (Batch_Size, 3)
        similarities = torch.mm(query_feats, proto_stack.t())

        # 4. Convert to Probabilities with temperature scaling
        probs = torch.softmax(temperature * similarities, dim=1)

        return probs, similarities


# ============================================================================
# Plant Class to Label Mapping
# ============================================================================

PLANT_CLASS_MAPPING = {
    5: {'class_name': 'Y', 'group_idx': 3, 'group_name': 'U', '3way_label': 0},
    8: {'class_name': 'm5C', 'group_idx': 1, 'group_name': 'C', '3way_label': 1},
    9: {'class_name': 'm6A', 'group_idx': 0, 'group_name': 'A', '3way_label': 2}
}


def sample_support_set_3way(dataset, plant_train_indices, k, target_classes):
    """
    Sample support set for 3-way multi-class classification.

    Samples K shots for EACH of the 3 classes.
    Total samples = 3*K (K from each class).

    Args:
        dataset: PlantSingleDataset instance containing Plant data
        plant_train_indices: List of available Plant training indices
        k: Number of shots per class (total samples = 3*k)
        target_classes: List of target class IDs [5, 8, 9]

    Returns:
        list: Support indices (balanced, total 3*k samples)
    """
    if k == 0 or k == '0':
        return []

    support_indices = []

    for cls_idx in target_classes:
        # Get indices for this specific class
        c_indices = dataset.get_plant_indices_by_class(cls_idx)
        valid_c_indices = list(set(c_indices) & set(plant_train_indices))

        # Sample K items from this class
        if len(valid_c_indices) >= k:
            selected_from_class = np.random.choice(valid_c_indices, k, replace=False).tolist()
        else:
            # If not enough samples, sample with replacement
            selected_from_class = np.random.choice(valid_c_indices, k, replace=True).tolist()
            print(f"Warning: Only {len(valid_c_indices)} samples available for class {cls_idx}, using replacement for {k} samples")

        support_indices.extend(selected_from_class)

    # Shuffle to mix classes
    np.random.shuffle(support_indices)

    return support_indices


def calculate_binary_metrics(y_true, y_pred, y_prob):
    """
    Calculate detailed binary metrics for One-vs-Rest evaluation.
    y_true: 0 or 1
    y_pred: 0 or 1
    y_prob: float (probability of positive class)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Sensitivity (Recall) & Specificity
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC / AUPRC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    return {
        'F1': f1, 'Prec': prec, 'Rec': rec, 'Acc': acc,
        'AUC': auc, 'AUPRC': auprc, 'Sn': sn, 'Sp': sp,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }


def evaluate_3way_detailed(feature_extractor, prototypes, test_loader, device, target_classes, temperature=10.0):
    """
    Run inference and compute One-vs-Rest metrics for all 3 classes.

    Uses metric-based prediction (prototypical networks) instead of
    gradient-based classification.

    Args:
        feature_extractor: Frozen feature extractor model
        prototypes: Dict of class prototypes
        test_loader: DataLoader for test set
        device: Device to evaluate on
        target_classes: List of target class indices [5, 8, 9]
        temperature: Temperature scaling for softmax

    Returns:
        Dictionary containing One-vs-Rest metrics for each class {class_id: {metrics_dict}}
    """
    feature_extractor.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    # Mappings: 0 -> Class 5, 1 -> Class 8, 2 -> Class 9
    class_to_label = {5: 0, 8: 1, 9: 2}

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Metric-based prediction using prototypes
            probs, similarities = predict_with_prototypes(
                feature_extractor, batch, prototypes, target_classes, device, temperature
            )

            # Get predictions
            preds = torch.argmax(probs, dim=1)

            # Map labels from (N, 12) one-hot to 0, 1, 2
            raw = batch.y.argmax(dim=1)
            target_labels = torch.zeros_like(raw)
            for k, v in class_to_label.items():
                target_labels[raw == k] = v

            # Filter out samples that don't belong to our 3 classes
            valid_mask = batch.y[:, target_classes].sum(dim=1) > 0

            all_probs.append(probs[valid_mask].cpu())
            all_preds.append(preds[valid_mask].cpu())
            all_labels.append(target_labels[valid_mask].cpu())

    all_probs = torch.cat(all_probs).numpy()  # Shape (N, 3)
    all_preds = torch.cat(all_preds).numpy()   # Shape (N,)
    all_labels = torch.cat(all_labels).numpy() # Shape (N,)

    results = {}

    # Loop over classes 0, 1, 2 (corresponding to 5, 8, 9)
    map_idx_to_real = {0: 5, 1: 8, 2: 9}

    for idx in range(3):
        real_cls = map_idx_to_real[idx]

        # Create One-vs-Rest Binary targets
        y_bin_true = (all_labels == idx).astype(int)
        y_bin_pred = (all_preds == idx).astype(int)
        y_bin_prob = all_probs[:, idx]

        metrics = calculate_binary_metrics(y_bin_true, y_bin_pred, y_bin_prob)
        results[real_cls] = metrics

    return results


def evaluate_3way_transductive(feature_extractor, prototypes, test_loader, device, target_classes,
                                support_mean=None, temperature=10.0, alpha_transductive=0.5,
                                num_iterations=2):
    """
    TRANSDUCTIVE PROTOTYPE REFINEMENT (TPR) with CL2N Feature Centering.

    This is an improved transductive inference method that:
    1. Uses CL2N (Centered Local 2-Normal) feature centering to reduce hubness
    2. Iteratively refines prototypes using the unlabelled test set structure
    3. Adapts prototypes to better represent the test distribution

    Key Innovation: The support set mean is NOT a good estimator of the test class center.
    We must use the unlabelled test set structure to refine the prototypes transductively.

    Algorithm:
    - Step 1: Extract ALL test features (global mode, no batch-by-batch)
    - Step 2 (CL2N): Calculate global mean of Support Set. Subtract this mean from both
             Prototypes and Test Features. This reduces the "hubness" problem.
    - Step 3 (Refinement Loop):
        * Predict soft probabilities using cosine similarity
        * Update prototypes: Compute weighted mean of test features based on predictions
        * Interpolate: P_new = alpha * P_old + (1-alpha) * P_test
        * Repeat for 2 iterations (optimal for most cases)
    - Step 4: Final prediction with refined prototypes

    Args:
        feature_extractor: Frozen feature extractor model
        prototypes: Dict of class prototypes (rectified from support set)
        test_loader: DataLoader for test set
        device: Device to evaluate on
        target_classes: List of target class indices [5, 8, 9]
        support_mean: Global mean of support features for CL2N centering (1, Hidden_Dim)
        temperature: Temperature scaling for softmax (higher = sharper predictions)
        alpha_transductive: Interpolation coefficient for prototype refinement [0, 1]
                            Higher = trust initial prototypes, Lower = trust query statistics
        num_iterations: Number of refinement iterations (default 2 is optimal)

    Returns:
        Dictionary containing One-vs-Rest metrics for each class {class_id: {metrics_dict}}
    """
    feature_extractor.eval()

    # ========================================================================
    # STEP 1: EXTRACT ALL TEST FEATURES (GLOBAL MODE)
    # ========================================================================
    print("Extracting global test features for transductive inference...")
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extraction"):
            batch = batch.to(device)

            # Get node features from feature extractor
            node_feats = feature_extractor(batch.x, batch.edge_index, batch.batch)

            # Global mean pooling per graph
            batch_size = batch.batch.max().item() + 1
            for i in range(batch_size):
                mask = batch.batch == i
                if mask.sum() > 0:
                    graph_emb = node_feats[mask].mean(dim=0)
                    all_feats.append(graph_emb)
                else:
                    # Handle edge case where a sample has no nodes
                    all_feats.append(torch.zeros(node_feats.shape[1], device=device))

            # Labels mapping: Map from (N, 12) one-hot to 0, 1, 2
            raw = batch.y.argmax(dim=1)
            mapped = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Map 5->0, 8->1, 9->2
            mapping = {5: 0, 8: 1, 9: 2}
            for k, v in mapping.items():
                mapped[raw == k] = v
            all_labels.append(mapped)

    # Convert to tensors
    Q = torch.stack(all_feats)  # (N_test, Dim)
    Y = torch.cat(all_labels)   # (N_test,)

    # ========================================================================
    # STEP 2: CL2N (FEATURE CENTERING)
    # ========================================================================
    # Subtract Support Mean from both Prototypes and Query
    mu = support_mean.to(device) if support_mean is not None else None

    # Normalize and center query features
    Q = F.normalize(Q, p=2, dim=1)
    if mu is not None:
        Q = Q - mu
        Q = F.normalize(Q, p=2, dim=1)

    # Stack prototypes in consistent order [5, 8, 9]
    P = torch.cat([prototypes[c] for c in target_classes], dim=0).clone()  # (3, Dim)

    # Center prototypes
    if mu is not None:
        P = P - mu
        P = F.normalize(P, p=2, dim=1)

    # ========================================================================
    # STEP 3: TRANSDUCTIVE REFINEMENT LOOP
    # ========================================================================
    print(f"Running {num_iterations} iterations of transductive prototype refinement...")

    for iteration in range(num_iterations):
        # Predict soft probabilities using cosine similarity
        sim = torch.mm(Q, P.t())  # (N_test, 3)
        probs = torch.softmax(temperature * sim, dim=1)  # (N_test, 3)

        # Re-estimate centers from Test Data (Soft K-Means)
        P_test = torch.zeros_like(P)
        for c in range(3):
            # Weighted mean of test features using probabilities as weights
            weights = probs[:, c].unsqueeze(1)  # (N_test, 1)
            if weights.sum() > 0:
                P_test[c] = (weights * Q).sum(dim=0) / weights.sum()
            else:
                # If no samples assigned to this class, keep original prototype
                P_test[c] = P[c]

        # Normalize the test-derived prototypes
        P_test = F.normalize(P_test, p=2, dim=1)

        # Interpolate: P_new = alpha * P_old + (1-alpha) * P_test
        P = alpha_transductive * P + (1 - alpha_transductive) * P_test
        P = F.normalize(P, p=2, dim=1)

        print(f"  Iteration {iteration + 1}/{num_iterations} complete")

    # ========================================================================
    # STEP 4: FINAL PREDICTION
    # ========================================================================
    final_sim = torch.mm(Q, P.t())
    final_probs = torch.softmax(temperature * final_sim, dim=1)

    # Get predictions
    final_preds = torch.argmax(final_probs, dim=1)

    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================
    results = {}
    map_idx_to_real = {0: 5, 1: 8, 2: 9}

    for idx in range(3):
        real_cls = map_idx_to_real[idx]

        # Create One-vs-Rest Binary targets
        y_bin_true = (Y == idx).cpu().numpy().astype(int)
        y_bin_pred = (final_preds == idx).cpu().numpy().astype(int)
        y_bin_prob = final_probs[:, idx].cpu().numpy()

        metrics = calculate_binary_metrics(y_bin_true, y_bin_pred, y_bin_prob)
        results[real_cls] = metrics

    return results


def print_detailed_3way_results(all_results_history, logger):
    """
    Print separate tables for Class 5, 8, and 9.
    all_results_history structure: { k_shot: { class_id: {metric_dict} } }

    Args:
        all_results_history: Results dictionary for all shot counts
        logger: Logger instance
    """
    class_info = {
        5: "Class 5 (Y)",
        8: "Class 8 (m5C)",
        9: "Class 9 (m6A)"
    }

    shot_keys = sorted(all_results_history.keys(), key=lambda x: int(x) if x!='full' else 999)

    for cls_id in [5, 8, 9]:
        logger.info(f"\n{'='*100}")
        logger.info(f"Detailed Results: {class_info[cls_id]}")
        logger.info(f"{'='*100}")

        table = PrettyTable()
        table.field_names = ["Shot", "F1", "Prec", "Rec", "Acc", "AUC", "AUPRC", "Sn", "Sp", "TP", "TN", "FP", "FN"]
        table.align = "r"

        for k in shot_keys:
            if cls_id not in all_results_history[k]:
                continue

            m = all_results_history[k][cls_id]

            table.add_row([
                k,
                f"{m['F1']:.4f}", f"{m['Prec']:.4f}", f"{m['Rec']:.4f}", f"{m['Acc']:.4f}",
                f"{m['AUC']:.4f}", f"{m['AUPRC']:.4f}",
                f"{m['Sn']:.4f}", f"{m['Sp']:.4f}",
                m['TP'], m['TN'], m['FP'], m['FN']
            ])

        logger.info(f"\n{table}")


def main(config_path='json/plant_single.json', checkpoint_path=None, temperature=10.0, alpha=0.7,
         use_transductive=True, alpha_transductive=0.5, num_iterations=2):
    global Config, config_dict
    Config, config_dict = load_config(config_path)

    logger = setup_logging(Config.log_dir, Config.experiment_name + '_3way_protonet')
    tb_writer = setup_tensorboard(Config.log_dir, Config.experiment_name + '_3way_protonet')

    logger.info(f"\n{'='*60}")
    logger.info("Plant 3-Way Multi-class Classification using Prototypical Networks")
    logger.info("Strategy: Metric-Based Learning | Cosine Similarity | No Fine-Tuning")
    logger.info("         Model as FIXED Feature Extractor")
    if use_transductive:
        logger.info("         TRANSDUCTIVE MODE: TPR + CL2N Enabled")
    logger.info(f"{'='*60}")

    # Set random seeds
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    random.seed(Config.random_seed)

    # Load Plant Dataset
    logger.info(f"Loading Plant Dataset...")
    full_dataset = PlantSingleDataset(
        plant_dir=Config.data.plant_data_dir,
        zero_dir=getattr(Config.data, 'zero_data_dir', 'npy/zero'),
        cache_dir=Config.data.cache_dir,
        use_cache=True,
        preload_cache=True
    )

    # Target classes for 3-way classification
    target_classes = [5, 8, 9]
    group_indices = [3, 1, 0]  # Corresponding group indices

    # Prepare Plant train/test indices for few-shot sampling
    plant_train_indices = []
    plant_test_indices = []

    np.random.seed(Config.random_seed)
    for c in target_classes:
        c_pos_indices = full_dataset.get_plant_indices_by_class(c)
        np.random.shuffle(c_pos_indices)

        split_idx = int(0.1 * len(c_pos_indices))
        plant_train_indices.extend(c_pos_indices[:split_idx])
        plant_test_indices.extend(c_pos_indices[split_idx:])

    plant_train_indices = list(set(plant_train_indices))
    plant_test_indices = list(set(plant_test_indices))
    plant_test_indices = [i for i in plant_test_indices if i not in plant_train_indices]

    # Test set: Plant test only (no Zero samples)
    test_dataset = Subset(full_dataset, plant_test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Plant Train Pool: {len(plant_train_indices)}, Plant Test: {len(plant_test_indices)}")
    logger.info(f"Test Set: Plant Only (Classes {target_classes})")

    # Load checkpoint for base model
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error("Checkpoint path is required for 3-way classification training!")
        return

    # Results storage
    all_results = {}

    # shot_counts = [0, 1, 3, 5, 7, 10, 50, 100]
    shot_counts = [0, 2, 4, 6, 8, 10, 50, 100]

    # ========================================================================
    # CREATE FEATURE EXTRACTOR (FROZEN MODEL)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"{'='*80}")
    logger.info(f"INITIALIZING PROTOTYPICAL NETWORKS FOR CLASSES {target_classes}")
    logger.info(f"  Class 5 (Y) -> Group U (index 3)")
    logger.info(f"  Class 8 (m5C) -> Group C (index 1)")
    logger.info(f"  Class 9 (m6A) -> Group A (index 0)")
    logger.info(f"  Strategy: Metric-Based | Temperature Scaling: {temperature}")
    logger.info(f"  NO GRADIENT UPDATES - Model is FROZEN")
    if use_transductive:
        logger.info(f"  TRANSDUCTIVE PROTOTYPE REFINEMENT (TPR): ENABLED")
        logger.info(f"  FEATURE CENTERING (CL2N): ENABLED")
        logger.info(f"    - Alpha Transductive: {alpha_transductive}")
        logger.info(f"    - Num Iterations: {num_iterations}")
    logger.info(f"{'='*80}")
    logger.info(f"{'='*80}")

    # Create model
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=Config.cnn_hidden_dim,
        cnn_kernel_sizes=Config.cnn_kernel_sizes,
        cnn_dropout=Config.cnn_dropout,
        gcn_hidden_dim=Config.gcn_hidden_dim,
        gcn_out_channels=Config.gcn_out_channels,
        gcn_num_layers=Config.gcn_num_layers,
        gcn_dropout=Config.gcn_dropout,
        num_classes=12,
        num_attn_heads=Config.num_attn_heads,
        attn_dropout=Config.attn_dropout,
        use_simple_pooling=Config.use_simple_pooling,
        use_hierarchical=getattr(Config, 'use_hierarchical', True),
        use_layer_norm=Config.use_layer_norm
    ).to(Config.device)

    # Load pre-trained weights
    checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded pre-trained weights from {checkpoint_path}")

    # FREEZE the entire model - no training will occur
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    logger.info("Model FROZEN - All parameters set to requires_grad=False")
    logger.info("Using model as FIXED feature extractor for prototypical networks")

    # Create feature extractor wrapper
    feature_extractor = FeatureExtractor(model).to(Config.device)
    feature_extractor.eval()

    # ========================================================================
    # EXTRACT PRE-TRAINED PROTOTYPES (0-SHOT KNOWLEDGE)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"PROTOTYPE RECTIFICATION (TIP-STYLE)")
    logger.info(f"{'='*80}")
    logger.info(f"Extracting pre-trained class query prototypes as 0-shot priors...")
    logger.info(f"This preserves the knowledge learned during pre-training,")
    logger.info(f"which is crucial since standard prototypical networks discard this.")
    logger.info(f"{'='*80}\n")

    prior_prototypes = get_pretrained_prototypes(model, target_classes, Config.device)

    logger.info(f"Pre-trained prototypes extracted:")
    for cls_id in target_classes:
        if cls_id in prior_prototypes:
            prior_norm = torch.norm(prior_prototypes[cls_id]).item()
            logger.info(f"  Class {cls_id} ({PLANT_CLASS_MAPPING[cls_id]['class_name']}): "
                       f"Prior norm = {prior_norm:.4f}")

    logger.info(f"\n{'='*80}")
    logger.info(f"RECTIFICATION STRATEGY: alpha = {alpha}")
    logger.info(f"  Formula: p_rect = {alpha} * p_prior + {1-alpha} * p_empirical")
    logger.info(f"  - Higher alpha (0.7-0.9): Bias towards 0-shot knowledge")
    logger.info(f"  - Lower alpha (0.1-0.3): Bias towards few-shot adaptation")
    logger.info(f"  - alpha=0.7 is recommended: Trust pre-training, refine with few-shot")
    logger.info(f"{'='*80}\n")

    # ====================================================================
    # Loop over shot counts for 3-way classification
    # ====================================================================
    for k_shot in shot_counts:
        logger.info(f"\n--- 3-Way Prototypical Network: Running {k_shot}-Shot Learning ---")

        if k_shot == 0:
            # Zero-shot: Cannot compute prototypes without support samples
            logger.warning(f"Zero-shot: Prototypical networks require at least 1 support sample per class")
            logger.warning(f"Skipping {k_shot}-shot evaluation")
            continue

        # 3-WAY Sampling: k_shot samples from EACH class (total = 3*k_shot)
        np.random.seed(Config.random_seed + k_shot)
        support_indices = sample_support_set_3way(
            full_dataset, plant_train_indices, k_shot, target_classes
        )

        if len(support_indices) == 0:
            logger.warning(f"No support samples found at {k_shot}-shot!")
            continue

        support_data_list = [full_dataset[i] for i in support_indices]

        # Count samples per class for logging
        class_counts = {5: 0, 8: 0, 9: 0}
        for i in support_indices:
            if i < full_dataset.num_plant:
                for c in target_classes:
                    if full_dataset.plant_y12[i, c] == 1:
                        class_counts[c] += 1

        logger.info(f"3-Way Sampling: {len(support_indices)} total samples "
                    f"(Class 5: {class_counts[5]}, Class 8: {class_counts[8]}, Class 9: {class_counts[9]})")

        # ====================================================================
        # PROTOTYPE COMPUTATION (With Rectification)
        # ====================================================================
        logger.info(f"Computing rectified class prototypes from {len(support_indices)} support samples...")
        logger.info(f"  Blending {alpha} * prior + {1-alpha} * empirical")

        if use_transductive:
            # Compute prototypes AND support mean for CL2N
            prototypes, support_mean = compute_rectified_prototypes(
                feature_extractor, support_data_list, target_classes, Config.device,
                prior_prototypes=prior_prototypes, alpha=alpha, return_support_mean=True
            )
            logger.info(f"Support mean (CL2N) computed: norm = {torch.norm(support_mean).item():.4f}")
        else:
            # Standard prototypical networks (no centering)
            prototypes = compute_rectified_prototypes(
                feature_extractor, support_data_list, target_classes, Config.device,
                prior_prototypes=prior_prototypes, alpha=alpha, return_support_mean=False
            )
            support_mean = None

        logger.info(f"Rectified prototypes computed:")
        for cls_id in target_classes:
            if cls_id in prototypes:
                proto_norm = torch.norm(prototypes[cls_id]).item()
                logger.info(f"  Class {cls_id} ({PLANT_CLASS_MAPPING[cls_id]['class_name']}): "
                           f"Rectified prototype norm = {proto_norm:.4f}")

        # ====================================================================
        # METRIC-BASED EVALUATION
        # ====================================================================
        if use_transductive:
            logger.info(f"Evaluating with TRANSDUCTIVE inference (TPR + CL2N)...")
            logger.info(f"  - Adapting prototypes to test distribution")
            logger.info(f"  - Iterations: {num_iterations}, Alpha: {alpha_transductive}")
            metrics = evaluate_3way_transductive(
                feature_extractor, prototypes, test_loader, Config.device,
                target_classes, support_mean=support_mean, temperature=temperature,
                alpha_transductive=alpha_transductive, num_iterations=num_iterations
            )
        else:
            logger.info(f"Evaluating with standard metric-based prediction (temperature={temperature})...")
            metrics = evaluate_3way_detailed(
                feature_extractor, prototypes, test_loader, Config.device,
                target_classes, temperature
            )

        # Store results
        all_results[k_shot] = metrics

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"3-Way Prototypical Network - {k_shot}-Shot Results")
        logger.info(f"{'='*60}")
        for cls_id in [5, 8, 9]:
            cls_metrics = metrics[cls_id]
            cls_name = PLANT_CLASS_MAPPING[cls_id]['class_name']
            logger.info(f"Class {cls_id} ({cls_name}) | F1: {cls_metrics['F1']:.4f}, "
                       f"Prec: {cls_metrics['Prec']:.4f}, Rec: {cls_metrics['Rec']:.4f}, "
                       f"Acc: {cls_metrics['Acc']:.4f}, AUC: {cls_metrics['AUC']:.4f}, "
                       f"AUPRC: {cls_metrics['AUPRC']:.4f}")

        if tb_writer:
            for cls_id in [5, 8, 9]:
                cls_metrics = metrics[cls_id]
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_F1', cls_metrics['F1'], k_shot)
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_Prec', cls_metrics['Prec'], k_shot)
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_Rec', cls_metrics['Rec'], k_shot)
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_Acc', cls_metrics['Acc'], k_shot)
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_AUC', cls_metrics['AUC'], k_shot)
                tb_writer.add_scalar(f'3way_protonet/class_{cls_id}_AUPRC', cls_metrics['AUPRC'], k_shot)

    # Print final results tables (one per class)
    print_detailed_3way_results(all_results, logger)

    # =========================================================================
    # Save Results to JSON
    # =========================================================================
    results_path = os.path.join(Config.checkpoint_dir, 'plant_3way_protonet_results.json')

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

    # Reorganize results for better JSON structure
    serializable_results = {}
    for k_shot, metrics in all_results.items():
        if k_shot == 'full':
            shot_key = 'full'
        else:
            shot_key = f"{k_shot}_shot"
        serializable_results[shot_key] = convert_to_serializable(metrics)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"3-Way Prototypical Network results saved to: {results_path}")
    logger.info(f"{'='*80}")

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Plant 3-way multi-class using Rectified Prototypical Networks with Transductive Refinement"
    )
    parser.add_argument('--config', type=str, default='json/plant_single.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to pre-trained checkpoint',
                       default="logs/rna_classification_20260111_111223/checkpoints/epoch_010.pt")
    parser.add_argument('--temperature', type=float, default=10.0,
                       help='Temperature scaling for softmax (higher = sharper predictions)')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Prototype rectification coefficient [0, 1]. '
                            'p_rect = alpha * p_prior + (1-alpha) * p_empirical. '
                            'Default=0.7 (bias towards pre-trained knowledge). '
                            'alpha=1.0: pure 0-shot, alpha=0.0: standard prototypical network')
    parser.add_argument('--use_transductive', action='store_true', default=True,
                       help='Enable Transductive Prototype Refinement (TPR) and CL2N Feature Centering. '
                            'This adapts prototypes to the test set distribution and should improve AUC.')
    parser.add_argument('--no_transductive', action='store_false', dest='use_transductive',
                       help='Disable transductive inference (use standard prototypical networks)')
    parser.add_argument('--alpha_transductive', type=float, default=0.5,
                       help='Transductive interpolation coefficient [0, 1]. '
                            'P_new = alpha * P_init + (1-alpha) * P_query. '
                            'Default=0.5 (balance initial prototypes and query statistics). '
                            'Higher: trust initial prototypes, Lower: trust query statistics')
    parser.add_argument('--num_iterations', type=int, default=2,
                       help='Number of transductive refinement iterations. Default=2. '
                            'Each iteration refines prototypes using test set probabilities.')
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        temperature=args.temperature,
        alpha=args.alpha,
        use_transductive=args.use_transductive,
        alpha_transductive=args.alpha_transductive,
        num_iterations=args.num_iterations
    )
