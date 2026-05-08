"""
EvoRMD-style model adapted for the human multi-label RNA classification task.

Architecture:
    Input: one-hot (Total_Nodes, 4) from PyG batch
      ↓
    Reshape: (Batch, 1001, 4)
      ↓
    Conv1dEmbedder: Conv1d(4→d_fm, kernel=7) + ReLU + Conv1d(d_fm→d_fm) → (B, 1001, d_fm)
      ↓
    TrainableAttention(d_fm): Linear(d_fm, 1) → softmax → (B, 1001)
      ↓
    MIL Pooling: weighted sum → (B, d_fm)
      ↓
    MulticlassClassifier(d_fm, num_classes, depth): MLP → (B, num_classes)
      ↓
    Output: raw logits (B, num_classes) — compatible with BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dEmbedder(nn.Module):
    """
    Two-layer Conv1d to project 4-dim one-hot RNA sequences into d_fm-dim embeddings.

    Replaces RNA-FM backbone from the original EvoRMD pipeline.
    """

    def __init__(self, in_channels=4, d_fm=640, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2  # keep sequence length unchanged
        self.conv1 = nn.Conv1d(in_channels, d_fm, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_fm, d_fm, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_fm)

    def forward(self, x):
        """
        Args:
            x: (B, L, 4) one-hot encoded RNA sequence
        Returns:
            (B, L, d_fm) token-level embeddings
        """
        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2)  # (B, 4, L)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, L, d_fm)
        x = self.ln(x)
        return x


class TrainableAttention(nn.Module):
    """
    Learnable attention module over token embeddings (from EvoRMD).

    Input:
      token_embeddings: (batch_size, seq_length, embedding_dim)
    Output:
      attention_weights: (batch_size, seq_length)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, token_embeddings):
        # token_embeddings: (B, L, D)
        attention_scores = self.attention(token_embeddings).squeeze(-1)  # (B, L)
        attention_weights = F.softmax(attention_scores, dim=-1)          # (B, L)
        return attention_weights


class MulticlassClassifier(nn.Module):
    """
    Multi-class classifier on top of the fused embedding (from EvoRMD).

    mlp_depth:
      - 1: single linear classification head
      - 2: two-layer MLP (Linear → ReLU → Linear)
      - 3: three-layer MLP (Linear → ReLU → Linear → ReLU → Linear)
    """

    def __init__(self, embedding_dim, num_classes, mlp_depth=2):
        super().__init__()
        self.mlp_depth = mlp_depth

        if mlp_depth == 1:
            self.fc = nn.Linear(embedding_dim, num_classes)
        elif mlp_depth == 2:
            self.fc1 = nn.Linear(embedding_dim, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
        elif mlp_depth == 3:
            self.fc1 = nn.Linear(embedding_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
        else:
            raise ValueError(f"Unsupported MLP depth: {mlp_depth}")

    def forward(self, x):
        if self.mlp_depth == 1:
            return self.fc(x)
        elif self.mlp_depth == 2:
            x = self.relu(self.fc1(x))
            return self.fc2(x)
        elif self.mlp_depth == 3:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)


class EvoRMDForHuman(nn.Module):
    """
    EvoRMD-style model for human multi-label RNA classification.

    Replaces RNA-FM with Conv1dEmbedder, keeps TrainableAttention + MulticlassClassifier.
    Same forward() interface as model_v3 for compatibility with existing train/test utilities.
    """

    def __init__(
        self,
        num_task=12,
        d_fm=640,
        mlp_depth=2,
        conv_kernel_size=7,
        conv_dropout=0.1,
        use_hierarchical=False,
    ):
        super().__init__()
        self.num_task = num_task
        self.d_fm = d_fm
        self.use_hierarchical = use_hierarchical

        # Conv1d embedder (replaces RNA-FM)
        self.embedder = Conv1dEmbedder(
            in_channels=4,
            d_fm=d_fm,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )

        # Trainable attention for MIL pooling
        self.attention = TrainableAttention(d_fm)

        # Classifier
        self.classifier = MulticlassClassifier(d_fm, num_task, mlp_depth=mlp_depth)

        # Group indices for hierarchical 4-class derivation (same as model_v3)
        self.group_indices = [[0, 1, 7, 9, 10], [2, 6, 8], [3, 11], [4, 5]]

    def forward(self, x, edge_index=None, batch=None, return_attention=False):
        """
        Forward pass compatible with train.py / test.py utilities.

        Args:
            x: Input tensor
               - Shape: (Total_Nodes, 4) for PyG format
               - Shape: (Batch, 1001, 4) for tensor format
            edge_index: Edge indices (unused, kept for interface compatibility)
            batch: Batch assignment vector (used for reshaping PyG → batch)
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False:
                logits (B, num_task)  OR  (logits_12, logits_4) if hierarchical
            If return_attention=True:
                (logits, attn_weights)  OR  (logits_12, logits_4, attn_weights) if hierarchical
        """
        # ---- Handle different input formats ----
        if x.dim() == 2:
            # PyG format: (Total_Nodes, 4)
            if batch is None:
                x = x.unsqueeze(0)  # (1, Total_Nodes, 4)
                batch_size = 1
            else:
                batch_size = batch.max().item() + 1
                x = x.view(batch_size, 1001, 4)
        elif x.dim() == 3:
            if x.size(1) == 4 and x.size(2) != 4:
                # (B, 4, 1001) → transpose to (B, 1001, 4)
                x = x.transpose(1, 2)
            batch_size = x.size(0)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # ---- Conv1d embedding ----
        token_embeddings = self.embedder(x)  # (B, L, d_fm)

        # ---- Attention pooling ----
        attn_weights = self.attention(token_embeddings)  # (B, L)
        pooled = torch.bmm(
            attn_weights.unsqueeze(1), token_embeddings
        ).squeeze(1)  # (B, d_fm)

        # ---- Classification ----
        logits = self.classifier(pooled)  # (B, num_task)

        # ---- Return ----
        if return_attention:
            if self.use_hierarchical:
                logits_4 = self._derive_4class(logits, batch_size, x.device)
                return logits, logits_4, attn_weights
            return logits, attn_weights
        else:
            if self.use_hierarchical:
                logits_4 = self._derive_4class(logits, batch_size, x.device)
                return logits, logits_4
            return logits

    def _derive_4class(self, logits_12, batch_size, device):
        """Derive 4-class logits from 12-class logits by max-pooling over nucleotide groups."""
        logits_4 = torch.zeros(batch_size, 4, device=device)
        for i, indices in enumerate(self.group_indices):
            logits_4[:, i] = logits_12[:, indices].max(dim=1)[0]
        return logits_4
