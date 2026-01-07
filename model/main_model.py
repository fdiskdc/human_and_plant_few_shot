"""
RNA_ClassQuery_Model - Multi-scale Class-Query Classification Model for RNA

This module implements the main model for RNA 12-class multi-label classification.
The model combines:
1. Parallel CNN for multi-scale local feature extraction
2. GCN for graph-structured feature propagation
3. Class-Query attention for per-class prediction
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple

from .blocks import (
    ParallelCNNBlock, GCNBlock, ClassQueryHead, ClassQueryHeadPooling,
    HierarchicalClassQueryHeadPooling
)


class RNA_ClassQuery_Model(nn.Module):
    """
    RNA Classification Model using Multi-scale CNN + GCN + Class-Query Attention

    Architecture:
        Input (Batch, 1001, 4) -> ParallelCNN -> GCN -> ClassQueryHead -> Output (Batch, 12)

    Forward pass flow:
        1. Input: one-hot RNA sequence (Batch, 1001, 4) or PyG Batch object
        2. ParallelCNN: Extract multi-scale local features
           - Output shape: (Total_Nodes, cnn_out_channels)
        3. GCN: Propagate features on secondary structure graph
           - Output shape: (Total_Nodes, gcn_hidden_dim)
        4. ClassQueryHead: Cross-attention between class queries and node features
           - Output shape: (Batch, 12) or (Batch, 12), (Batch, 4) for hierarchical
    """

    def __init__(
        self,
        # CNN parameters
        cnn_hidden_dim: int = 64,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        cnn_dropout: float = 0.1,
        # GCN parameters
        gcn_hidden_dim: int = 128,
        gcn_out_channels: int = 128,
        gcn_num_layers: int = 3,
        gcn_dropout: float = 0.3,
        # Class-Query parameters
        num_classes: int = 12,
        num_attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_simple_pooling: bool = False,
        use_hierarchical: bool = False,
        # Other
        use_layer_norm: bool = True
    ):
        """
        Args:
            cnn_hidden_dim: Hidden dimension for each CNN branch
            cnn_kernel_sizes: Kernel sizes for parallel CNN branches
            cnn_dropout: Dropout for CNN block
            gcn_hidden_dim: Hidden dimension for GCN layers
            gcn_out_channels: Output dimension for GCN block
            gcn_num_layers: Number of GCN layers (2-3 recommended)
            gcn_dropout: Dropout for GCN block
            num_classes: Number of output classes (12)
            num_attn_heads: Number of attention heads in Class-Query
            attn_dropout: Dropout for Class-Query attention
            use_simple_pooling: If True, use simple pooling attention; otherwise use Transformer decoder
            use_hierarchical: If True, use hierarchical head (returns 12-class and 4-class logits)
            use_layer_norm: If True, use LayerNorm in CNN; otherwise use BatchNorm1d
        """
        super().__init__()

        # Store configuration
        self.cnn_out_channels = len(cnn_kernel_sizes) * cnn_hidden_dim
        self.gcn_out_channels = gcn_out_channels
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical

        # 1. Parallel CNN Block
        # Input: (Batch, 4, 1001) or (Total_Nodes, 4)
        # Output: (Total_Nodes, cnn_out_channels)
        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout
        )

        # 2. GCN Block
        # Input: (Total_Nodes, cnn_out_channels) + edge_index
        # Output: (Total_Nodes, gcn_out_channels)
        self.gcn_block = GCNBlock(
            in_channels=self.cnn_out_channels,
            hidden_dim=gcn_hidden_dim,
            out_channels=gcn_out_channels,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
            use_residual=True
        )

        # 3. Class-Query Classification Head
        # Input: (Total_Nodes, gcn_out_channels) + batch
        # Output: (Batch, num_classes) or (Batch, 12), (Batch, 4) for hierarchical
        if use_hierarchical:
            self.class_query_head = HierarchicalClassQueryHeadPooling(
                hidden_dim=gcn_out_channels,
                dropout=attn_dropout
            )
        elif use_simple_pooling:
            self.class_query_head = ClassQueryHeadPooling(
                hidden_dim=gcn_out_channels,
                num_classes=num_classes,
                dropout=attn_dropout
            )
        else:
            self.class_query_head = ClassQueryHead(
                hidden_dim=gcn_out_channels,
                num_classes=num_classes,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
                use_decoder=True
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass

        Args:
            x: Node features
               - Shape: (Batch, 1001, 4) for batched tensor input
               - Shape: (Total_Nodes, 4) for PyG Batch object input
            edge_index: Edge indices for graph structure, shape (2, Num_Edges)
            batch: Batch assignment vector for PyG, shape (Total_Nodes,)
                    Required when x is in PyG format (Total_Nodes, 4)

        Returns:
            If use_hierarchical=True: (logits_12class, logits_4class) tuple
            Otherwise: logits_12class only
        """
        # Handle PyG Batch object
        if isinstance(x, Data) or isinstance(x, Batch):
            # x is a PyG Data/Batch object
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        # Verify input dimensions
        if x.dim() == 3:
            # Shape: (Batch, 1001, 4)
            batch_size = x.size(0)
            seq_len = x.size(1)
            assert seq_len == 1001, f"Expected sequence length 1001, got {seq_len}"

            # Create batch vector for PyG compatibility
            if batch is None:
                batch = torch.arange(
                    batch_size, device=x.device
                ).repeat_interleave(seq_len)

        elif x.dim() == 2:
            # Shape: (Total_Nodes, 4) - PyG format
            assert batch is not None, "batch vector must be provided for PyG format input"
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Step 1: Parallel CNN feature extraction
        # Input: x shape depends on format
        # Output: (Total_Nodes, cnn_out_channels)
        node_features = self.cnn_block(x, batch)

        # Step 2: GCN for graph-structured feature propagation
        # Input: (Total_Nodes, cnn_out_channels), edge_index
        # Output: (Total_Nodes, gcn_out_channels)
        node_features = self.gcn_block(node_features, edge_index)

        # Step 3: Class-Query attention for classification
        # Input: (Total_Nodes, gcn_out_channels), batch
        # Output: (Batch, num_classes) or (Batch, 12), (Batch, 4)
        if self.use_hierarchical:
            logits_12class, logits_4class = self.class_query_head(node_features, batch)
            return logits_12class, logits_4class
        else:
            logits = self.class_query_head(node_features, batch)
            return logits

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with optional thresholding for multi-label classification

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment vector
            threshold: Threshold for binary prediction (default 0.5)

        Returns:
            If use_hierarchical=True:
                predictions_12class, probs_12class, predictions_4class, probs_4class
            Otherwise:
                predictions, probabilities
        """
        self.eval()
        with torch.no_grad():
            if self.use_hierarchical:
                logits_12, logits_4 = self.forward(x, edge_index, batch)
                probs_12 = torch.sigmoid(logits_12)
                preds_12 = (probs_12 >= threshold).long()
                probs_4 = torch.sigmoid(logits_4)
                preds_4 = (probs_4 >= threshold).long()
                return preds_12, probs_12, preds_4, probs_4
            else:
                logits = self.forward(x, edge_index, batch)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= threshold).long()
                return predictions, probabilities

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability (requires modified forward)

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment vector

        Returns:
            attention_weights: Attention weights per class per node
                               shape (Batch, num_classes, 1001)
        """
        # This would require hooking into the attention mechanism
        # For now, return placeholder
        raise NotImplementedError("Attention weight extraction not yet implemented")


class RNA_ClassQuery_Model_Large(nn.Module):
    """
    Larger version of RNA_ClassQuery_Model with more capacity

    Uses wider and deeper networks for potentially better performance.
    """

    def __init__(
        self,
        cnn_hidden_dim: int = 128,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7, 9),
        cnn_dropout: float = 0.1,
        gcn_hidden_dim: int = 256,
        gcn_out_channels: int = 256,
        gcn_num_layers: int = 4,
        gcn_dropout: float = 0.3,
        num_classes: int = 12,
        num_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        use_simple_pooling: bool = False,
        use_hierarchical: bool = False,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.cnn_out_channels = len(cnn_kernel_sizes) * cnn_hidden_dim
        self.gcn_out_channels = gcn_out_channels
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical

        # 1. Parallel CNN Block (larger)
        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout
        )

        # 2. GCN Block (deeper and wider)
        self.gcn_block = GCNBlock(
            in_channels=self.cnn_out_channels,
            hidden_dim=gcn_hidden_dim,
            out_channels=gcn_out_channels,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
            use_residual=True
        )

        # 3. Class-Query Classification Head
        if use_hierarchical:
            self.class_query_head = HierarchicalClassQueryHeadPooling(
                hidden_dim=gcn_out_channels,
                dropout=attn_dropout
            )
        elif use_simple_pooling:
            self.class_query_head = ClassQueryHeadPooling(
                hidden_dim=gcn_out_channels,
                num_classes=num_classes,
                dropout=attn_dropout
            )
        else:
            self.class_query_head = ClassQueryHead(
                hidden_dim=gcn_out_channels,
                num_classes=num_classes,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
                use_decoder=True
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> tuple:
        """Forward pass - same as base model"""
        # Handle PyG Batch object
        if isinstance(x, Data) or isinstance(x, Batch):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            assert seq_len == 1001, f"Expected sequence length 1001, got {seq_len}"
            if batch is None:
                batch = torch.arange(
                    batch_size, device=x.device
                ).repeat_interleave(seq_len)

        # Forward through blocks
        node_features = self.cnn_block(x, batch)
        node_features = self.gcn_block(node_features, edge_index)

        if self.use_hierarchical:
            logits_12, logits_4 = self.class_query_head(node_features, batch)
            return logits_12, logits_4
        else:
            logits = self.class_query_head(node_features, batch)
            return logits

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with thresholding"""
        self.eval()
        with torch.no_grad():
            if self.use_hierarchical:
                logits_12, logits_4 = self.forward(x, edge_index, batch)
                probs_12 = torch.sigmoid(logits_12)
                preds_12 = (probs_12 >= threshold).long()
                probs_4 = torch.sigmoid(logits_4)
                preds_4 = (probs_4 >= threshold).long()
                return preds_12, probs_12, preds_4, probs_4
            else:
                logits = self.forward(x, edge_index, batch)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= threshold).long()
                return predictions, probabilities
