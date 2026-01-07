"""
Sub-modules for RNA_ClassQuery_Model

This module contains:
- ParallelCNNBlock: Multi-scale CNN feature extraction
- GCNBlock: Graph Convolutional Network block
- ClassQueryHead: Class-Query classification head using Cross-Attention
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
from typing import Tuple, Optional


class ParallelCNNBlock(nn.Module):
    """
    Multi-scale CNN feature extraction block

    Architecture:
    - Input: (Batch, 4, 1001) - one-hot encoded RNA sequence
    - 4 parallel 1D convolution branches with kernel sizes [1, 3, 5, 7]
    - Each branch maintains sequence length via padding='same'
    - Concatenate outputs along channel dimension
    - Layer normalization

    Output: (Batch, 4 * hidden_dim, 1001) then transposed to (Batch * 1001, 4 * hidden_dim)
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Input channels (4 for one-hot A, C, G, U)
            hidden_dim: Hidden dimension for each convolution branch
            kernel_sizes: Kernel sizes for parallel branches
            use_layer_norm: If True, use LayerNorm; otherwise use BatchNorm1d
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.out_channels = len(kernel_sizes) * hidden_dim

        # Create parallel convolution branches
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=k,
                padding='same',  # Maintain sequence length
                bias=True
            )
            for k in kernel_sizes
        ])

        # Normalization layer (applied after concatenation)
        if use_layer_norm:
            # LayerNorm normalizes over (C, L) dimensions
            self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, 1001))
        else:
            # BatchNorm1d normalizes over C dimension
            self.norm = nn.BatchNorm1d(self.out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor, shape (Batch, 4, 1001) or (Batch * 1001, 4)
               If shape is (Batch, 1001, 4), will be transposed automatically
            batch: Batch assignment vector for PyG batch objects, shape (Batch * 1001,)

        Returns:
            Node features for GCN, shape (Total_Nodes, out_channels)
        """
        # Handle different input shapes
        if x.dim() == 3 and x.size(1) == 1001 and x.size(2) == 4:
            # Shape: (Batch, 1001, 4) -> (Batch, 4, 1001)
            x = x.transpose(1, 2)
        elif x.dim() == 2 and x.size(1) == 4:
            # Shape: (Total_Nodes, 4) -> (1, 4, Total_Nodes)
            # Need to reshape for conv1d
            if batch is not None:
                # Reshape from (Total_Nodes, 4) to (Batch, 4, 1001)
                batch_size = batch.max().item() + 1
                x = x.view(batch_size, 1001, 4).transpose(1, 2)
            else:
                # Single sample case
                x = x.t().unsqueeze(0)  # (1, 4, seq_len)

        # Input shape: (Batch, 4, 1001)
        batch_size = x.size(0)

        # Apply parallel convolutions
        branch_outputs = []
        for conv in self.conv_branches:
            # Each branch output: (Batch, hidden_dim, 1001)
            out = conv(x)
            branch_outputs.append(out)

        # Concatenate along channel dimension
        # Shape: (Batch, 4 * hidden_dim, 1001)
        concatenated = torch.cat(branch_outputs, dim=1)

        # Apply normalization
        if isinstance(self.norm, nn.LayerNorm):
            # LayerNorm expects (Batch, C, L)
            normalized = self.norm(concatenated)
        else:
            # BatchNorm1d expects (Batch, C, L)
            normalized = self.norm(concatenated)

        # Apply activation and dropout
        features = self.dropout(self.activation(normalized))
        # Shape: (Batch, 4 * hidden_dim, 1001)

        # Reshape for PyG: (Batch, 4 * hidden_dim, 1001) -> (Total_Nodes, 4 * hidden_dim)
        features = features.transpose(1, 2)  # (Batch, 1001, 4 * hidden_dim)
        features = features.reshape(-1, self.out_channels)  # (Batch * 1001, 4 * hidden_dim)

        return features


class GCNBlock(nn.Module):
    """
    Graph Convolutional Network block

    Architecture:
    - Input: Node features from CNN + edge_index
    - 2-3 GCN layers with residual connections
    - Dropout between layers

    Output: Node features, shape (Total_Nodes, hidden_dim)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        """
        Args:
            in_channels: Input feature dimension (4 * hidden_dim from CNN)
            hidden_dim: Hidden dimension for GCN layers
            out_channels: Output feature dimension
            num_layers: Number of GCN layers (2-3 recommended)
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_residual = use_residual

        # Input projection layer (if in_channels != hidden_dim)
        self.input_proj = None
        if in_channels != hidden_dim:
            self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # FIX: When input_proj is used, all layers take hidden_dim as input
            # Otherwise, first layer takes in_channels, subsequent layers take hidden_dim
            if self.input_proj is not None:
                # input_proj will project in_channels to hidden_dim
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim if i > 0 or (in_channels == hidden_dim) else in_channels
            self.gcn_layers.append(
                GCNConv(in_dim, out_channels if i == num_layers - 1 else hidden_dim)
            )
            # LayerNorm for each GCN layer
            self.norms.append(nn.LayerNorm(out_channels if i == num_layers - 1 else hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features, shape (Total_Nodes, in_channels)
            edge_index: Edge indices, shape (2, Num_Edges)

        Returns:
            Node features, shape (Total_Nodes, out_channels)
        """
        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        # Shape: (Total_Nodes, hidden_dim)

        # Store input for residual connection
        residual = x

        # Apply GCN layers
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            # GCN forward
            x = gcn(x, edge_index)
            # Shape: (Total_Nodes, hidden_dim or out_channels)

            # Apply normalization, activation, and dropout (except for final layer)
            if i < self.num_layers - 1:
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)

                # Residual connection
                if self.use_residual and x.shape == residual.shape:
                    x = x + residual
                    residual = x
            else:
                # Final layer - only normalization
                x = norm(x)

        return x  # Shape: (Total_Nodes, out_channels)


class ClassQueryHead(nn.Module):
    """
    Class-Query classification head using Cross-Attention

    Architecture:
    - Query: 12 learnable class embeddings, shape (12, hidden_dim)
    - Key/Value: GCN output node features, shape (Total_Nodes, hidden_dim)
    - Cross-attention between queries and node features
    - Output: (Batch, 12) logits

    This allows each class query to attend to relevant positions in the sequence.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_decoder: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension for queries and node features
            num_classes: Number of classes (12 for this task)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_decoder: If True, use TransformerDecoderLayer; otherwise use MultiheadAttention
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        if use_decoder:
            # Using TransformerDecoderLayer for cross-attention
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=1)

            # Output projection
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Using MultiheadAttention directly
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            # Output projection
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: Node features from GCN, shape (Total_Nodes, hidden_dim)
            batch: Batch assignment vector, shape (Total_Nodes,)

        Returns:
            Class logits, shape (Batch, num_classes)
        """
        batch_size = batch.max().item() + 1
        num_nodes = batch.size(0)
        device = node_features.device

        # Expand class queries for each sample in batch
        # Shape: (Batch, num_classes, hidden_dim)
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # Prepare memory (node features) for cross-attention
        # Need to organize nodes by batch and pad to same length
        # PyG provides a convenient way with global pooling, but we need sequence structure

        # Group nodes by batch
        max_nodes = 1001  # Fixed sequence length

        # Create padded tensor: (Batch, max_nodes, hidden_dim)
        memory = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Get nodes for this batch
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask]  # (num_nodes_in_batch, hidden_dim)

            num_batch_nodes = batch_nodes.size(0)
            memory[b, :num_batch_nodes, :] = batch_nodes
            mask[b, num_batch_nodes:] = True  # True means masked/padding

        # Transpose mask for TransformerDecoder (True = ignore)
        # TransformerDecoder expects mask in different format, so we use key_padding_mask
        # key_padding_mask: (Batch, Seq) where True indicates padding
        memory_mask = mask  # (Batch, max_nodes)

        # Cross-attention: queries attend to memory (node features)
        # queries: (Batch, num_classes, hidden_dim)
        # memory: (Batch, max_nodes, hidden_dim)
        attended_features = self.cross_attention(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_mask
        )
        # Shape: (Batch, num_classes, hidden_dim)

        # Apply output projection
        logits = self.output_proj(attended_features)
        # Shape: (Batch, num_classes, 1)

        # Squeeze last dimension
        logits = logits.squeeze(-1)
        # Shape: (Batch, num_classes)

        # Ensure logits is on the same device as input
        return logits.to(node_features.device)


# Alternative pooling-based class query head (simpler version)
class ClassQueryHeadPooling(nn.Module):
    """
    Simplified Class-Query head using attention pooling

    This version computes attention weights between class queries and node features,
    then aggregates node features via weighted pooling.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension for queries and node features
            num_classes: Number of classes (12 for this task)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # Attention scoring function
        self.attention_scale = hidden_dim ** 0.5

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: Node features from GCN, shape (Total_Nodes, hidden_dim)
            batch: Batch assignment vector, shape (Total_Nodes,)

        Returns:
            Class logits, shape (Batch, num_classes)
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        # Class queries: (num_classes, hidden_dim)
        queries = self.class_queries.to(device)  # (num_classes, hidden_dim)

        # Compute attention scores for each batch
        logits_list = []

        for b in range(batch_size):
            # Get nodes for this batch
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask]  # (num_nodes, hidden_dim)

            num_nodes = batch_nodes.size(0)

            # Compute attention scores: (num_classes, num_nodes)
            # score[i, j] = query[i] @ node[j] / scale
            scores = torch.matmul(queries, batch_nodes.t()) / self.attention_scale
            # Shape: (num_classes, num_nodes)

            # Apply softmax over nodes for each class
            attn_weights = torch.softmax(scores, dim=1)  # (num_classes, num_nodes)

            # Aggregate node features via attention weights
            # aggregated[i] = sum_j attn_weights[i, j] * batch_nodes[j]
            aggregated = torch.matmul(attn_weights, batch_nodes)
            # Shape: (num_classes, hidden_dim)

            # Apply output projection
            class_logits = self.output_proj(aggregated)
            # Shape: (num_classes, 1)

            logits_list.append(class_logits.squeeze(-1))  # (num_classes,)

        # Stack logits from all batches
        logits = torch.stack(logits_list, dim=0)
        # Shape: (Batch, num_classes)

        # Ensure logits is on the same device as input
        return logits.to(node_features.device)


# ============================================================================
# Hierarchical Class-Query Head (Group-to-Class Derivation)
# ============================================================================

class HierarchicalClassQueryHeadPooling(nn.Module):
    """
    Hierarchical Class-Query head with Group-to-Class derivation.

    Architecture:
    - Level 1: 4 learnable group queries (A, C, G, U)
    - Level 2: Each group query is passed through an MLP to derive class queries
    - Output: Both 4-class and 12-class logits

    Group sizes: A=5, C=3, G=2, U=2 (total 12 classes)

    The derivation order ensures the final 12 queries follow LABEL_MAPPING:
    [Am, Atol, Cm, Gm, Tm, Y, ac4C, m1A, m5C, m6A, m6Am, m7G]
    """

    # Group to class indices mapping (ensures correct order)
    GROUP_TO_CLASS_INDICES = {
        'A': [0, 1, 7, 9, 10],    # Am, Atol, m1A, m6A, m6Am
        'C': [2, 6, 8],           # Cm, ac4C, m5C
        'G': [3, 11],             # Gm, m7G
        'U': [4, 5]               # Tm, Y
    }

    GROUP_SIZES = {'A': 5, 'C': 3, 'G': 2, 'U': 2}
    GROUP_ORDER = ['A', 'C', 'G', 'U']

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension for queries and node features
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_groups = 4
        self.num_classes = 12

        # Level 1: Learnable group queries (4 groups: A, C, G, U)
        self.group_queries = nn.Parameter(torch.randn(self.num_groups, hidden_dim))

        # Level 2: Group-specific MLPs for deriving class queries
        # Each MLP takes a group query and outputs class-specific queries
        self.group_mlps = nn.ModuleDict({
            'A': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['A'], dropout),
            'C': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['C'], dropout),
            'G': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['G'], dropout),
            'U': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['U'], dropout)
        })

        # Attention scoring function
        self.attention_scale = hidden_dim ** 0.5

        # Output projections (separate for 4-class and 12-class)
        self.output_proj_4class = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.output_proj_12class = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _make_derivation_mlp(self, hidden_dim: int, num_outputs: int, dropout: float) -> nn.Module:
        """
        Create an MLP that derives class queries from a group query.

        Args:
            hidden_dim: Input and output hidden dimension
            num_outputs: Number of class queries to derive
            dropout: Dropout probability

        Returns:
            MLP module
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * num_outputs)
        )

    def _derive_class_queries(self) -> torch.Tensor:
        """
        Derive 12 class queries from 4 group queries using group-specific MLPs.

        The derived queries are ordered according to LABEL_MAPPING indices 0-11:
        [Am, Atol, Cm, Gm, Tm, Y, ac4C, m1A, m5C, m6A, m6Am, m7G]

        Returns:
            Derived class queries, shape (12, hidden_dim)
        """
        device = self.group_queries.device

        # Derive class queries for each group
        derived_queries = {}

        for group_idx, group_name in enumerate(self.GROUP_ORDER):
            group_query = self.group_queries[group_idx]  # (hidden_dim,)

            # Pass through group-specific MLP
            mlp = self.group_mlps[group_name]
            output = mlp(group_query)  # (num_classes_in_group * hidden_dim,)

            # Reshape to separate class queries
            num_classes = self.GROUP_SIZES[group_name]
            class_queries = output.view(num_classes, self.hidden_dim)  # (num_classes_in_group, hidden_dim)

            derived_queries[group_name] = class_queries

        # Concatenate in the correct order to match LABEL_MAPPING
        # Order: Am(0), Atol(1), Cm(2), Gm(3), Tm(4), Y(5), ac4C(6), m1A(7), m5C(8), m6A(9), m6Am(10), m7G(11)
        all_queries = []
        for group_name in self.GROUP_ORDER:
            all_queries.append(derived_queries[group_name])

        # Concatenate: A's 5 + C's 3 + G's 2 + U's 2 = 12
        concatenated = torch.cat(all_queries, dim=0)  # (12, hidden_dim)

        # Reorder to match LABEL_MAPPING order
        # Current order after concat: A[0,1,7,9,10], C[2,6,8], G[3,11], U[4,5]
        # We need: [0,1,2,3,4,5,6,7,8,9,10,11]
        reorder_indices = []
        for group_name in self.GROUP_ORDER:
            reorder_indices.extend(self.GROUP_TO_CLASS_INDICES[group_name])

        # Create properly ordered tensor
        ordered_queries = torch.zeros_like(concatenated)
        for i, idx in enumerate(reorder_indices):
            ordered_queries[idx] = concatenated[i]

        return ordered_queries  # (12, hidden_dim) in correct order [0,1,2,3,4,5,6,7,8,9,10,11]

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> tuple:
        """
        Optimized Forward pass using vectorized PyG operations
        """
        # Import PyG utils (or move to top of file)
        from torch_geometric.utils import softmax
        from torch_geometric.nn import global_add_pool
        
        # 1. Prepare Queries
        group_queries = self.group_queries  # (4, D)
        class_queries = self._derive_class_queries()  # (12, D)
        
        # 2. Parallel Computation for 4-Class Task
        # Score: (N_nodes, D) @ (D, 4) -> (N_nodes, 4)
        scores_4 = torch.matmul(node_features, group_queries.t()) / self.attention_scale
        
        # Softmax: (N_nodes, 4) - normalizes per graph based on batch index
        attn_weights_4 = softmax(scores_4, batch, dim=0)
        
        # Weighted Features: (N_nodes, 4, D)
        # Expansion: (N, 4, 1) * (N, 1, D)
        weighted_4 = attn_weights_4.unsqueeze(-1) * node_features.unsqueeze(1)
        
        # Pooling: Aggregate weighted features per graph
        # Flatten (N, 4, D) -> (N, 4*D) for global_add_pool -> (Batch, 4*D)
        # Then reshape back to (Batch, 4, D)
        agg_4 = global_add_pool(weighted_4.flatten(1), batch).view(-1, 4, self.hidden_dim)
        
        # Logits: (Batch, 4)
        logits_4class = self.output_proj_4class(agg_4).squeeze(-1)

        # 3. Parallel Computation for 12-Class Task (Same Logic)
        # Score: (N_nodes, 12)
        scores_12 = torch.matmul(node_features, class_queries.t()) / self.attention_scale
        
        # Softmax: (N_nodes, 12)
        attn_weights_12 = softmax(scores_12, batch, dim=0)
        
        # Weighted: (N_nodes, 12, D)
        weighted_12 = attn_weights_12.unsqueeze(-1) * node_features.unsqueeze(1)
        
        # Pooling: (Batch, 12*D) -> (Batch, 12, D)
        agg_12 = global_add_pool(weighted_12.flatten(1), batch).view(-1, 12, self.hidden_dim)
        
        # Logits: (Batch, 12)
        logits_12class = self.output_proj_12class(agg_12).squeeze(-1)

        return logits_12class, logits_4class
