
"""
RNA_ClassQuery_Model - Multi-scale Class-Query Classification Model for RNA

This module implements the main model for RNA 12-class multi-label classification.
The model combines:
1. Parallel CNN for multi-scale local feature extraction
2. GCN for graph-structured feature propagation
3. Class-Query attention for per-class prediction

Sub-modules:
- ParallelCNNBlock: Multi-scale CNN feature extraction
- GCNBlock: Graph Convolutional Network block
- ClassQueryHead: Class-Query classification head using Cross-Attention
- HierarchicalClassQueryHeadPooling: Hierarchical head with Group-to-Class derivation
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple

# Import GROUP_TO_CLASS_INDICES for hierarchical head
from utils.common import GROUP_TO_CLASS_INDICES


# ============================================================================
# Sub-modules for RNA_ClassQuery_Model
# ============================================================================

class ParallelCNNBlock(nn.Module):
    """
    Multi-scale CNN feature extraction block
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.out_channels = len(kernel_sizes) * hidden_dim

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=k,
                padding='same',
                bias=True
            )
            for k in kernel_sizes
        ])

        if use_layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, 1001))
        else:
            self.norm = nn.BatchNorm1d(self.out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 1001 and x.size(2) == 4:
            x = x.transpose(1, 2)
        elif x.dim() == 2 and x.size(1) == 4:
            if batch is not None:
                batch_size = batch.max().item() + 1
                x = x.view(batch_size, 1001, 4).transpose(1, 2)
            else:
                x = x.t().unsqueeze(0)

        branch_outputs = []
        for conv in self.conv_branches:
            out = conv(x)
            branch_outputs.append(out)

        concatenated = torch.cat(branch_outputs, dim=1)

        if isinstance(self.norm, nn.LayerNorm):
            normalized = self.norm(concatenated)
        else:
            normalized = self.norm(concatenated)

        features = self.dropout(self.activation(normalized))
        features = features.transpose(1, 2)
        features = features.reshape(-1, self.out_channels)

        return features


class GCNBlock(nn.Module):
    """
    Graph Convolutional Network block
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
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.input_proj = None
        if in_channels != hidden_dim:
            self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if self.input_proj is not None:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim if i > 0 or (in_channels == hidden_dim) else in_channels
            self.gcn_layers.append(
                GCNConv(in_dim, out_channels if i == num_layers - 1 else hidden_dim)
            )
            self.norms.append(nn.LayerNorm(out_channels if i == num_layers - 1 else hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)

        residual = x

        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            x = gcn(x, edge_index)
            
            if i < self.num_layers - 1:
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)

                if self.use_residual and x.shape == residual.shape:
                    x = x + residual
                    residual = x
            else:
                x = norm(x)

        return x


class ClassQueryHead(nn.Module):
    """
    Class-Query classification head using Cross-Attention
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_decoder: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_decoder = use_decoder

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        if use_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def prune_heads(self, valid_class_indices):
        """
        Physically prune the class queries to only include valid indices.
        """
        with torch.no_grad():
            new_queries = self.class_queries.data[valid_class_indices].clone()
            self.class_queries = nn.Parameter(new_queries)
            self.num_classes = len(valid_class_indices)
            print(f"ClassQueryHead Pruned: {len(valid_class_indices)} classes remaining.")

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        device = node_features.device

        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        max_nodes = 1001 

        memory = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for b in range(batch_size):
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask] 

            num_batch_nodes = batch_nodes.size(0)
            memory[b, :num_batch_nodes, :] = batch_nodes
            mask[b, num_batch_nodes:] = True

        memory_mask = mask

        attended_features = self.cross_attention(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_mask
        )

        logits = self.output_proj(attended_features)
        logits = logits.squeeze(-1)

        return logits.to(node_features.device)


class ClassQueryHeadPooling(nn.Module):
    """
    Simplified Class-Query head using attention pooling
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.attention_scale = hidden_dim ** 0.5

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def prune_heads(self, valid_class_indices):
        with torch.no_grad():
            new_queries = self.class_queries.data[valid_class_indices].clone()
            self.class_queries = nn.Parameter(new_queries)
            self.num_classes = len(valid_class_indices)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight return for supervision.

        Args:
            node_features: Node features from GCN
            batch: Batch assignment vector

        Returns:
            logits: Classification logits [Batch_Size, Num_Classes]
            attn_weights: Attention weights [Batch_Size, Num_Classes, Seq_Len=1001]
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        queries = self.class_queries.to(device)

        logits_list = []
        attn_weights_list = []

        for b in range(batch_size):
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask]

            # scores: [Num_Classes, Seq_Len]
            scores = torch.matmul(queries, batch_nodes.t()) / self.attention_scale
            # attn_weights: [Num_Classes, Seq_Len]
            attn_weights = torch.softmax(scores, dim=1)
            aggregated = torch.matmul(attn_weights, batch_nodes)
            class_logits = self.output_proj(aggregated)

            logits_list.append(class_logits.squeeze(-1))
            attn_weights_list.append(attn_weights)

        logits = torch.stack(logits_list, dim=0)
        # Stack attention weights: [Batch_Size, Num_Classes, Seq_Len]
        all_attn_weights = torch.stack(attn_weights_list, dim=0)

        return logits.to(node_features.device), all_attn_weights.to(node_features.device)


# class HierarchicalClassQueryHeadPooling(nn.Module):
#     """
#     Hierarchical Class-Query head with Group-to-Class derivation.
#     """

#     GROUP_TO_CLASS_INDICES = {
#         'A': [0, 1, 7, 9, 10], 
#         'C': [2, 6, 8],
#         'G': [3, 11],
#         'U': [4, 5] 
#     }

#     GROUP_SIZES = {'A': 5, 'C': 3, 'G': 2, 'U': 2}
#     GROUP_ORDER = ['A', 'C', 'G', 'U']

#     def __init__(
#         self,
#         hidden_dim: int = 128,
#         dropout: float = 0.1
#     ):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.num_groups = 4
#         self.num_classes = 12

#         self.group_queries = nn.Parameter(torch.randn(self.num_groups, hidden_dim))

#         self.group_mlps = nn.ModuleDict({
#             'A': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['A'], dropout),
#             'C': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['C'], dropout),
#             'G': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['G'], dropout),
#             'U': self._make_derivation_mlp(hidden_dim, self.GROUP_SIZES['U'], dropout)
#         })

#         self.attention_scale = hidden_dim ** 0.5

#         self.output_proj_4class = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#         self.output_proj_12class = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def _make_derivation_mlp(self, hidden_dim: int, num_outputs: int, dropout: float) -> nn.Module:
#         return nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim * num_outputs)
#         )

#     def _derive_class_queries(self) -> torch.Tensor:
#         derived_queries = {}

#         for group_idx, group_name in enumerate(self.GROUP_ORDER):
#             group_query = self.group_queries[group_idx] 
#             mlp = self.group_mlps[group_name]
#             output = mlp(group_query) 
#             class_queries = output.view(self.GROUP_SIZES[group_name], self.hidden_dim) 
#             derived_queries[group_name] = class_queries

#         all_queries = []
#         for group_name in self.GROUP_ORDER:
#             all_queries.append(derived_queries[group_name])

#         concatenated = torch.cat(all_queries, dim=0)

#         reorder_indices = []
#         for group_name in self.GROUP_ORDER:
#             reorder_indices.extend(self.GROUP_TO_CLASS_INDICES[group_name])

#         ordered_queries = torch.zeros_like(concatenated)
#         for i, idx in enumerate(reorder_indices):
#             ordered_queries[idx] = concatenated[i]

#         return ordered_queries 

#     def prune_heads(self, valid_class_indices, valid_group_indices):
#         """
#         Prune the head to only compute specific classes and groups via index masking.
#         """
#         self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
#         self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))
#         print(f"Hierarchical Head Pruned: Active Classes={valid_class_indices}, Active Groups={valid_group_indices}")

#     def forward(
#         self,
#         node_features: torch.Tensor,
#         batch: torch.Tensor
#     ) -> tuple:
#         all_group_queries = self.group_queries 
#         all_class_queries = self._derive_class_queries() 

#         # [Pruning Logic]
#         if hasattr(self, 'valid_group_indices'):
#             group_queries = all_group_queries[self.valid_group_indices] 
#         else:
#             group_queries = all_group_queries

#         if hasattr(self, 'valid_class_indices'):
#             class_queries = all_class_queries[self.valid_class_indices] 
#         else:
#             class_queries = all_class_queries

#         # Group Task
#         scores_4 = torch.matmul(node_features, group_queries.t()) / self.attention_scale
#         attn_weights_4 = softmax(scores_4, batch, dim=0)
#         weighted_4 = attn_weights_4.unsqueeze(-1) * node_features.unsqueeze(1)
#         num_active_groups = group_queries.size(0)
#         agg_4 = global_add_pool(weighted_4.flatten(1), batch).view(-1, num_active_groups, self.hidden_dim)
#         logits_4class = self.output_proj_4class(agg_4).squeeze(-1)

#         # Class Task
#         scores_12 = torch.matmul(node_features, class_queries.t()) / self.attention_scale
#         attn_weights_12 = softmax(scores_12, batch, dim=0)
#         weighted_12 = attn_weights_12.unsqueeze(-1) * node_features.unsqueeze(1)
#         num_active_classes = class_queries.size(0)
#         agg_12 = global_add_pool(weighted_12.flatten(1), batch).view(-1, num_active_classes, self.hidden_dim)
#         logits_12class = self.output_proj_12class(agg_12).squeeze(-1)

#         return logits_12class, logits_4class


class HierarchicalClassQueryHeadPooling(nn.Module):
    def __init__(self, hidden_dim, num_classes, group_to_class_indices, dropout=0.1, use_layer_norm=True):
        """
        Hierarchical Head with Attention Pooling and Query Derivation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices
        self.num_groups = 4 # A, C, G, U

        # Map group indices to group names: 0->'A', 1->'C', 2->'G', 3->'U'
        self.group_names = ['A', 'C', 'G', 'U']

        # 1. Group Queries (Trainable parameters) [4, Hidden_Dim]
        self.group_queries = nn.Parameter(torch.randn(self.num_groups, hidden_dim))

        # 2. Group-wise Independent Projectors (Derivation)
        self.group_projectors = nn.ModuleList()
        print(f"Initializing HierarchicalClassQueryHeadPooling with group_to_class_indices: {group_to_class_indices}")
        for g_idx in range(self.num_groups):
            group_name = self.group_names[g_idx]
            if group_name in group_to_class_indices:
                num_subclasses = len(group_to_class_indices[group_name])
                print(f"  Group {g_idx} ('{group_name}'): {num_subclasses} subclasses, indices: {group_to_class_indices[group_name]}")
            else:
                num_subclasses = 0
                print(f"  Group {g_idx} ('{group_name}'): NOT found in group_to_class_indices")
            
            # MLP: Group_Query -> Subclass_Queries
            projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_subclasses * hidden_dim)
            )
            self.group_projectors.append(projector)

        # 3. Output Projections
        self.output_proj_12 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_proj_4 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Scale factor for attention
        self.attention_scale = hidden_dim ** 0.5

    def _derive_class_queries(self):
        """
        Derive Class Queries from Group Queries using projectors.
        """
        all_sub_queries = []
        all_global_indices = []

        for g_idx in range(self.num_groups):
            group_name = self.group_names[g_idx]
            if group_name not in self.group_to_class_indices:
                continue
            
            g_query = self.group_queries[g_idx].unsqueeze(0) 
            sub_flat = self.group_projectors[g_idx](g_query)
            
            num_subs = len(self.group_to_class_indices[group_name])
            sub_queries = sub_flat.view(num_subs, self.hidden_dim)
            
            all_sub_queries.append(sub_queries)
            all_global_indices.extend(self.group_to_class_indices[group_name])

        flat_queries = torch.cat(all_sub_queries, dim=0)
        indices_tensor = torch.tensor(all_global_indices, device=flat_queries.device)
        
        # Use the same dtype as flat_queries to handle mixed precision (AMP)
        ordered_queries = torch.zeros(self.num_classes, self.hidden_dim, 
                                   dtype=flat_queries.dtype, 
                                   device=flat_queries.device)
        ordered_queries.index_copy_(0, indices_tensor, flat_queries)
        
        return ordered_queries

    def forward(self, node_features: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            node_features: [Total_Nodes, Dim]
            batch: [Total_Nodes]
        Returns:
            logits_12: [Batch, 12]
            logits_4: [Batch, 4]
            attn_weights_12: [Batch, 12, Seq_Len] (For supervision/visualization)
        """
        batch_size = batch.max().item() + 1
        device = node_features.device
        
        # 1. Prepare Queries
        class_queries = self._derive_class_queries() # [12, Dim]
        group_queries = self.group_queries           # [4, Dim]

        logits_12_list = []
        logits_4_list = []
        attn_weights_12_list = []

        # 2. Iterate over batch (Attention Pooling)
        for b in range(batch_size):
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask] # [Seq_Len, Dim]
            
            # --- 12-Class Task ---
            # Score: [12, Dim] @ [Dim, Seq_Len] -> [12, Seq_Len]
            scores_12 = torch.matmul(class_queries, batch_nodes.t()) / self.attention_scale
            attn_12 = torch.softmax(scores_12, dim=1) # Attention Weights
            
            # Pooling: [12, Seq_Len] @ [Seq_Len, Dim] -> [12, Dim]
            weighted_nodes_12 = torch.matmul(attn_12, batch_nodes)
            
            # Predict
            logits_12 = self.output_proj_12(weighted_nodes_12).squeeze(-1) # [12]
            
            # --- 4-Class Task (Group) ---
            # Score: [4, Dim] @ [Dim, Seq_Len] -> [4, Seq_Len]
            scores_4 = torch.matmul(group_queries, batch_nodes.t()) / self.attention_scale
            attn_4 = torch.softmax(scores_4, dim=1)
            
            # Pooling
            weighted_nodes_4 = torch.matmul(attn_4, batch_nodes)
            
            # Predict
            logits_4 = self.output_proj_4(weighted_nodes_4).squeeze(-1) # [4]

            # Collect results
            logits_12_list.append(logits_12)
            logits_4_list.append(logits_4)
            attn_weights_12_list.append(attn_12)

        # Stack results
        logits_12_final = torch.stack(logits_12_list, dim=0) # [Batch, 12]
        logits_4_final = torch.stack(logits_4_list, dim=0)   # [Batch, 4]
        attn_weights_12_final = torch.stack(attn_weights_12_list, dim=0) # [Batch, 12, Seq_Len]

        # 返回三个值：12类Logits, 4类Logits, 12类Attention权重
        return logits_12_final, logits_4_final, attn_weights_12_final

class RNA_ClassQuery_Model(nn.Module):
    """
    RNA Classification Model using Multi-scale CNN + GCN + Class-Query Attention
    """

    def __init__(
        self,
        cnn_hidden_dim: int = 64,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        cnn_dropout: float = 0.1,
        gcn_hidden_dim: int = 128,
        gcn_out_channels: int = 128,
        gcn_num_layers: int = 3,
        gcn_dropout: float = 0.3,
        num_classes: int = 12,
        num_attn_heads: int = 4,
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

        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout
        )

        self.gcn_block = GCNBlock(
            in_channels=self.cnn_out_channels,
            hidden_dim=gcn_hidden_dim,
            out_channels=gcn_out_channels,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
            use_residual=True
        )

        if use_hierarchical:
            self.class_query_head = HierarchicalClassQueryHeadPooling(
                hidden_dim=gcn_out_channels,
                num_classes=num_classes,
                group_to_class_indices=GROUP_TO_CLASS_INDICES,
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

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        """
        Public interface to prune the classification head for specific tasks.
        """
        if self.use_hierarchical:
            if valid_group_indices is None:
                raise ValueError("Hierarchical model requires valid_group_indices for pruning.")
            self.class_query_head.prune_heads(valid_class_indices, valid_group_indices)
        else:
            # For standard head, we just pass the class indices
            if hasattr(self.class_query_head, 'prune_heads'):
                 self.class_query_head.prune_heads(valid_class_indices)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple:
        """
        Forward pass with optional attention weight return.

        Args:
            x: Input features
            edge_index: Graph edge indices
            batch: Batch assignment vector
            return_attention: If True, return attention weights (only works with use_simple_pooling=True)

        Returns:
            If use_hierarchical: (logits_12class, logits_4class)
            Elif use_simple_pooling and return_attention: (logits, attn_weights)
            Else: logits
        """
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

        elif x.dim() == 2:
            assert batch is not None, "batch vector must be provided for PyG format input"
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        node_features = self.cnn_block(x, batch)
        node_features = self.gcn_block(node_features, edge_index)

        if self.use_hierarchical:
            # Hierarchical head returns 3 values: logits_12class, logits_4class, attn_weights_12
            logits_12class, logits_4class, attn_weights_12 = self.class_query_head(node_features, batch)

            if return_attention:
                return logits_12class, logits_4class, attn_weights_12
            else:
                # Training typically only needs logits
                return logits_12class, logits_4class

        elif self.use_simple_pooling:
            logits, attn_weights = self.class_query_head(node_features, batch)
            if return_attention:
                return logits, attn_weights
            return logits
        else:
            logits = self.class_query_head(node_features, batch)
            return logits
