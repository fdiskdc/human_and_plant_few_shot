
"""
main_model_multirm.py - RGCNFormer 多尺度类查询模型 (MultIRM 51nt窗口) / RGCNFormer multi-scale class-query model (MultIRM 51nt window)

为 MultIRM 数据集定制的 RGCNFormer 主模型。序列长度固定为 51nt，类别与组层级定义：
A (Am/m1A/m6A/m6Am/AtoI)、C (Cm/m5C)、G (Gm/m7G)、U (Um/m5U/Psi)。
使用层级 HierarchicalClassQueryHeadPooling (内置 group_to_class_indices，非外部导入)。
A MultIRM-dataset-tailored RGCNFormer. Sequence length fixed at 51nt with the canonical
4-group-to-12-class hierarchy (A/C/G/U). Uses an inlined HierarchicalClassQueryHeadPooling
with embedded group_to_class_indices (no external GROUP_TO_CLASS_INDICES dependency).

功能模块 / Modules:
- ParallelCNNBlock: 多尺度并行 1D CNN (1/3/5/7 核, 51nt 序列) / Multi-scale parallel 1D CNN (kernels 1/3/5/7, 51nt sequences)
- GCNBlock: 残差图卷积块 (多层 GCNConv + LayerNorm) / Residual GCN block (multi-layer GCNConv + LayerNorm)
- HierarchicalClassQueryHeadPooling: 内嵌的 4-组到 12-类层级查询头 (MultIRM 专属组映射) / Inlined 4-group-to-12-class hierarchical query head (MultIRM-specific group mapping)
- RNA_ClassQuery_Model: 端到端 MultIRM 模型 / End-to-end MultIRM model

输入 / Inputs:
- x: (B, 51, 4) 或 (Total_Nodes, 4) one-hot 51nt 子序列 / (B, 51, 4) or (Total_Nodes, 4) one-hot 51nt subsequence
- edge_index: (2, E) PyG 边索引 / (2, E) PyG edge indices
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector

输出 / Outputs:
- return_attention=False: (logits_12 [B,12], logits_4 [B,4]) / (logits_12 [B,12], logits_4 [B,4])
- return_attention=True: (logits_12 [B,12], logits_4 [B,4], attn_weights_12 [B,12,51]) / (logits_12 [B,12], logits_4 [B,4], attn_weights_12 [B,12,51])

数据流 / Data Flow:
1. 51nt one-hot 子序列进入多尺度 CNN 提取 k-mer 局部特征 / 51nt one-hot enters multi-scale CNN for k-mer local features
2. CNN 输出 reshape 为图节点，经 GCN 残差块进行图结构传播 / CNN output reshaped to graph nodes, propagated through GCN residual block
3. 内嵌的层级头使用 MultIRM 专属 group_to_class_indices 通过组查询派生 12 类查询 / Inlined hierarchical head uses MultIRM-specific group_to_class_indices to derive 12 class queries
4. 两个 MHA (12 类 + 4 组) 分别交叉注意力到密集节点特征，输出双层 logits / Two MHAs (12-class + 4-group) cross-attend to dense node features, output dual-level logits

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch_geometric.data, torch_geometric.nn.GCNConv / torch, torch.nn, torch_geometric.data, torch_geometric.nn.GCNConv
- 被调用 / Called by: train_human_multirm.py, train_multirm_dataset.py, test_multirm_4class.py, test_multirm_oversampling.py, inference_multirm_segmented.py / train_human_multirm.py, train_multirm_dataset.py, test_multirm_4class.py, test_multirm_oversampling.py, inference_multirm_segmented.py

使用示例 / Usage Example:
    from model.main_model_multirm import RNA_ClassQuery_Model
    model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
    logits_12, logits_4 = model(x, edge_index, batch)  # 51nt

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""


import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple


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
        dropout: float = 0.1,
        seq_len: int = 51  # MultIRM sequence length
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.seq_len = seq_len
        # Each branch outputs hidden_dim // len(kernel_sizes), concatenated to hidden_dim
        self.out_channels = hidden_dim
        self.branch_out_channels = hidden_dim // len(kernel_sizes)

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.branch_out_channels,
                kernel_size=k,
                padding='same',
                bias=True
            )
            for k in kernel_sizes
        ])

        if use_layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, seq_len))
        else:
            self.norm = nn.BatchNorm1d(self.out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == self.seq_len and x.size(2) == 4:
            x = x.transpose(1, 2)
        elif x.dim() == 2 and x.size(1) == 4:
            if batch is not None:
                batch_size = batch.max().item() + 1
                x = x.view(batch_size, self.seq_len, 4).transpose(1, 2)
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


class HierarchicalClassQueryHeadPooling(nn.Module):
    def __init__(self, hidden_dim, num_classes=12, dropout=0.1, use_layer_norm=True, num_heads=8):
        """
        Hierarchical Head with Attention Pooling and Query Derivation for MultIRM.
        Uses PyTorch MultiheadAttention for efficient parallel computation.
        
        4-class grouping rule:
        - A (Adenine): Am(0), m1A(4), m6A(7), m6Am(8), AtoI(11)
        - C (Cytosine): Cm(1), m5C(5)
        - G (Guanine): Gm(2), m7G(9)
        - U (Uracil): Um(3), m5U(6), Psi(10)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.seq_len = 51  # MultIRM sequence length

        # Map group indices to group names: 0->'A', 1->'C', 2->'G', 3->'U'
        self.group_names = ['A', 'C', 'G', 'U']

        # Group to class indices mapping for MultIRM (new order)
        # Group 0 (A): Am(0), m1A(4), m6A(7), m6Am(8), AtoI(11)
        # Group 1 (C): Cm(1), m5C(5)
        # Group 2 (G): Gm(2), m7G(9)
        # Group 3 (U): Um(3), m5U(6), Psi(10)
        self.group_to_class_indices = {
            'A': [0, 4, 7, 8, 11],    # Am, m1A, m6A, m6Am, AtoI
            'C': [1, 5],               # Cm, m5C
            'G': [2, 9],               # Gm, m7G
            'U': [3, 6, 10]            # Um, m5U, Psi
        }

        # 1. Group Queries (Trainable parameters) [4, Hidden_Dim]
        self.group_queries = nn.Parameter(torch.randn(4, hidden_dim))

        # 2. Group-wise Independent Projectors (Derivation)
        self.group_projectors = nn.ModuleList()
        print(f"Initializing HierarchicalClassQueryHeadPooling for MultIRM with group_to_class_indices: {self.group_to_class_indices}")
        for g_idx in range(4):
            group_name = self.group_names[g_idx]
            class_indices = self.group_to_class_indices[group_name]
            num_subclasses = len(class_indices)
            print(f"  Group {g_idx} ('{group_name}'): {num_subclasses} subclasses, indices: {class_indices}")

            # MLP: Group_Query -> Subclass_Queries
            projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_subclasses * hidden_dim)
            )
            self.group_projectors.append(projector)

        # 3. MultiheadAttention Layers for efficient parallel computation
        # MHA for 12-class task
        self.mha_12 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MHA for 4-group task
        self.mha_4 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 4. Output Projections
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

    def prune_heads(self, valid_class_indices, valid_group_indices):
        """
        Prune head to only compute specific classes and groups via index masking.
        
        Args:
            valid_class_indices: List of valid class indices to keep
            valid_group_indices: List of valid group indices to keep
        """
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))
        print(f"Hierarchical Head Pruned: Active Classes={valid_class_indices}, Active Groups={valid_group_indices}")

    def _derive_class_queries(self):
        """
        Derive Class Queries from Group Queries using projectors.
        
        This function implements the 4query-12query generation logic according to the rule:
        - A (Adenine): m6A, m1A, m6Am, Am, AtoI
        - C (Cytosine): m5C, Cm
        - G (Guanine): m7G, Gm
        - U (Uracil): m5U, Psi (Ψ), Um
        """
        all_sub_queries = []
        all_global_indices = []

        for g_idx in range(4):
            group_name = self.group_names[g_idx]
            class_indices = self.group_to_class_indices[group_name]
            
            g_query = self.group_queries[g_idx].unsqueeze(0)  # [1, Hidden_Dim]
            sub_flat = self.group_projectors[g_idx](g_query)
            
            num_subs = len(class_indices)
            sub_queries = sub_flat.view(num_subs, self.hidden_dim)  # [Num_Subclasses, Hidden_Dim]
            
            all_sub_queries.append(sub_queries)
            all_global_indices.extend(class_indices)

        # Concatenate all subclass queries
        flat_queries = torch.cat(all_sub_queries, dim=0)  # [12, Hidden_Dim]
        
        # Reorder to match global class index order [0, 1, 2, ..., 11]
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
            logits_12: [Batch, 12] or [Batch, Num_Valid_Classes] if pruned
            logits_4: [Batch, 4] or [Batch, Num_Valid_Groups] if pruned
            attn_weights_12: [Batch, 12, Seq_Len] or [Batch, Num_Valid_Classes, Seq_Len] if pruned
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        # 1. Prepare Queries
        class_queries = self._derive_class_queries()  # [12, Dim]
        group_queries = self.group_queries            # [4, Dim]

        # 2. Apply pruning if valid indices are registered
        if hasattr(self, 'valid_class_indices'):
            class_queries = class_queries[self.valid_class_indices]

        if hasattr(self, 'valid_group_indices'):
            group_queries = group_queries[self.valid_group_indices]

        # 3. Prepare Inputs (Dense Batch)
        # Reshape node_features from [Total_Nodes, Dim] to [Batch, Seq_Len, Dim]
        dense_nodes = node_features.view(batch_size, self.seq_len, -1)

        # 4. Prepare Queries - Expand to batch dimension
        # [Num_Classes, Dim] -> [Batch, Num_Classes, Dim]
        queries_12 = class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries_4 = group_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 5. MHA for 12-Class Task (No Loop!)
        # attn_out_12: [Batch, Num_Classes, Dim]
        # attn_weights_12: [Batch, Num_Heads, Num_Classes, Seq_Len] (from MHA)
        # average_attn_weights=True returns [Batch, Num_Classes, Seq_Len]
        attn_out_12, attn_weights_12 = self.mha_12(
            query=queries_12,
            key=dense_nodes,
            value=dense_nodes,
            average_attn_weights=True  # Returns averaged weights: [Batch, Num_Classes, Seq_Len]
        )

        # 6. MHA for 4-Group Task (No Loop!)
        attn_out_4, _ = self.mha_4(
            query=queries_4,
            key=dense_nodes,
            value=dense_nodes,
            average_attn_weights=True
        )

        # 7. Final Projection
        # [Batch, Num_Classes, Dim] -> [Batch, Num_Classes, 1] -> [Batch, Num_Classes]
        logits_12_final = self.output_proj_12(attn_out_12).squeeze(-1)
        logits_4_final = self.output_proj_4(attn_out_4).squeeze(-1)

        # Return: 12-class Logits, 4-class Logits, 12-class Attention Weights
        return logits_12_final.to(device), logits_4_final.to(device), attn_weights_12.to(device)


class RNA_ClassQuery_Model(nn.Module):
    """
    RNA Classification Model using Multi-scale CNN + GCN + Class-Query Attention
    Designed for MultIRM dataset with 12-class hierarchical classification
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
        use_hierarchical: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.cnn_out_channels = cnn_hidden_dim
        self.gcn_out_channels = gcn_out_channels
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical

        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout,
            seq_len=51  # MultIRM sequence length
        )

        self.gcn_block = GCNBlock(
            in_channels=self.cnn_out_channels,
            hidden_dim=gcn_hidden_dim,
            out_channels=gcn_out_channels,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
            use_residual=True
        )

        # Use HierarchicalClassQueryHeadPooling for MultIRM
        self.class_query_head = HierarchicalClassQueryHeadPooling(
            hidden_dim=gcn_out_channels,
            num_classes=num_classes,
            dropout=attn_dropout,
            use_layer_norm=use_layer_norm
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
            raise NotImplementedError("Pruning is only supported for hierarchical mode")

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
            return_attention: If True, return attention weights

        Returns:
            If return_attention: (logits_12class, logits_4class, attn_weights_12)
            Else: (logits_12class, logits_4class)
        """
        if isinstance(x, Data) or isinstance(x, Batch):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
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

        # Hierarchical head returns 3 values: logits_12class, logits_4class, attn_weights_12
        logits_12class, logits_4class, attn_weights_12 = self.class_query_head(node_features, batch)

        if return_attention:
            return logits_12class, logits_4class, attn_weights_12
        else:
            return logits_12class, logits_4class