
"""
mrmodn_collect_atten.py - RGCNFormer 收集注意力输出变体 (1001nt) / RGCNFormer variant that collects attention outputs (1001nt)

mrmodn.py 的修改版，将 MultiheadAttention 的中间输出 (attn_out_12, attn_out_4) 一并返回，
用于可视化和下游分析。前向签名扩展为 5 元组。
A modified version of mrmodn.py that returns the MultiheadAttention intermediate
outputs (attn_out_12, attn_out_4) for visualization and downstream analysis. The forward
signature is extended to a 5-tuple.

功能模块 / Modules:
- ParallelCNNBlock: 多尺度并行 1D CNN 块 (1/3/5/7 核) / Multi-scale parallel 1D CNN block (kernels 1/3/5/7)
- GCNBlock: 残差图卷积块 / Residual GCN block
- HierarchicalClassQueryHeadPooling: 层级头 (forward 返回 attn_out 而非仅 logits) / Hierarchical head (forward returns attn_out instead of only logits)
- RNA_ClassQuery_Model_Collect_Atten: 端到端模型，专门用于收集注意力输出 / End-to-end model specialized for collecting attention outputs

输入 / Inputs:
- x: (B, 1001, 4) 或 (Total_Nodes, 4) one-hot RNA 序列 / (B, 1001, 4) or (Total_Nodes, 4) one-hot RNA sequence
- edge_index: (2, E) PyG 边索引 / (2, E) PyG edge indices
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector
- return_attention: 总是返回注意力 (接口保留) / always returns attention (interface reserved)

输出 / Outputs:
- 5 元组 / 5-tuple: (logits_12 [B,12], logits_4 [B,4], attn_out_12 [B,12,Dim], attn_out_4 [B,4,Dim], attn_weights_12 [B,12,1001]) / (logits_12 [B,12], logits_4 [B,4], attn_out_12 [B,12,Dim], attn_out_4 [B,4,Dim], attn_weights_12 [B,12,1001])

数据流 / Data Flow:
1. one-hot 进入多尺度 CNN 提取 k-mer 特征 / one-hot enters multi-scale CNN for k-mer features
2. GCN 残差块进行图传播 / GCN residual block performs graph propagation
3. 层级头 (含 2 个 MHA) 返回 5 个值：双层 logits、attn_out_12、attn_out_4、attn_weights_12 / Hierarchical head (2 MHAs) returns 5 values: dual-level logits, attn_out_12, attn_out_4, attn_weights_12
4. attn_out_* 是 MHA 在 value 上的加权汇聚，可用于 UMAP / 可视化下游分析 / attn_out_* are MHA's weighted aggregations over values, used for UMAP / visualization downstream

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch_geometric.data, torch_geometric.nn.GCNConv, utils.common.GROUP_TO_CLASS_INDICES / torch, torch.nn, torch_geometric.data, torch_geometric.nn.GCNConv, utils.common.GROUP_TO_CLASS_INDICES
- 被调用 / Called by: collect_human_atten.py, prepare_umap_data.py, inference_mrmodn_full.py / collect_human_atten.py, prepare_umap_data.py, inference_mrmodn_full.py

使用示例 / Usage Example:
    from model.mrmodn_collect_atten import RNA_ClassQuery_Model_Collect_Atten
    model = RNA_ClassQuery_Model_Collect_Atten(num_classes=12, use_hierarchical=True)
    l12, l4, ao12, ao4, aw = model(x, edge_index, batch)

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

# Import GROUP_TO_CLASS_INDICES for hierarchical head
from utils.common import GROUP_TO_CLASS_INDICES


# ============================================================================
# Sub-modules for RNA_ClassQuery_Model
# ============================================================================

class ParallelCNNBlock(nn.Module):
    """
    多尺度并行一维卷积块 / Multi-scale parallel 1D CNN feature extraction block.

    Attributes / 属性:
        conv_branches (nn.ModuleList): [中文] 4 个并行 1D 卷积 / [English] 4 parallel 1D convs.
        norm (nn.Module): [中文] LayerNorm 或 BatchNorm1d / [English] LayerNorm or BatchNorm1d.
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
        初始化 ParallelCNNBlock / Initialize ParallelCNNBlock.

        Args / 参数:
            in_channels (int): [中文] 输入通道 / [English] input channels. Defaults to 4.
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim. Defaults to 64.
            kernel_sizes (Tuple[int, ...]): [中文] 卷积核 / [English] kernel sizes. Defaults to (1,3,5,7).
            use_layer_norm (bool): [中文] 使用 LayerNorm / [English] use LayerNorm. Defaults to True.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
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
            self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, 1001))
        else:
            self.norm = nn.BatchNorm1d(self.out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：多尺度卷积 → 归一化 → 激活 / Forward: multi-scale conv → norm → activation.

        Args / 参数:
            x (torch.Tensor): [中文] 输入 / [English] input.
            batch (Optional[torch.Tensor]): [中文] PyG 批索引 / [English] PyG batch index.

        Returns / 返回:
            torch.Tensor: [中文] `(Total_Nodes, hidden_dim)` 节点特征 / [English] node features.
        """
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
    多层残差图卷积块 / Multi-layer residual GCN block.

    Attributes / 属性:
        gcn_layers (nn.ModuleList): [中文] GCN 卷积层 / [English] GCN conv layers.
        norms (nn.ModuleList): [中文] LayerNorm / [English] LayerNorms.
        use_residual (bool): [中文] 是否残差 / [English] use residual.
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
        初始化 GCNBlock / Initialize GCNBlock.

        Args / 参数:
            in_channels (int): [中文] 输入特征维度 / [English] input feature dim.
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim. Defaults to 128.
            out_channels (int): [中文] 输出维度 / [English] output dim. Defaults to 128.
            num_layers (int): [中文] GCN 层数 / [English] GCN layers. Defaults to 3.
            dropout (float): [中文] dropout / [English] dropout. Defaults to 0.3.
            use_residual (bool): [中文] 残差 / [English] use residual. Defaults to True.
        """
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
        """
        前向传播：多层 GCN + 残差 / Forward: multi-layer GCN + residual.

        Args / 参数:
            x (torch.Tensor): [中文] 节点特征 / [English] node features.
            edge_index (torch.Tensor): [中文] 边索引 / [English] edge indices.

        Returns / 返回:
            torch.Tensor: [中文] 输出节点特征 / [English] output node features.
        """
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
    """
    层级类查询头（收集注意力变体）/ Hierarchical class-query head (collect-attn variant).

    修改自 HierarchicalClassQueryHeadPooling，额外返回注意力输出向量。
    Modified to return attention outputs (attn_out) in addition to weights.

    Attributes / 属性:
        num_groups (int): [中文] 组数 (固定 4) / [English] number of groups (fixed 4).
        group_queries (nn.Parameter): [中文] 组查询 / [English] group queries.
        mha_12, mha_4 (nn.MultiheadAttention): [中文] 12 类 / 4 组 MHA / [English] 12-class / 4-group MHA.
    """

    def __init__(self, hidden_dim, num_classes, group_to_class_indices, dropout=0.1, use_layer_norm=True, num_heads=8):
        """
        初始化 HierarchicalClassQueryHeadPooling / Initialize HierarchicalClassQueryHeadPooling.

        Args / 参数:
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim.
            num_classes (int): [中文] 类别数 / [English] number of classes.
            group_to_class_indices (Dict[str, List[int]]): [中文] 组→子类映射 / [English] group-to-class mapping.
            dropout (float): [中文] dropout / [English] dropout. Defaults to 0.1.
            use_layer_norm (bool): [中文] 用 LayerNorm / [English] use LayerNorm. Defaults to True.
            num_heads (int): [中文] MHA 头数 / [English] MHA heads. Defaults to 8.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices
        self.num_groups = 4  # A, C, G, U
        self.num_heads = num_heads
        self.seq_len = 1001

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

            projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_subclasses * hidden_dim)
            )
            self.group_projectors.append(projector)

        # 3. MultiheadAttention Layers
        self.mha_12 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

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
        剪枝到指定的类和组 / Prune to only compute specific classes and groups via index masking.

        Args / 参数:
            valid_class_indices (List[int]): [中文] 类索引 / [English] class indices.
            valid_group_indices (List[int]): [中文] 组索引 / [English] group indices.
        """
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))
        print(f"Hierarchical Head Pruned: Active Classes={valid_class_indices}, Active Groups={valid_group_indices}")

    def _derive_class_queries(self):
        """
        从组查询派生子类查询 / Derive class queries from group queries.

        Returns / 返回:
            torch.Tensor: [中文] 类查询 `(num_classes, hidden_dim)` / [English] class queries.
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
        
        ordered_queries = torch.zeros(self.num_classes, self.hidden_dim,
                                   dtype=flat_queries.dtype,
                                   device=flat_queries.device)
        ordered_queries.index_copy_(0, indices_tensor, flat_queries)
        
        return ordered_queries

    def forward(self, node_features: torch.Tensor, batch: torch.Tensor):
        """
        前向传播（返回注意力输出向量） / Modified forward to return attention outputs.

        Args / 参数:
            node_features (torch.Tensor): [中文] 节点特征 / [English] node features.
            batch (torch.Tensor): [中文] PyG 批索引 / [English] PyG batch index.

        Returns / 返回:
            Tuple: [中文] `(logits_12, logits_4, attn_out_12, attn_out_4, attn_weights_12)` / [English] five tensors.
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
        dense_nodes = node_features.view(batch_size, self.seq_len, -1)

        # 4. Prepare Queries - Expand to batch dimension
        queries_12 = class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries_4 = group_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 5. MHA for 12-Class Task
        attn_out_12, attn_weights_12 = self.mha_12(
            query=queries_12,
            key=dense_nodes,
            value=dense_nodes,
            average_attn_weights=True
        )

        # 6. MHA for 4-Group Task
        attn_out_4, _ = self.mha_4(
            query=queries_4,
            key=dense_nodes,
            value=dense_nodes,
            average_attn_weights=True
        )

        # 7. Final Projection
        logits_12_final = self.output_proj_12(attn_out_12).squeeze(-1)
        logits_4_final = self.output_proj_4(attn_out_4).squeeze(-1)

        # Return: 12-class Logits, 4-class Logits, attn_out_12, attn_out_4, attn_weights_12
        return logits_12_final.to(device), logits_4_final.to(device), attn_out_12.to(device), attn_out_4.to(device), attn_weights_12.to(device)


class RNA_ClassQuery_Model_Collect_Atten(nn.Module):
    """
    RNA Classification Model modified to collect attention outputs
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

        self.cnn_out_channels = cnn_hidden_dim
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
        else:
            raise ValueError("RNA_ClassQuery_Model_Collect_Atten only supports use_hierarchical=True")

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        """
        Public interface to prune the classification head for specific tasks.
        """
        if self.use_hierarchical:
            if valid_group_indices is None:
                raise ValueError("Hierarchical model requires valid_group_indices for pruning.")
            self.class_query_head.prune_heads(valid_class_indices, valid_group_indices)
        else:
            raise ValueError("Pruning only supported for hierarchical model")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple:
        """
        Forward pass that returns attention outputs.

        Args:
            x: Input features
            edge_index: Graph edge indices
            batch: Batch assignment vector
            return_attention: Always returns attention outputs for this model

        Returns:
            (logits_12class, logits_4class, attn_out_12, attn_out_4, attn_weights_12)
            - logits_12class: [Batch, 12] 12-class logits
            - logits_4class: [Batch, 4] 4-class logits
            - attn_out_12: [Batch, 12, Dim] attention output vectors
            - attn_out_4: [Batch, 4, Dim] group attention output vectors
            - attn_weights_12: [Batch, 12, Seq_Len] attention weights over sequence positions
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

        # Hierarchical head returns 5 values: logits_12class, logits_4class, attn_out_12, attn_out_4, attn_weights_12
        logits_12class, logits_4class, attn_out_12, attn_out_4, attn_weights_12 = self.class_query_head(node_features, batch)

        return logits_12class, logits_4class, attn_out_12, attn_out_4, attn_weights_12