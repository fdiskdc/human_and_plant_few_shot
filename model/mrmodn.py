
"""
mrmodn.py - RGCNFormer 多尺度类查询分类模型 (人类12类mRNA修饰) / RGCNFormer multi-scale class-query classification model (human 12-class mRNA modification)

实现RNA 12类多标签修饰分类主模型，结合多尺度CNN局部特征提取、GCN图结构特征传播，
以及基于可学习类查询的注意力分类头。序列长度固定为1001nt。
Supports three classification head modes: standard Class-Query attention, simple attention pooling,
and hierarchical 4-group (A/C/G/U) to 12-class query derivation. Used as the primary backbone
for human multi-label RNA modification prediction at full sequence length.

功能模块 / Modules:
- ParallelCNNBlock: 多尺度并行一维卷积块 (核大小 1/3/5/7) / Multi-scale parallel 1D CNN with kernel sizes 1/3/5/7
- GCNBlock: 残差图卷积块 (多层 GCNConv + LayerNorm) / Residual GCN block (multi-layer GCNConv + LayerNorm)
- ClassQueryHead: 基于 TransformerDecoder 的类查询交叉注意力头 / TransformerDecoder-based class-query cross-attention head
- ClassQueryHeadPooling: 基于缩放点积注意力的简化类查询池化头 / Scaled dot-product attention pooling class-query head
- HierarchicalClassQueryHeadPooling: 4-组到12-类的层级查询派生 + MHA 头 / 4-group-to-12-class hierarchical query derivation + MHA head
- RNA_ClassQuery_Model: 端到端模型，整合 CNN + GCN + 选定的分类头 / End-to-end model integrating CNN + GCN + selected head

输入 / Inputs:
- x: (B, 1001, 4) 或 (Total_Nodes, 4) one-hot RNA序列 / (B, 1001, 4) or (Total_Nodes, 4) one-hot RNA sequence
- edge_index: (2, E) PyG 格式的图边索引 / (2, E) PyG-format graph edge indices
- batch: (Total_Nodes,) 批次分配向量 (PyG 格式时需要) / (Total_Nodes,) batch assignment vector (required for PyG format)

输出 / Outputs:
- 标准头 (training) / Standard head (training): (logits [B,12], None) / (logits [B,12], None)
- 标准头 (eval) / Standard head (eval): logits [B, 12] / logits [B, 12]
- 简单池化头 / Simple pooling head: (logits [B,12], attn_weights [B,12,1001]) / (logits [B,12], attn_weights [B,12,1001])
- 层级头 / Hierarchical head: (logits_12 [B,12], logits_4 [B,4], attn_weights_12 [B,12,1001]) / (logits_12 [B,12], logits_4 [B,4], attn_weights_12 [B,12,1001])

数据流 / Data Flow:
1. one-hot 序列进入多尺度 CNN 提取局部 k-mer 特征 / one-hot sequence enters multi-scale CNN to extract local k-mer features
2. CNN 输出 reshape 为图节点特征，经 GCN 进行图结构传播 (带残差) / CNN output reshaped as graph node features, propagated through GCN with residuals
3. GCN 节点特征送入选定的分类头，生成 12 类 logits (及 4 组 logits) / GCN node features fed to selected head to produce 12-class logits (and 4-group logits)

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch_geometric.nn.GCNConv, utils.common.GROUP_TO_CLASS_INDICES / torch, torch.nn, torch_geometric.nn.GCNConv, utils.common.GROUP_TO_CLASS_INDICES
- 被调用 / Called by: train_human.py, train_plant.py, test_gen3.py, collect_human.py, collect_human_atten.py, prepare_umap_data.py, SpatialMotif.py, 3x3.py, 3x3_2.py, fewshot_*.py / train_human.py, train_plant.py, test_gen3.py, collect_human.py, collect_human_atten.py, prepare_umap_data.py, SpatialMotif.py, 3x3.py, 3x3_2.py, fewshot_*.py

使用示例 / Usage Example:
    from model.mrmodn import RNA_ClassQuery_Model
    model = RNA_ClassQuery_Model(num_classes=12, use_hierarchical=True)
    logits_12, logits_4, attn = model(x, edge_index, batch)  # hierarchical

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
    多尺度并行一维卷积块 (M2D 模块核心) / Multi-scale parallel 1D CNN block (M2D module core).

    使用 4 个不同核大小的并行 1D 卷积捕获 RNA 序列的 k-mer 局部模式 (k=1,3,5,7)。
    拼接后通过 LayerNorm → ReLU → Dropout 产生节点级特征向量。
    Uses 4 parallel 1D convolutions with different kernel sizes (k=1,3,5,7) to capture
    k-mer local patterns. Concatenated output is normalized and activated to produce
    node-level feature vectors.

    Attributes / 属性:
        in_channels (int): [中文] 输入通道数 (固定 4) / [English] input channels (fixed at 4 for ACGU).
        hidden_dim (int): [中文] 隐藏维度 (4 个分支各贡献 1/4) / [English] hidden dim (4 branches contribute 1/4 each).
        kernel_sizes (Tuple[int, ...]): [中文] 卷积核大小元组 / [English] tuple of kernel sizes.
        seq_len (int): [中文] 序列长度 (固定 1001) / [English] sequence length (fixed at 1001).
        conv_branches (nn.ModuleList): [中文] 4 个并行 1D 卷积 / [English] 4 parallel 1D convs.
        norm (nn.Module): [中文] LayerNorm 或 BatchNorm1d / [English] LayerNorm or BatchNorm1d.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        seq_len: int = 1001
    ):
        """
        初始化 ParallelCNNBlock / Initialize the ParallelCNNBlock.

        Args / 参数:
            in_channels (int): [中文] 输入通道数 / [English] input channels. Defaults to 4.
            hidden_dim (int): [中文] 隐藏维度 (必须能被 kernel_sizes 长度整除) / [English] hidden dim.
            kernel_sizes (Tuple[int, ...]): [中文] 卷积核元组 / [English] kernel sizes tuple. Defaults to (1,3,5,7).
            use_layer_norm (bool): [中文] 是否使用 LayerNorm / [English] use LayerNorm. Defaults to True.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
            seq_len (int): [中文] 序列长度 / [English] sequence length. Defaults to 1001.
        """
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
        """
        前向传播：多尺度卷积 → 归一化 → 激活 / Forward pass: multi-scale conv → norm → activation.

        支持两种输入格式：
        - `(B, L, 4)` 密集批处理 (B 个样本，每条长度 L)
        - `(Total_Nodes, 4)` PyG 格式 (需要 `batch` 索引)

        Supports two input formats:
        - `(B, L, 4)` dense batch
        - `(Total_Nodes, 4)` PyG format (requires `batch` index)

        Args / 参数:
            x (torch.Tensor): [中文] 输入 one-hot / [English] input one-hot tensor.
            batch (Optional[torch.Tensor]): [中文] PyG 批索引 / [English] PyG batch index.

        Returns / 返回:
            torch.Tensor: [中文] 节点级特征 `(Total_Nodes, hidden_dim)` / [English] node features.
        """
        if x.dim() == 3 and x.size(1) == self.seq_len and x.size(2) == 4:
            x = x.transpose(1, 2)
        elif x.dim() == 2 and x.size(1) == 4:
            if batch is not None:
                # Dynamic calculation of batch_size and seq_len for variable-length sequences
                batch_size = batch.max().item() + 1
                num_nodes_per_sample = batch.bincount()
                seq_len = num_nodes_per_sample[0].item()  # Assume all samples have same length
                x = x.view(batch_size, seq_len, 4).transpose(1, 2)
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

    使用 `num_layers` 层 `GCNConv` 做图结构特征传播，每层后接 LayerNorm + ReLU + Dropout，
    并通过残差连接缓解过平滑问题。3 层是经验最优配置。
    Uses `num_layers` GCNConv layers with LayerNorm + ReLU + Dropout, plus residual
    connections to mitigate over-smoothing. 3 layers is empirically optimal.

    Attributes / 属性:
        in_channels (int): [中文] 输入特征维度 / [English] input feature dim.
        hidden_dim (int): [中文] GCN 隐藏维度 / [English] GCN hidden dim.
        out_channels (int): [中文] 输出维度 / [English] output dim.
        num_layers (int): [中文] GCN 层数 / [English] number of GCN layers.
        use_residual (bool): [中文] 是否使用残差 / [English] use residual connections.
        gcn_layers (nn.ModuleList): [中文] GCN 卷积层列表 / [English] GCN conv layer list.
        norms (nn.ModuleList): [中文] LayerNorm 列表 / [English] LayerNorm list.
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
            num_layers (int): [中文] GCN 层数 / [English] number of GCN layers. Defaults to 3.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.3.
            use_residual (bool): [中文] 残差开关 / [English] enable residual. Defaults to True.
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
        前向传播：多层 GCN + 残差 / Forward pass: multi-layer GCN + residuals.

        Args / 参数:
            x (torch.Tensor): [中文] 节点特征 `(Total_Nodes, in_channels)` / [English] node features.
            edge_index (torch.Tensor): [中文] PyG 边索引 `(2, E)` / [English] PyG edge indices.

        Returns / 返回:
            torch.Tensor: [中文] 输出节点特征 `(Total_Nodes, out_channels)` / [English] output features.
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


class ClassQueryHead(nn.Module):
    """
    基于类查询的交叉注意力分类头 / Class-Query Cross-Attention classification head.

    维护 `num_classes` 个可学习的 `class_queries`，通过 `TransformerDecoder` (或纯 `MultiheadAttention`)
    与 GCN 节点特征做交叉注意力，输出 12 类 logits。可通过 `prune_heads()` 剪枝到子类集。
    Maintains `num_classes` learnable `class_queries` and uses `TransformerDecoder` (or pure
    `MultiheadAttention`) to perform cross-attention with GCN node features, producing 12-class
    logits. Can be pruned via `prune_heads()`.

    Attributes / 属性:
        class_queries (nn.Parameter): [中文] 可学习类查询 `(num_classes, hidden_dim)` / [English] class queries.
        cross_attention (nn.Module): [中文] TransformerDecoder 或 MHA / [English] cross-attention module.
        output_proj (nn.Sequential): [中文] 输出投影头 / [English] output projection.
        use_decoder (bool): [中文] 是否使用 TransformerDecoder / [English] whether to use TransformerDecoder.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_decoder: bool = True,
        seq_len: int = 1001
    ):
        """
        初始化 ClassQueryHead / Initialize ClassQueryHead.

        Args / 参数:
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim. Defaults to 128.
            num_classes (int): [中文] 类别数 / [English] number of classes. Defaults to 12.
            num_heads (int): [中文] 注意力头数 / [English] number of MHA heads. Defaults to 4.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
            use_decoder (bool): [中文] 是否使用 TransformerDecoder / [English] use TransformerDecoder. Defaults to True.
            seq_len (int): [中文] 序列长度 / [English] sequence length. Defaults to 1001.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_decoder = use_decoder
        self.seq_len = seq_len

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
        物理剪枝类查询 / Physically prune the class queries to keep only valid indices.

        用于少样本迁移场景：从 12 类剪枝到子类集。
        Used in few-shot transfer to reduce from 12 classes to a subset.

        Args / 参数:
            valid_class_indices (List[int]): [中文] 要保留的类索引列表 / [English] list of class indices to keep.
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
        """
        前向传播：类查询 ↔ 节点特征交叉注意力 / Forward: class queries ↔ node features cross-attention.

        Args / 参数:
            node_features (torch.Tensor): [中文] GCN 节点特征 `(Total_Nodes, hidden_dim)` / [English] GCN node features.
            batch (torch.Tensor): [中文] PyG 批索引 `(Total_Nodes,)` / [English] PyG batch index.

        Returns / 返回:
            torch.Tensor: [中文] 12 类 logits `(B, 12)` / [English] 12-class logits.
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        # Calculate actual max_nodes from batch
        num_nodes_per_sample = batch.bincount()
        max_nodes = num_nodes_per_sample[0].item()  # Assume all samples have same length

        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)

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
    简化版类查询池化头（返回注意力权重） / Simplified Class-Query head with attention pooling (returns weights).

    使用缩放点积注意力计算类查询与节点特征的关联，输出 logits 同时返回注意力权重供可视化。
    Uses scaled dot-product attention to compute class-to-node associations. Returns both logits
    and attention weights for downstream visualization.

    Attributes / 属性:
        class_queries (nn.Parameter): [中文] 类查询 `(num_classes, hidden_dim)` / [English] class queries.
        attention_scale (float): [中文] 缩放因子 sqrt(hidden_dim) / [English] attention scaling factor.
        output_proj (nn.Sequential): [中文] 输出投影 / [English] output projection.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        dropout: float = 0.1
    ):
        """
        初始化 ClassQueryHeadPooling / Initialize ClassQueryHeadPooling.

        Args / 参数:
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim. Defaults to 128.
            num_classes (int): [中文] 类别数 / [English] number of classes. Defaults to 12.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
        """
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
        """
        物理剪枝类查询 / Physically prune class queries.

        Args / 参数:
            valid_class_indices (List[int]): [中文] 要保留的类索引 / [English] class indices to keep.
        """
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
        前向传播（返回注意力权重） / Forward pass returning attention weights for supervision.

        Args / 参数:
            node_features (torch.Tensor): [中文] GCN 节点特征 / [English] GCN node features.
            batch (torch.Tensor): [中文] PyG 批索引 / [English] PyG batch index.

        Returns / 返回:
            Tuple[torch.Tensor, torch.Tensor]: [中文] `(logits (B,12), attn_weights (B,12,L))` / [English] logits and attention weights.
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


class HierarchicalClassQueryHeadPooling(nn.Module):
    """
    层级类查询头（4 组 → 12 类）/ Hierarchical class-query head (4 groups → 12 classes).

    MoHE 模块的核心实现：维护 4 个可学习组查询 (A/C/G/U)，通过组级投影器派生 12 个子类查询，
    再用 MultiheadAttention 做交叉注意力。同时输出 12 类和 4 组 logits，4 组作为辅助监督信号。
    Core implementation of MoHE: maintains 4 learnable group queries (A/C/G/U), derives 12
    subclass queries via group-level projectors, then performs cross-attention with MHA.
    Outputs both 12-class and 4-group logits; the 4-group is used as auxiliary supervision.

    Attributes / 属性:
        num_groups (int): [中文] 组数 (固定 4) / [English] number of groups (fixed 4).
        group_queries (nn.Parameter): [中文] 组查询 `(4, hidden_dim)` / [English] group queries.
        group_projectors (nn.ModuleList): [中文] 组→子类的投影器 / [English] group-to-subclass projectors.
        mha_12 (nn.MultiheadAttention): [中文] 12 类注意力 / [English] 12-class MHA.
        mha_4 (nn.MultiheadAttention): [中文] 4 组注意力 / [English] 4-group MHA.
    """

    def __init__(self, hidden_dim, num_classes, group_to_class_indices, dropout=0.1, use_layer_norm=True, num_heads=8, seq_len=1001):
        """
        初始化 HierarchicalClassQueryHeadPooling / Initialize HierarchicalClassQueryHeadPooling.

        Args / 参数:
            hidden_dim (int): [中文] 隐藏维度 / [English] hidden dim.
            num_classes (int): [中文] 类别数 / [English] number of classes. Defaults to 12.
            group_to_class_indices (Dict[str, List[int]]): [中文] 组到子类索引映射 / [English] group-to-class mapping.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
            use_layer_norm (bool): [中文] 输出投影是否用 LayerNorm / [English] use LayerNorm in output proj. Defaults to True.
            num_heads (int): [中文] MHA 头数 / [English] number of MHA heads. Defaults to 8.
            seq_len (int): [中文] 序列长度 / [English] sequence length. Defaults to 1001.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices
        self.num_groups = 4 # A, C, G, U
        self.num_heads = num_heads
        self.seq_len = seq_len  # Sequence length for dense batch processing

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
        通过索引掩码剪枝到指定的类和组 / Prune to only compute specific classes and groups via index masking.

        Args / 参数:
            valid_class_indices (List[int]): [中文] 要保留的类索引 / [English] class indices to keep.
            valid_group_indices (List[int]): [中文] 要保留的组索引 / [English] group indices to keep.
        """
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))
        print(f"Hierarchical Head Pruned: Active Classes={valid_class_indices}, Active Groups={valid_group_indices}")

    def _derive_class_queries(self):
        """
        从组查询派生 12 个子类查询 / Derive 12 class queries from group queries using projectors.

        Returns / 返回:
            torch.Tensor: [中文] 类查询 `(num_classes, hidden_dim)` / [English] class queries tensor.
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
        前向传播：层级类查询交叉注意力 / Forward: hierarchical class-query cross-attention.

        Args / 参数:
            node_features (torch.Tensor): [中文] GCN 节点特征 `(Total_Nodes, dim)` / [English] GCN node features.
            batch (torch.Tensor): [中文] PyG 批索引 / [English] PyG batch index.

        Returns / 返回:
            Tuple: [中文] `(logits_12, logits_4, attn_weights_12)` / [English] three tensors.
                - logits_12: `(B, 12)` 12 类 logits
                - logits_4: `(B, 4)` 4 组 logits
                - attn_weights_12: `(B, 12, L)` 12 类注意力
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        # Calculate actual sequence length from batch
        num_nodes_per_sample = batch.bincount()
        actual_seq_len = num_nodes_per_sample[0].item()  # Assume all samples have same length

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
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

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
    mRModN 端到端模型（多尺度 CNN + GCN + 类查询分类头）/ mRModN end-to-end model.

    整合 ParallelCNNBlock (M2D) + GCNBlock + ClassQueryHead，可选使用层级头 (MoHE) 或
    简化池化头。序列长度默认 1001nt (Human full-length)。
    Integrates ParallelCNNBlock (M2D) + GCNBlock + ClassQueryHead, optionally with
    hierarchical head (MoHE) or simplified pooling head. Default sequence length 1001nt.

    Attributes / 属性:
        cnn_block (ParallelCNNBlock): [中文] M2D 多尺度 CNN / [English] M2D multi-scale CNN.
        gcn_block (GCNBlock): [中文] GCN 图传播 / [English] GCN graph propagation.
        class_query_head (nn.Module): [中文] 选定的分类头 / [English] selected classification head.
        use_hierarchical (bool): [中文] 是否使用层级头 / [English] use hierarchical head.
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
        use_layer_norm: bool = True,
        seq_len: int = 1001
    ):
        super().__init__()

        self.cnn_out_channels = cnn_hidden_dim  # CNN output is now cnn_hidden_dim, not len(kernel_sizes) * cnn_hidden_dim
        self.gcn_out_channels = gcn_out_channels
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical
        self.seq_len = seq_len

        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout,
            seq_len=seq_len
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
                dropout=attn_dropout,
                seq_len=seq_len
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
                use_decoder=True,
                seq_len=seq_len
            )

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        """
        公开接口：剪枝分类头到指定的类/组 / Public interface to prune the classification head.

        Args / 参数:
            valid_class_indices (List[int]): [中文] 要保留的类索引 / [English] class indices to keep.
            valid_group_indices (Optional[List[int]]): [中文] 层级头需要的组索引 / [English] group indices (required for hierarchical).
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
        前向传播（端到端 CNN+GCN+Head） / End-to-end forward pass.

        Args / 参数:
            x (torch.Tensor): [中文] 输入序列 / [English] input sequence.
            edge_index (torch.Tensor): [中文] PyG 边索引 / [English] PyG edge indices.
            batch (Optional[torch.Tensor]): [中文] PyG 批索引 / [English] PyG batch index.
            return_attention (bool): [中文] 是否返回注意力权重 / [English] whether to return attention weights.

        Returns / 返回:
            tuple: [中文] 视头部模式返回不同元组 / [English] depends on head mode.
        """
        if isinstance(x, Data) or isinstance(x, Batch):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
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

            if self.training:
                return logits_12class, logits_4class, attn_weights_12
            else:
                return logits_12class, logits_4class, attn_weights_12

        elif self.use_simple_pooling:
            logits, attn_weights = self.class_query_head(node_features, batch)
            if self.training:
                return logits, attn_weights
            return logits
        else:
            logits = self.class_query_head(node_features, batch)
            # For compatibility, we assume non-pooling heads don't return attention in this setup
            if self.training:
                return logits, None 
            return logits
