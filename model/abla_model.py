
"""
abla_model.py - 层级类查询头 (MOHE) 消融变体 / Hierarchical class-query head (MOHE) ablation variants

包含 4 个消融分类头变体 (1query, 4query, 12query, fullattn)，用于消融研究层级类查询
机制的贡献。AblationModel 是统一入口：根据 query_type 选择对应头，并支持 1query 退化
backbone。所有变体对外暴露与 RNA_ClassQuery_Model 兼容的 (logits_12, logits_4, attn) 接口。
Contains four ablation head variants (1query, 4query, 12query, fullattn) for studying the
contribution of the hierarchical class-query mechanism. AblationModel is a unified entry
point that selects the head by query_type and supports a degraded backbone for 1query.
All variants expose a (logits_12, logits_4, attn) interface compatible with RNA_ClassQuery_Model.

功能模块 / Modules:
- HierarchicalClassQueryHead1Query: 退化基线 — 无 query/attention/LN，仅均值池化 + Linear / Degraded baseline — no query/attention/LN, mean-pool + Linear only
- HierarchicalClassQueryHead4Query: 4 组层级查询 + 组投影派生 12 类 (基线复刻) / 4-group hierarchical queries + group projectors derive 12 classes (baseline replica)
- HierarchicalClassQueryHead12Query: 12 个独立可学习查询 + 12 个独立 MHA / 12 independent learnable queries + 12 independent MHA modules
- HierarchicalClassQueryHeadFullAttn: 无可学习查询，纯 TransformerEncoder 自注意力 + 均值池化 / No learnable queries, pure TransformerEncoder self-attention + mean-pool
- AblationModel: 统一消融模型，按 query_type 注册头 (支持 group_query_dim) / Unified ablation model, registers head by query_type (supports group_query_dim)

输入 / Inputs:
- x: (B, 1001, 4) 或 (Total_Nodes, 4) one-hot RNA 序列 / (B, 1001, 4) or (Total_Nodes, 4) one-hot RNA sequence
- edge_index: (2, E) PyG 边索引 / (2, E) PyG edge indices
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector
- query_type: str in {"1query","4query","12query","fullattn"} 选择头 / str in {"1query","4query","12query","fullattn"} to select head
- group_query_dim: int 可学习的 query 参数维度 (None=hidden_dim) / int dim of learnable query params (None=hidden_dim)

输出 / Outputs:
- logits_12: (B, 12) 12 类 logits (prune 后可更小) / (B, 12) 12-class logits (smaller after pruning)
- logits_4: (B, 4) 4 组 (A/C/G/U) logits (prune 后可更小) / (B, 4) 4-group (A/C/G/U) logits (smaller after pruning)
- attn_weights_12: (B, 12, 1001) 12 类对 1001nt 位置的注意力权重 / (B, 12, 1001) per-class attention weights over 1001nt positions

数据流 / Data Flow:
1. 序列经 ParallelCNNBlock (1query 退化用单核) 提取局部 k-mer 特征 / Sequence passes through ParallelCNNBlock (1query uses single kernel)
2. (1query 除外) 特征经 GCNBlock 图传播；1query 路径经 backbone_proj 映射到 head 维度 / (except 1query) Features propagated through GCNBlock; 1query path uses backbone_proj to head dim
3. 选定的分类头处理节点特征，生成 12 类 logits、4 组 logits、12 类注意力权重 / Selected head processes node features, produces 12-class logits, 4-group logits, 12-class attention weights

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch_geometric.data, torch_geometric.nn, utils.common.GROUP_TO_CLASS_INDICES, model.mrmodn.ParallelCNNBlock/GCNBlock / torch, torch.nn, torch_geometric.data, torch_geometric.nn, utils.common.GROUP_TO_CLASS_INDICES, model.mrmodn.ParallelCNNBlock/GCNBlock
- 被调用 / Called by: abla_mohe.py, cal_flops_mohe.py / abla_mohe.py, cal_flops_mohe.py

使用示例 / Usage Example:
    from model.abla_model import AblationModel
    model = AblationModel(query_type="4query", group_query_dim=128)
    logits_12, logits_4, attn = model(x, edge_index, batch)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""


import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple

from utils.common import GROUP_TO_CLASS_INDICES
from model.mrmodn import ParallelCNNBlock, GCNBlock


def _make_query_proj(query_dim, hidden_dim):
    """Create a projection layer if query_dim != hidden_dim, else identity."""
    if query_dim == hidden_dim:
        return nn.Identity()
    return nn.Linear(query_dim, hidden_dim)


# ============================================================================
# Head 1: Single Query + 12-way MLP
# ============================================================================

class HierarchicalClassQueryHead1Query(nn.Module):
    """
    1-query: maximally degraded baseline.
    - NO query, NO attention, NO LayerNorm, NO MLP.
    - Pure mean pooling over the sequence → single Linear classifier.
    - Should perform worst among all ablation variants.
    """

    def __init__(self, hidden_dim=128, num_classes=12,
                 group_to_class_indices=None, dropout=0.1,
                 use_layer_norm=True, num_heads=8, seq_len=1001,
                 group_query_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Single Linear: hidden_dim -> num_classes + 4 (12-class + 4-group)
        self.classifier_12 = nn.Linear(hidden_dim, num_classes)
        self.classifier_4 = nn.Linear(hidden_dim, 4)

    def prune_heads(self, valid_class_indices, valid_group_indices):
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))

    def forward(self, node_features, batch):
        batch_size = batch.max().item() + 1
        device = node_features.device
        actual_seq_len = batch.bincount()[0].item()
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

        # Mean pooling — no attention, just average
        pooled = dense_nodes.mean(dim=1)  # [Batch, Dim]

        logits_12 = self.classifier_12(pooled)
        logits_4 = self.classifier_4(pooled)

        # Uniform attention weights for visualization compatibility
        attn_weights_12 = torch.ones(batch_size, self.num_classes, actual_seq_len,
                                     device=device) / actual_seq_len

        if hasattr(self, 'valid_class_indices'):
            logits_12 = logits_12[:, self.valid_class_indices]
            attn_weights_12 = attn_weights_12[:, self.valid_class_indices]
        if hasattr(self, 'valid_group_indices'):
            logits_4 = logits_4[:, self.valid_group_indices]

        return logits_12.to(device), logits_4.to(device), attn_weights_12.to(device)


# ============================================================================
# Head 2: 4-Group Hierarchical (Baseline Replica)
# ============================================================================

class HierarchicalClassQueryHead4Query(nn.Module):
    """
    4-query hierarchical: faithful replica of the original HierarchicalClassQueryHeadPooling
    with an added group_query_dim parameter.  When group_query_dim != hidden_dim a linear
    projection maps queries into hidden_dim before the group projectors / MHA.
    """

    def __init__(self, hidden_dim=128, num_classes=12,
                 group_to_class_indices=None, dropout=0.1,
                 use_layer_norm=True, num_heads=8, seq_len=1001,
                 group_query_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices or GROUP_TO_CLASS_INDICES
        self.num_groups = 4
        self.num_heads = num_heads
        self.group_names = ['A', 'C', 'G', 'U']
        q_dim = group_query_dim or hidden_dim

        # Group queries in q_dim space
        self.group_queries = nn.Parameter(torch.randn(self.num_groups, q_dim))
        self.query_proj = _make_query_proj(q_dim, hidden_dim)

        # Group-wise projectors (input is always hidden_dim after projection)
        self.group_projectors = nn.ModuleList()
        for g_idx in range(self.num_groups):
            group_name = self.group_names[g_idx]
            num_sub = len(self.group_to_class_indices.get(group_name, []))
            self.group_projectors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_sub * hidden_dim)
            ))

        self.mha_12 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                             dropout=dropout, batch_first=True)
        self.mha_4 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                            dropout=dropout, batch_first=True)

        norm12 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.output_proj_12 = nn.Sequential(
            norm12, nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )
        norm4 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.output_proj_4 = nn.Sequential(
            norm4, nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )

    def prune_heads(self, valid_class_indices, valid_group_indices):
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))

    def _derive_class_queries(self):
        all_sub, all_idx = [], []
        for g_idx in range(self.num_groups):
            g_name = self.group_names[g_idx]
            if g_name not in self.group_to_class_indices:
                continue
            gq = self.query_proj(self.group_queries[g_idx].unsqueeze(0))  # [1, hidden_dim]
            sub_flat = self.group_projectors[g_idx](gq)
            n = len(self.group_to_class_indices[g_name])
            all_sub.append(sub_flat.view(n, self.hidden_dim))
            all_idx.extend(self.group_to_class_indices[g_name])

        flat = torch.cat(all_sub, dim=0)
        idx_t = torch.tensor(all_idx, device=flat.device)
        ordered = torch.zeros(self.num_classes, self.hidden_dim, dtype=flat.dtype, device=flat.device)
        ordered.index_copy_(0, idx_t, flat)
        return ordered

    def forward(self, node_features, batch):
        batch_size = batch.max().item() + 1
        device = node_features.device
        actual_seq_len = batch.bincount()[0].item()
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

        class_queries = self._derive_class_queries()
        group_queries = self.query_proj(self.group_queries)  # [4, hidden_dim]

        if hasattr(self, 'valid_class_indices'):
            class_queries = class_queries[self.valid_class_indices]
        if hasattr(self, 'valid_group_indices'):
            group_queries = group_queries[self.valid_group_indices]

        q12 = class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        q4 = group_queries.unsqueeze(0).expand(batch_size, -1, -1)

        attn_out_12, attn_weights_12 = self.mha_12(query=q12, key=dense_nodes, value=dense_nodes,
                                                     average_attn_weights=True)
        attn_out_4, _ = self.mha_4(query=q4, key=dense_nodes, value=dense_nodes,
                                    average_attn_weights=True)

        logits_12 = self.output_proj_12(attn_out_12).squeeze(-1)
        logits_4 = self.output_proj_4(attn_out_4).squeeze(-1)

        return logits_12.to(device), logits_4.to(device), attn_weights_12.to(device)


# ============================================================================
# Head 3: 12 Independent Queries (No Group Projector)
# ============================================================================

class HierarchicalClassQueryHead12Query(nn.Module):
    """
    12-query: 12 independent learnable queries, each with its OWN MHA module.
    Each query independently cross-attends to the full sequence via a dedicated
    MultiheadAttention layer — no weight sharing between queries.
    This makes the head significantly heavier than 4-query.
    4-group logits are derived by mean-pooling class features per group.
    """

    def __init__(self, hidden_dim=128, num_classes=12,
                 group_to_class_indices=None, dropout=0.1,
                 use_layer_norm=True, num_heads=8, seq_len=1001,
                 group_query_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices or GROUP_TO_CLASS_INDICES
        self.group_names = ['A', 'C', 'G', 'U']
        q_dim = group_query_dim or hidden_dim

        # 12 independent learnable queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, q_dim))
        self.query_proj = _make_query_proj(q_dim, hidden_dim)

        # 12 independent MHA modules — one per class
        self.per_class_mha = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                  dropout=dropout, batch_first=True)
            for _ in range(num_classes)
        ])

        # 12 per-class output projections (each outputs 1 logit)
        self.per_class_proj = nn.ModuleList()
        for _ in range(num_classes):
            norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.per_class_proj.append(nn.Sequential(
                norm, nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
            ))

        # 4-group output projection
        norm4 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.output_proj_4 = nn.Sequential(
            norm4, nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )

    def prune_heads(self, valid_class_indices, valid_group_indices):
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))

    def forward(self, node_features, batch):
        batch_size = batch.max().item() + 1
        device = node_features.device
        actual_seq_len = batch.bincount()[0].item()
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

        cq = self.query_proj(self.class_queries)  # [12, hidden_dim]

        # Determine which class indices to process
        if hasattr(self, 'valid_class_indices'):
            active_indices = self.valid_class_indices.tolist()
        else:
            active_indices = list(range(self.num_classes))

        # Each query independently attends via its own MHA
        all_logits = []
        all_attn_out = []
        all_attn_weights = []
        for cls_idx in active_indices:
            # query: [Batch, 1, Dim]
            q = cq[cls_idx].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            attn_out, attn_w = self.per_class_mha[cls_idx](
                query=q, key=dense_nodes, value=dense_nodes,
                average_attn_weights=True
            )  # attn_out: [B, 1, D], attn_w: [B, 1, Seq]
            logit = self.per_class_proj[cls_idx](attn_out.squeeze(1))  # [B, 1]
            all_logits.append(logit)
            all_attn_out.append(attn_out.squeeze(1))  # [B, D]
            all_attn_weights.append(attn_w.squeeze(1))  # [B, Seq]

        # Stack: [Batch, Num_Active, 1] -> squeeze -> [Batch, Num_Active]
        logits_12 = torch.cat(all_logits, dim=1)          # [B, N_active]
        attn_out_all = torch.stack(all_attn_out, dim=1)   # [B, N_active, D]
        attn_weights_12 = torch.stack(all_attn_weights, dim=1)  # [B, N_active, Seq]

        # 4-group: mean-pool class features per group
        group_feats = []
        for g_name in self.group_names:
            indices = self.group_to_class_indices[g_name]
            if hasattr(self, 'valid_class_indices'):
                vl = self.valid_class_indices.tolist()
                li = [vl.index(i) for i in indices if i in vl]
            else:
                li = indices
            gf = attn_out_all[:, li, :].mean(dim=1, keepdim=True) if li else \
                torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            group_feats.append(gf)

        group_features = torch.cat(group_feats, dim=1)
        logits_4 = self.output_proj_4(group_features).squeeze(-1)

        if hasattr(self, 'valid_group_indices'):
            logits_4 = logits_4[:, self.valid_group_indices]

        return logits_12.to(device), logits_4.to(device), attn_weights_12.to(device)


# ============================================================================
# Head 4: Full Attention (12 Queries + Self-Attention)
# ============================================================================

class HierarchicalClassQueryHeadFullAttn(nn.Module):
    """
    fullattn: NO learnable queries. The RNA sequence features themselves serve
    as Q, K, V in a TransformerEncoder (pure self-attention). The output is
    mean-pooled to a single vector, then classified by two MLP heads.
    This tests whether the sequence can classify itself without external queries.
    """

    def __init__(self, hidden_dim=128, num_classes=12,
                 group_to_class_indices=None, dropout=0.1,
                 use_layer_norm=True, num_heads=8, seq_len=1001,
                 group_query_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.classifier_12 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes)
        )
        self.classifier_4 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 4)
        )

    def prune_heads(self, valid_class_indices, valid_group_indices):
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        self.register_buffer('valid_group_indices', torch.tensor(valid_group_indices, dtype=torch.long))

    def forward(self, node_features, batch):
        batch_size = batch.max().item() + 1
        device = node_features.device
        actual_seq_len = batch.bincount()[0].item()
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

        # Self-attention: Q=K=V=dense_nodes
        attn_out = self.encoder(dense_nodes)  # [B, Seq, Dim]

        # Mean pool over sequence
        pooled = attn_out.mean(dim=1)  # [B, Dim]

        logits_12 = self.classifier_12(pooled)
        logits_4 = self.classifier_4(pooled)

        # Approximate attention weights via dot-product for visualization
        attn_weights_12 = torch.softmax(
            torch.bmm(attn_out, dense_nodes.transpose(1, 2)), dim=-1
        )  # [B, Seq, Seq]

        # Average over query positions to get [B, 12, Seq] for compatibility
        attn_weights_12 = attn_weights_12.mean(dim=1, keepdim=True)  # [B, 1, Seq]
        attn_weights_12 = attn_weights_12.expand(-1, self.num_classes, -1)

        if hasattr(self, 'valid_class_indices'):
            logits_12 = logits_12[:, self.valid_class_indices]
            attn_weights_12 = attn_weights_12[:, self.valid_class_indices]
        if hasattr(self, 'valid_group_indices'):
            logits_4 = logits_4[:, self.valid_group_indices]

        return logits_12.to(device), logits_4.to(device), attn_weights_12.to(device)


# ============================================================================
# Unified Ablation Model
# ============================================================================

# Registry: query_type string -> head class
_HEAD_REGISTRY = {
    "1query":    HierarchicalClassQueryHead1Query,
    "4query":    HierarchicalClassQueryHead4Query,
    "12query":   HierarchicalClassQueryHead12Query,
    "fullattn":  HierarchicalClassQueryHeadFullAttn,
}


class AblationModel(nn.Module):
    """
    Drop-in replacement for RNA_ClassQuery_Model.

    Args:
        query_type:       one of "1query", "4query", "12query", "fullattn"
        group_query_dim:  dimensionality of the learnable query parameters
                          (projected to gcn_out_channels before MHA).
                          If None, defaults to gcn_out_channels.
    """

    QUERY_TYPES = list(_HEAD_REGISTRY.keys())

    def __init__(
        self,
        query_type: str = "4query",
        group_query_dim: int = None,
        cnn_hidden_dim: int = 64,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        cnn_dropout: float = 0.1,
        gcn_hidden_dim: int = 128,
        gcn_out_channels: int = 128,
        gcn_num_layers: int = 3,
        gcn_dropout: float = 0.3,
        num_classes: int = 12,
        num_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        use_layer_norm: bool = True,
        seq_len: int = 1001,
        # Legacy parameter kept for backward compat with abla_mohe.py
        ablation_type: str = None,
    ):
        super().__init__()

        # Accept either query_type or ablation_type
        if ablation_type is not None and query_type == "4query":
            query_type = ablation_type
        assert query_type in self.QUERY_TYPES, \
            f"Unknown query_type '{query_type}'. Choose from {self.QUERY_TYPES}"

        self.query_type = query_type
        self.group_query_dim = group_query_dim or gcn_out_channels
        self.num_classes = num_classes
        self.seq_len = seq_len

        # Backbone — reduced for 1query, full for others
        if query_type == "1query":
            # Reduced backbone: single CNN branch, no GCN
            self.cnn_block = ParallelCNNBlock(
                in_channels=4, hidden_dim=16,
                kernel_sizes=(5,),
                use_layer_norm=False, dropout=0.0,
                seq_len=seq_len
            )
            self.gcn_block = None
            # Projection: CNN output (16) -> head hidden_dim (128)
            self.backbone_proj = nn.Linear(16, gcn_out_channels)
        else:
            # Full backbone
            self.cnn_block = ParallelCNNBlock(
                in_channels=4, hidden_dim=cnn_hidden_dim,
                kernel_sizes=cnn_kernel_sizes,
                use_layer_norm=use_layer_norm, dropout=cnn_dropout,
                seq_len=seq_len
            )
            self.gcn_block = GCNBlock(
                in_channels=cnn_hidden_dim, hidden_dim=gcn_hidden_dim,
                out_channels=gcn_out_channels, num_layers=gcn_num_layers,
                dropout=gcn_dropout, use_residual=True
            )
            self.backbone_proj = None

        # Instantiate the selected head
        head_cls = _HEAD_REGISTRY[query_type]
        self.class_query_head = head_cls(
            hidden_dim=gcn_out_channels,
            num_classes=num_classes,
            group_to_class_indices=GROUP_TO_CLASS_INDICES,
            dropout=attn_dropout,
            use_layer_norm=use_layer_norm,
            num_heads=num_attn_heads,
            seq_len=seq_len,
            group_query_dim=self.group_query_dim,
        )

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        if hasattr(self.class_query_head, 'prune_heads'):
            if valid_group_indices is not None:
                self.class_query_head.prune_heads(valid_class_indices, valid_group_indices)
            else:
                self.class_query_head.prune_heads(valid_class_indices)

    def forward(self, x, edge_index, batch=None, return_attention=False):
        if isinstance(x, Data) or isinstance(x, Batch):
            batch_obj = x
            x = batch_obj.x
            edge_index = batch_obj.edge_index
            batch = batch_obj.batch

        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            if batch is None:
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(seq_len)
        elif x.dim() == 2:
            assert batch is not None

        node_features = self.cnn_block(x, batch)
        if self.gcn_block is not None:
            node_features = self.gcn_block(node_features, edge_index)
        if self.backbone_proj is not None:
            node_features = self.backbone_proj(node_features)

        logits_12, logits_4, attn_weights = self.class_query_head(node_features, batch)
        return logits_12, logits_4, attn_weights
