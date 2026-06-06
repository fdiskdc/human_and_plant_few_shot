
"""
evormd_human.py - EvoRMD 风格的轻量 CNN+注意力 MIL 分类模型 / EvoRMD-style lightweight CNN+attention MIL classification model

EvoRMD 流水线的人类多标签修饰适配版本。Conv1dEmbedder 替代原 RNA-FM 主干，
TrainableAttention 实现多示例学习 (MIL) 池化，MulticlassClassifier 用 MLP 头输出
12 类 logits。前向接口与 model_v3 完全兼容，可直接接入 train.py / test.py。
A human multi-label adaptation of the EvoRMD pipeline. Conv1dEmbedder replaces the
original RNA-FM backbone, TrainableAttention provides multiple-instance learning (MIL)
pooling, and MulticlassClassifier is an MLP head producing 12-class logits. The forward
interface is fully compatible with model_v3 and plugs into train.py / test.py.

功能模块 / Modules:
- Conv1dEmbedder: 双层 1D 卷积 + LayerNorm，将 4 维 one-hot 投影到 d_fm 维 token 嵌入 / Two-layer 1D conv + LayerNorm, maps 4-dim one-hot to d_fm-dim token embeddings
- TrainableAttention: 线性层生成 token 注意力权重 (MIL 池化) / Linear layer producing token attention weights (MIL pooling)
- MulticlassClassifier: 1/2/3 层 MLP 分类头 (depth 可配) / 1/2/3-layer MLP classification head (configurable depth)
- EvoRMDForHuman: 端到端 EvoRMD-style 模型，forward 兼容 model_v3 接口 / End-to-end EvoRMD-style model with model_v3-compatible forward

输入 / Inputs:
- x: (Total_Nodes, 4) 或 (B, 1001, 4) one-hot RNA 序列 (1001nt) / (Total_Nodes, 4) or (B, 1001, 4) one-hot RNA sequence (1001nt)
- edge_index: (2, E) PyG 边索引 (此模型不使用) / (2, E) PyG edge indices (unused)
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector
- return_attention: bool 是否返回注意力权重 (广播到 num_task) / bool, return attention weights (broadcast to num_task)

输出 / Outputs:
- 默认 / Default: logits [B, num_task] / logits [B, num_task]
- 层级模式 / Hierarchical: (logits_12 [B,12], logits_4 [B,4]) 派生自 4 组 max-pool / (logits_12 [B,12], logits_4 [B,4]) derived via 4-group max-pool
- return_attention=True: (logits, attn_per_task [B,num_task,1001]) 或 5 元组 / (logits, attn_per_task [B,num_task,1001]) or 5-tuple

数据流 / Data Flow:
1. (B, 1001, 4) one-hot 通过 Conv1dEmbedder 编码为 (B, 1001, d_fm) token 嵌入 / one-hot encoded to (B, 1001, d_fm) tokens
2. TrainableAttention 输出 (B, 1001) 注意力权重，加权求和得到 (B, d_fm) 池化向量 / TrainableAttention produces (B, 1001) weights, weighted sum yields (B, d_fm) pool
3. MulticlassClassifier (MLP) 输出 (B, num_task) 12 类 logits / MulticlassClassifier (MLP) outputs (B, num_task) 12-class logits
4. 若 hierarchical=True，则通过 4 组 max-pool 派生 (B, 4) 组级 logits / If hierarchical=True, derive (B, 4) group-level logits via 4-group max-pool

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch.nn.functional / torch, torch.nn, torch.nn.functional
- 被调用 / Called by: train_human_evormd.py, inference_evormd_segmented.py / train_human_evormd.py, inference_evormd_segmented.py

使用示例 / Usage Example:
    from model.evormd_human import EvoRMDForHuman
    model = EvoRMDForHuman(num_task=12, d_fm=640, mlp_depth=2, use_hierarchical=True)
    logits_12, logits_4, attn = model(x, edge_index, batch, return_attention=True)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dEmbedder(nn.Module):
    """
    双层 Conv1d 嵌入器（替换 RNA-FM backbone） / Two-layer Conv1d embedder replacing RNA-FM.

    将 4 维 one-hot 投影到 d_fm 维 token embedding。
    Projects 4-dim one-hot RNA sequences into d_fm-dim token-level embeddings.

    Attributes / 属性:
        conv1, conv2 (nn.Conv1d): [中文] 两层 1D 卷积 / [English] two 1D conv layers.
        ln (nn.LayerNorm): [中文] 最后一层归一化 / [English] final layer norm.
    """

    def __init__(self, in_channels=4, d_fm=640, kernel_size=7, dropout=0.1):
        """
        初始化 Conv1dEmbedder / Initialize Conv1dEmbedder.

        Args / 参数:
            in_channels (int): [中文] 输入通道 / [English] input channels. Defaults to 4.
            d_fm (int): [中文] 嵌入维度 / [English] embedding dim. Defaults to 640.
            kernel_size (int): [中文] 卷积核 / [English] kernel size. Defaults to 7.
            dropout (float): [中文] dropout 比率 / [English] dropout rate. Defaults to 0.1.
        """
        super().__init__()
        padding = kernel_size // 2  # keep sequence length unchanged
        self.conv1 = nn.Conv1d(in_channels, d_fm, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_fm, d_fm, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_fm)

    def forward(self, x):
        """
        前向传播：4 维 one-hot → d_fm 维 token 嵌入 / Forward: 4-dim one-hot → d_fm-dim embeddings.

        Args / 参数:
            x (torch.Tensor): [中文] `(B, L, 4)` one-hot 序列 / [English] one-hot sequence.

        Returns / 返回:
            torch.Tensor: [中文] `(B, L, d_fm)` 嵌入 / [English] token embeddings.
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
    序列级可学习注意力 / Learnable sequence-level attention (from EvoRMD).

    Attributes / 属性:
        attention (nn.Linear): [中文] 标量得分投影 / [English] scalar score projection.
    """

    def __init__(self, embedding_dim):
        """
        初始化 TrainableAttention / Initialize TrainableAttention.

        Args / 参数:
            embedding_dim (int): [中文] 嵌入维度 / [English] embedding dim.
        """
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, token_embeddings):
        """
        前向传播：序列级 softmax 注意力 / Forward: sequence-level softmax attention.

        Args / 参数:
            token_embeddings (torch.Tensor): [中文] `(B, L, D)` / [English] token embeddings.

        Returns / 返回:
            torch.Tensor: [中文] `(B, L)` 注意力权重 / [English] attention weights.
        """
        # token_embeddings: (B, L, D)
        attention_scores = self.attention(token_embeddings).squeeze(-1)  # (B, L)
        attention_weights = F.softmax(attention_scores, dim=-1)          # (B, L)
        return attention_weights


class MulticlassClassifier(nn.Module):
    """
    融合嵌入的多类分类器（来自 EvoRMD） / Multi-class classifier on top of fused embedding.

    mlp_depth 控制 MLP 层数（1/2/3） / Controls MLP depth (1/2/3 layers).

    Attributes / 属性:
        mlp_depth (int): [中文] MLP 深度 / [English] MLP depth.
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
        """
        前向传播：MLP 分类 / Forward: MLP classification.

        Args / 参数:
            x (torch.Tensor): [中文] 输入嵌入 / [English] input embedding.

        Returns / 返回:
            torch.Tensor: [中文] 分类 logits / [English] classification logits.
        """
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
    EvoRMD 风格的人类多标签 RNA 分类模型 / EvoRMD-style model for human RNA classification.

    用 Conv1dEmbedder 替换原 RNA-FM backbone，保留 TrainableAttention + MulticlassClassifier。
    接口与 model_v3 兼容，可直接复用现有训练/测试工具。
    Replaces RNA-FM with Conv1dEmbedder, keeps TrainableAttention + MulticlassClassifier.
    Same `forward()` interface as `model_v3` for compatibility with existing utilities.

    Attributes / 属性:
        embedder (Conv1dEmbedder): [中文] Conv1d 嵌入器 / [English] Conv1d embedder.
        attention (TrainableAttention): [中文] 序列级注意力 / [English] sequence attention.
        classifier (MulticlassClassifier): [中文] MLP 分类头 / [English] MLP classifier.
        use_hierarchical (bool): [中文] 是否使用层级头 / [English] use hierarchical head.
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
        """
        初始化 EvoRMDForHuman / Initialize EvoRMDForHuman.

        Args / 参数:
            num_task (int): [中文] 任务数 / [English] number of tasks. Defaults to 12.
            d_fm (int): [中文] 嵌入维度 / [English] embedding dim. Defaults to 640.
            mlp_depth (int): [中文] MLP 深度 / [English] MLP depth. Defaults to 2.
            conv_kernel_size (int): [中文] 卷积核 / [English] conv kernel size. Defaults to 7.
            conv_dropout (float): [中文] dropout / [English] dropout. Defaults to 0.1.
            use_hierarchical (bool): [中文] 层级头开关 / [English] use hierarchical head. Defaults to False.
        """
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
        前向传播（与 train.py / test.py 兼容） / Forward pass compatible with train/test utilities.

        Args / 参数:
            x (torch.Tensor): [中文] 输入张量 / [English] input tensor.
                - PyG: `(Total_Nodes, 4)`
                - tensor: `(Batch, 1001, 4)`
            edge_index: [中文] 边索引 (未使用) / [English] unused, kept for interface.
            batch (Optional[torch.Tensor]): [中文] 批索引 / [English] batch index.
            return_attention (bool): [中文] 是否返回注意力 / [English] whether to return attention.

        Returns / 返回:
            tuple or torch.Tensor: [中文] 视模式返回 logits 或元组 / [English] depends on mode.
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
        # Expand shared attention (B, L) → per-task (B, num_task, L)
        # to match the format expected by compute_attention_supervision_loss
        attn_per_task = attn_weights.unsqueeze(1).expand(-1, self.num_task, -1)

        if return_attention:
            if self.use_hierarchical:
                logits_4 = self._derive_4class(logits, batch_size, x.device)
                return logits, logits_4, attn_per_task
            return logits, attn_per_task
        else:
            if self.use_hierarchical:
                logits_4 = self._derive_4class(logits, batch_size, x.device)
                return logits, logits_4, None
            return logits

    def _derive_4class(self, logits_12, batch_size, device):
        """
        从 12 类 logits 派生 4 组 logits（按核苷酸分组 max-pool） / Derive 4-class logits from 12-class by max-pooling over groups.

        Args / 参数:
            logits_12 (torch.Tensor): [中文] 12 类 logits / [English] 12-class logits.
            batch_size (int): [中文] 批大小 / [English] batch size.
            device: [中文] torch device / [English] torch device.

        Returns / 返回:
            torch.Tensor: [中文] 4 组 logits `(B, 4)` / [English] 4-group logits.
        """
        logits_4 = torch.zeros(batch_size, 4, device=device)
        for i, indices in enumerate(self.group_indices):
            logits_4[:, i] = logits_12[:, indices].max(dim=1)[0]
        return logits_4
