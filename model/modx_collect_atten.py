
"""
modx_collect_atten.py - modX 模型 收集注意力输出变体 (1001nt) / modX variant that collects attention outputs (1001nt)

modx.py 的修改版，保留 RNAClassifierWithWord2Vec 模型结构，但前向同时返回
context_vector 和 (广播到 12 类的) attention_weights，用于可视化与下游分析。
类末尾保留 alias `RNAClassifierWithWord2Vec = RNAClassifierWithWord2Vec_Collect_Atten`
以兼容原模型检查点加载。
A modified version of modx.py that keeps the same RNAClassifierWithWord2Vec structure
but additionally returns the context_vector and (broadcast-to-12-classes) attention_weights
in the forward. An alias `RNAClassifierWithWord2Vec = RNAClassifierWithWord2Vec_Collect_Atten`
is registered at the end for checkpoint-loading compatibility.

功能模块 / Modules:
- BahdanauAttention: 单头加性注意力 / Single-head additive attention
- RNAClassifierWithWord2Vec_Collect_Atten: input_proj + BiLSTM + Bahdanau + FC，return_attention=True 时额外返回 context_vector / input_proj + BiLSTM + Bahdanau + FC, additionally returns context_vector when return_attention=True
- RNAClassifierWithWord2Vec (alias): 兼容原模型权重加载的别名 / Alias for original model weight loading

输入 / Inputs:
- x: (B*1001, 4) 或 (B, 1001, 4) one-hot RNA 序列 / (B*1001, 4) or (B, 1001, 4) one-hot RNA sequence
- edge_index: (2, E) PyG 边索引 (未使用) / (2, E) PyG edge indices (unused)
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector
- return_attention: bool 是否返回 context_vector 和 attention_weights / bool whether to return context_vector and attention_weights

输出 / Outputs:
- 默认 / Default: logits [B, 12] / logits [B, 12]
- return_attention=True: (logits [B,12], context_vector [B, 2*hidden_dim], attn_weights [B,12,1001]) / (logits [B,12], context_vector [B, 2*hidden_dim], attn_weights [B,12,1001])

数据流 / Data Flow:
1. one-hot 序列经 input_proj 映射到 embedding 空间 / one-hot projected to embedding via input_proj
2. BiLSTM 编码为序列特征 / BiLSTM encodes into sequence features
3. Bahdanau 注意力汇聚为 context_vector (2*hidden_dim) / Bahdanau attention aggregates to context_vector (2*hidden_dim)
4. FC 输出 12 类 logits；return_attention=True 时同时返回 context_vector 和广播的 attn_weights / FC outputs 12-class logits; additionally returns context_vector and broadcasted attn_weights

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, torch.nn.functional / torch, torch.nn, torch.nn.functional
- 被调用 / Called by: collect_modx_atten.py, inference_modx_segmented.py (checkpoint 加载) / collect_modx_atten.py, inference_modx_segmented.py (checkpoint loading)

使用示例 / Usage Example:
    from model.modx_collect_atten import RNAClassifierWithWord2Vec_Collect_Atten
    model = RNAClassifierWithWord2Vec_Collect_Atten(input_dim=4, hidden_dim=128, output_dim=12)
    logits, ctx, attn = model(x, edge_index, batch, return_attention=True)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        hidden = hidden.unsqueeze(1)
        # hidden expanded shape: (batch_size, seq_len, hidden_dim)
        hidden = hidden.expand(-1, encoder_outputs.size(1), -1)

        # Calculate alignment scores
        alignment_scores = self.v(torch.tanh(self.W(hidden + encoder_outputs)))

        # Softmax to get attention weights
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(alignment_scores, dim=1)

        # Calculate context vector
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)

        return context_vector, attention_weights

class RNAClassifierWithWord2Vec_Collect_Atten(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=64, hidden_dim=128, num_layers=2, output_dim=12, dropout=0.5):
        super(RNAClassifierWithWord2Vec_Collect_Atten, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = 1001  # Fixed sequence length

        # Linear projection layer for one-hot encoded input
        # Maps 4-dimensional one-hot vectors to embedding_dim
        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Bi-Directional LSTM Model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Attention Layer - Input dimension is 2 * hidden_dim due to BiLSTM
        self.attention = BahdanauAttention(2 * hidden_dim)

        # Fully Connected Layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        # Drop out layer for Overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, batch=None, return_attention=False):
        """
        Forward pass for RNAClassifierWithWord2Vec_Collect_Atten.

        Args:
            x: Input features - can be PyG Data object or tensor
               If Data: uses x attribute (node features)
               If tensor: shape (batch_size * seq_len, input_dim)
            edge_index: Graph edge indices (unused in this model, kept for compatibility)
            batch: Batch assignment vector (used for splitting)
            return_attention: Whether to return context_vector and attention_weights

        Returns:
            If return_attention is False: logits (batch_size, output_dim)
            If return_attention is True: (logits, context_vector, attention_weights)
                - context_vector: (batch_size, 2 * hidden_dim)
                - attention_weights: (batch_size, output_dim, seq_len)
        """
        # Handle PyG Data object input
        if hasattr(x, 'x'):  # PyG Data object
            batch_obj = x
            x = batch_obj.x
            batch = batch_obj.batch

        # x shape: (batch_size * seq_len, input_dim)
        # Reshape to (batch_size, seq_len, input_dim)
        if batch is not None:
            batch_size = batch.max().item() + 1
        else:
            # Infer batch size from total nodes and seq_len
            batch_size = x.size(0) // self.seq_len

        # Reshape to (batch_size, seq_len, input_dim)
        x = x.view(batch_size, self.seq_len, self.input_dim)

        # Project input to embedding space
        x = self.input_proj(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM forward pass
        lstm_out, (h, c) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, 2 * hidden_dim)
        # h shape: (num_layers * 2, batch_size, hidden_dim)

        # Concatenate the hidden states from the last layer of both directions
        # The last forward and backward hidden states are h[-2] and h[-1]
        last_h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        # last_h shape: (batch_size, 2 * hidden_dim)

        # Calculate Context Vector and Attention Weights
        # Use last_h as the query (hidden state) and lstm_out as the encoder outputs
        context_vector, attention_weights = self.attention(last_h, lstm_out)
        # context_vector shape: (batch_size, 2 * hidden_dim)
        # attention_weights shape: (batch_size, seq_len, 1)

        # Pass context vector through dropout layer
        context_vector = self.dropout(context_vector)

        # Pass context vector to the fully connected layer
        out = self.fc(context_vector)
        # out shape: (batch_size, output_dim)

        if return_attention:
            # Return logits, context_vector, and attention_weights
            # Expand attention weights to match original model format for compatibility
            # Original format: [batch_size, num_classes, seq_len]
            # Current format: [batch_size, seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
            # Expand to [batch_size, output_dim, seq_len] - same attention for all classes
            attn_expanded = attention_weights.unsqueeze(1).expand(-1, self.output_dim, -1)
            return out, context_vector, attn_expanded
        else:
            return out

# Alias for compatibility with original model name when loading checkpoints
RNAClassifierWithWord2Vec = RNAClassifierWithWord2Vec_Collect_Atten
