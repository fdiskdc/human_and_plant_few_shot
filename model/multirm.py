
"""
multirm.py - MultIRM 基线 BiLSTM+Bahdanau 注意力模型 (51nt窗口) / MultIRM baseline BiLSTM+Bahdanau attention model (51nt window)

实现 RNA 多标签修饰分类的基线模型族 (NaiveNet, NaiveNet_v1, NaiveNet_v2, model_v3)，
组合 1D CNN、BiLSTM 和 Bahdanau 注意力。model_v3 是与 train.py 接口兼容的主基线，
支持 12 类多标签输出和可选的 4 类层级分组。
Contains several baseline variants for multi-label RNA modification classification,
combining 1D CNN, BiLSTM, and Bahdanau attention. The model_v3 class is the primary
baseline that matches the train.py interface and supports 12-class multi-label output
with optional 4-class hierarchical grouping.

功能模块 / Modules:
- NaiveNet: 纯 CNN 基线 (无 LSTM) / Pure CNN baseline (no LSTM)
- NaiveNet_v1: CNN + BiLSTM + Bahdanau Attention 基线 / CNN + BiLSTM + Bahdanau attention baseline
- NaiveNet_v2: CNN + BiLSTM + Flatten+FC 基线 / CNN + BiLSTM + Flatten+FC baseline
- BahdanauAttention: 经典 Bahdanau 加性注意力 (h*W + context) / Classical Bahdanau additive attention
- model_v3: 与 train.py 兼容的 BiLSTM + Bahdanau 主基线 / train.py-compatible BiLSTM + Bahdanau primary baseline

输入 / Inputs:
- x: (B, 4, 51) 或 (Total_Nodes, 4) one-hot RNA子序列 (51nt) / (B, 4, 51) or (Total_Nodes, 4) one-hot RNA subsequence (51nt)
- edge_index: (2, E) PyG 边索引 (此模型未使用，保留接口兼容) / (2, E) PyG edge indices (unused, kept for interface)
- batch: (Total_Nodes,) 批次分配向量 / (Total_Nodes,) batch assignment vector
- return_attention: bool, 是否返回注意力权重 / bool, whether to return attention weights

输出 / Outputs:
- 默认 / Default: logits [B, num_task] / logits [B, num_task]
- return_attention=True: (logits [B,12], attn_weights [B,12,51]) 或 (logits_12, logits_4, attn_weights) 层级模式 / (logits [B,12], attn_weights [B,12,51]) or (logits_12, logits_4, attn_weights) hierarchical mode
- return_attention=False 层级: (logits_12, logits_4, None) / (logits_12, logits_4, None)

数据流 / Data Flow:
1. one-hot 子序列通过 3 层 1D CNN 提取局部特征 (51->36->18 等) / one-hot subsequence passes through 3-layer 1D CNN for local features
2. CNN 特征 reshape 为 (B, Seq, Feat) 后送入 BiLSTM / CNN features reshaped to (B, Seq, Feat) and fed to BiLSTM
3. 最后隐藏态 h_n 与 BiLSTM 输出经 Bahdanau 注意力汇聚为上下文向量 / Last hidden state h_n and BiLSTM output aggregated by Bahdanau attention into context vector
4. 上下文向量经每类独立的 FC 头输出 12 个二分类 logits / Context vector through per-class FC heads outputs 12 binary logits

相关文件 / Related Files:
- 调用 / Calls: torch, torch.nn, numpy / torch, torch.nn, numpy
- 被调用 / Called by: train_human_multirm.py, train_multirm_dataset.py, test_multirm_4class.py, test_multirm_oversampling.py, inference_multirm_segmented.py, collect_multirm_atten.py / train_human_multirm.py, train_multirm_dataset.py, test_multirm_4class.py, test_multirm_oversampling.py, inference_multirm_segmented.py, collect_multirm_atten.py

使用示例 / Usage Example:
    from model.multirm import model_v3
    model = model_v3(num_task=12, use_hierarchical=True)
    logits, logits_4, attn = model(x, edge_index, batch, return_attention=True)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import torch
import numpy as np
from torch import nn

# from util_layers import *


class NaiveNet(nn.Module):
    """
        CNN only
    """
    def __init__(self,input_size=None,num_task=None):
        self.num_task = num_task
        super(NaiveNet, self).__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0), #[bs, 8, 72]
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 72]
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 36]
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 36]
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0) #[bs 128 18]
                        )
        # self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1
        in_features_2 = (in_features_1 - 2) // 2 + 1
        in_features_3 = (in_features_2 - 2) // 2 + 1
        self.Flatten = nn.Flatten()
        self.SharedFC = nn.Sequential(nn.Linear(in_features=128*in_features_3,out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout()
                                    )
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=1024,out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                       nn.Linear(in_features=256,out_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=64,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):
        x = self.NaiveCNN(x)
        output = self.Flatten(x) # flatten output
        shared_layer = self.SharedFC(output)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(shared_layer)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class NaiveNet_v1(nn.Module):
    """
        CNN + LSTM + Attention
    """
    def __init__(self,input_size=None,num_task=None):
        self.num_task = num_task
        super(NaiveNet_v1, self).__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=1),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=1)
                        )
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=256,out_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=64,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1]) # pareprae input for Attention
        context_vector,attention_weights = self.Attention(h_n,output) # Attention (batch_size, num_task, unit)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(context_vector[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs

class NaiveNet_v2(nn.Module):
    """
        CNN + LSTM
    """
    def __init__(self,input_size=None,num_task=None):
        self.num_task = num_task
        super(NaiveNet_v2, self).__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0)
                        )
        in_features_1 = (input_size - 7) // 2 + 1
        in_features_2 = (in_features_1 - 2) // 2 + 1
        in_features_3 = (in_features_2 - 2) // 2 + 1
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        self.Flatten = nn.Flatten()
        self.SharedFC = nn.Sequential(nn.Linear(in_features=in_features_3*256,out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout()
                                    )
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=1024,out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                       nn.Linear(in_features=256,out_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=64,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        output = self.Flatten(output) # flatten output
        shared_layer = self.SharedFC(output)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(shared_layer)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        return outs


class BahdanauAttention(nn.Module):
    """Minimal Bahdanau Attention implementation for model_v3 compatibility."""
    def __init__(self, in_features, hidden_units, num_task):
        super().__init__()
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.num_task = num_task

        # Score network: Wa * tanh(Ua * h + Wa * context)
        self.score_layer = nn.Sequential(
            nn.Linear(in_features * 2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, h_n, output):
        """
        Args:
            h_n: Hidden state, shape (batch_size, in_features)
            output: LSTM output, shape (batch_size, seq_len, in_features)
        Returns:
            context_vector: Context vector, shape (batch_size, num_task, in_features)
            attention_weights: Attention weights, shape (batch_size, num_task, seq_len)
        """
        batch_size, seq_len, in_features = output.shape

        # Expand h_n to match each time step
        # h_n: (batch_size, in_features) -> (batch_size, seq_len, in_features)
        h_n_expanded = h_n.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate h_n with each output time step
        # concat: (batch_size, seq_len, in_features * 2)
        concat = torch.cat([h_n_expanded, output], dim=-1)

        # Compute attention scores
        # scores: (batch_size, seq_len, 1)
        scores = self.score_layer(concat)

        # Apply softmax to get attention weights
        # attention_weights: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)

        # Compute context vector as weighted sum of output
        # context: (batch_size, in_features)
        context = torch.sum(attention_weights * output, dim=1)

        # Expand context for num_task (replicate for each task)
        # context_vector: (batch_size, num_task, in_features)
        context_vector = context.unsqueeze(1).expand(-1, self.num_task, -1)

        # Expand attention weights for num_task
        # attention_weights: (batch_size, num_task, seq_len)
        attention_weights = attention_weights.squeeze(-1).unsqueeze(1).expand(-1, self.num_task, -1)

        return context_vector, attention_weights


class model_v3(nn.Module):
    """
    Baseline model using BiLSTM + Bahdanau Attention.
    Modified for compatibility with train.py.
    """

    def __init__(self, num_task=12, use_embedding=False, cnn_hidden_dim=64,
                 cnn_kernel_sizes=(1, 3, 5, 7), gcn_hidden_dim=128,
                 gcn_out_channels=128, gcn_num_layers=3, gcn_dropout=0.3,
                 num_classes=12, num_attn_heads=4, attn_dropout=0.1,
                 use_simple_pooling=False, use_hierarchical=False,
                 use_layer_norm=True, cnn_dropout=0.1):
        """
        Args:
            num_task: Number of tasks/classes (default 12 for compatibility)
            use_embedding: Whether to use embedding (False for this baseline)
            Additional params for train.py compatibility (ignored)
        """
        super(model_v3, self).__init__()

        self.num_task = num_task
        self.use_embedding = use_embedding
        self.use_hierarchical = use_hierarchical

        # LSTM hidden size (bidirectional will double this)
        self.lstm_hidden_size = 128

        # Store for compatibility with train.py
        # The actual output size is lstm_hidden_size * 2 for bidirectional
        self.gcn_out_channels = self.lstm_hidden_size * 2  # 256

        if self.use_embedding:
            # Note: EmbeddingSeq not available in this codebase
            # This branch is kept for compatibility but will raise error if used
            raise NotImplementedError("Embedding mode not supported - use_embedding=False required")
        else:
            # BiLSTM for sequence modeling
            # Input: (batch_size, seq_len, 4) after transpose
            # Output: (batch_size, seq_len, lstm_hidden_size * 2) for bidirectional
            self.NaiveBiLSTM = nn.LSTM(
                input_size=4,
                hidden_size=self.lstm_hidden_size,
                batch_first=True,
                bidirectional=True
            )

        # Bahdanau Attention mechanism
        # in_features must match LSTM output size: lstm_hidden_size * 2
        self.Attention = BahdanauAttention(
            in_features=self.lstm_hidden_size * 2,
            hidden_units=100,
            num_task=num_task
        )

        # Task-specific FC layers (one per class)
        # in_features must match LSTM output size: lstm_hidden_size * 2
        for i in range(num_task):
            setattr(self, "NaiveFC%d" % i, nn.Sequential(
                nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=128, out_features=1)
            ))

    def forward(self, x, edge_index=None, batch=None, return_attention=False):
        """
        Forward pass compatible with train.py

        Args:
            x: Input tensor
               - Shape: (Total_Nodes, 4) for PyG format
               - Shape: (Batch, 1001, 4) for tensor format
            edge_index: Edge indices (unused in this baseline, kept for compatibility)
            batch: Batch assignment vector (used for reshaping)
            return_attention: Whether to return attention weights (for attention supervision)

        Returns:
            If return_attention=True: (logits, attention_weights) or (logits_12class, logits_4class, attention_weights)
            If return_attention=False: logits or (logits_12class, logits_4class)
        """
        # Handle different input formats
        # LSTM with batch_first=True expects: (Batch, Seq_Len, Input_Size) = (Batch, 1001, 4)
        if x.dim() == 2:
            # PyG format: (Total_Nodes, 4)
            if batch is None:
                # Single sample: (Total_Nodes, 4) -> (1, Total_Nodes, 4)
                x = x.unsqueeze(0)
                batch_size = 1
            else:
                # Batch: (Total_Nodes, 4) -> (Batch, 1001, 4)
                batch_size = batch.max().item() + 1
                x = x.view(batch_size, 1001, 4)
            # Now x is (Batch, 1001, 4) - correct for LSTM
        elif x.dim() == 3:
            # Could be either (Batch, 1001, 4) or (Batch, 4, 1001)
            if x.size(1) == 4:
                # (Batch, 4, 1001) - need to transpose to (Batch, 1001, 4)
                batch_size = x.size(0)
                x = x.transpose(1, 2)
            else:
                # Already (Batch, 1001, 4) - correct for LSTM
                batch_size = x.size(0)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # BiLSTM forward
        # output: (Batch, Seq_Len, 512) where 512 = 2 * 256 (bidirectional)
        # h_n: (2, Batch, 256) -> need to reshape
        output, (h_n, c_n) = self.NaiveBiLSTM(x)

        # Reshape h_n: (2, Batch, 256) -> (Batch, 512)
        # Concatenate forward and backward hidden states
        h_n = h_n.transpose(0, 1).contiguous().view(batch_size, -1)

        # Apply attention
        # context_vector: (Batch, num_task, 512)
        # attention_weights: (Batch, num_task, Seq_Len)
        context_vector, attention_weights = self.Attention(h_n, output)

        # Apply task-specific FC layers
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" % i)
            y = FClayer(context_vector[:, i, :])  # (Batch, 1)
            outs.append(y)

        # Stack outputs: list of (Batch, 1) -> (Batch, num_task)
        logits = torch.cat(outs, dim=1)

        # Handle attention return
        if return_attention:
            if self.use_hierarchical:
                # For hierarchical, we need both 12-class and 4-class logits
                # Simple approach: derive 4-class from 12-class by grouping
                # Group indices: A=[0,1,7,9,10], C=[2,6,8], G=[3,11], U=[4,5]
                group_indices = [[0, 1, 7, 9, 10], [2, 6, 8], [3, 11], [4, 5]]

                logits_4class = torch.zeros(batch_size, 4, device=x.device)
                for i, indices in enumerate(group_indices):
                    # Max pooling over grouped classes
                    logits_4class[:, i] = logits[:, indices].max(dim=1)[0]

                return logits, logits_4class, attention_weights
            else:
                return logits, attention_weights
        else:
            # Handle hierarchical output if requested (without attention)
            if self.use_hierarchical:
                # For hierarchical, we need both 12-class and 4-class logits
                # Simple approach: derive 4-class from 12-class by grouping
                # Group indices: A=[0,1,7,9,10], C=[2,6,8], G=[3,11], U=[4,5]
                group_indices = [[0, 1, 7, 9, 10], [2, 6, 8], [3, 11], [4, 5]]

                logits_4class = torch.zeros(batch_size, 4, device=x.device)
                for i, indices in enumerate(group_indices):
                    # Max pooling over grouped classes
                    logits_4class[:, i] = logits[:, indices].max(dim=1)[0]

                # Return 3 values to match original model's interface in test_epoch
                # The third value is a placeholder for attention weights
                return logits, logits_4class, None

            return logits
