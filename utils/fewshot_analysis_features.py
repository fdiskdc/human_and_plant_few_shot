"""
utils/fewshot_analysis_features.py - 特征提取器 / Feature Extractor

FeatureExtractor 类用 hooks 提取模型中间层特征;load_model_from_checkpoint 加载模型;extract_dataset_features 提取数据集特征。
FeatureExtractor class uses hooks to extract intermediate layer features; load_model_from_checkpoint loads model; extract_dataset_features extracts features.

功能模块 / Modules:
- FeatureExtractor: 用 hooks 提取特征 / Extract features via hooks
- load_model_from_checkpoint: 加载模型 / Load model
- extract_dataset_features: 提取数据集特征 / Extract dataset features

输入 / Inputs:
- model: nn.Module 模型 / nn.Module model
- data: 数据加载器 / Data loader
- checkpoint 路径 / Checkpoint path

输出 / Outputs:
- features: numpy 数组 / numpy array
- labels: numpy 数组 / numpy array

数据流 / Data Flow:
1. 加载模型 / Load model
2. 注册 hook / Register hook
3. 前向 + 收集特征 / Forward + collect features
4. 返回特征 / Return features

相关文件 / Related Files:
- 调用 / Calls: torch, torch_geometric
- 被调用 / Called by: zero_shot_fewshot_*.py, utils.fewshot_analysis_*

使用示例 / Usage Example:
    extractor = FeatureExtractor(model, layer_name='attention_output')
    features = extractor.extract(dataloader)

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from model.main_model import RNA_ClassQuery_Model


class FeatureExtractor:
    """
    Extract graph-level or class-aware features from intermediate layers.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.features = {}
        self.hooks = []
        self.attention_features = None

    def register_hooks(self):
        def tensor_hook(name):
            def _hook(_module, _inputs, output):
                self.features[name] = output.detach()
            return _hook

        def attention_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                self.attention_features = output[0].detach()
            else:
                self.attention_features = output.detach()

        self.hooks.append(self.model.cnn_block.register_forward_hook(tensor_hook('cnn_output')))
        self.hooks.append(self.model.gcn_block.register_forward_hook(tensor_hook('gcn_output')))

        if hasattr(self.model.class_query_head, 'mha_12'):
            self.hooks.append(
                self.model.class_query_head.mha_12.register_forward_hook(attention_hook)
            )
        elif hasattr(self.model.class_query_head, 'cross_attention'):
            self.hooks.append(
                self.model.class_query_head.cross_attention.register_forward_hook(attention_hook)
            )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, data_loader, layer_name='gcn_output'):
        self.model.eval()
        self.register_hooks()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {layer_name}", leave=False):
                batch = batch.to(self.device)
                self.features = {}
                self.attention_features = None

                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                if isinstance(outputs, tuple):
                    logits_12 = outputs[0]
                else:
                    logits_12 = outputs

                if layer_name == 'attention_output':
                    if self.attention_features is None:
                        raise RuntimeError("Attention features were not captured by hooks.")
                    graph_features = self.attention_features.mean(dim=1)
                else:
                    if layer_name not in self.features:
                        raise RuntimeError(f"Layer '{layer_name}' was not captured by hooks.")
                    node_features = self.features[layer_name]
                    batch_size = int(batch.batch.max().item()) + 1
                    pooled = []
                    for b in range(batch_size):
                        mask = batch.batch == b
                        pooled.append(node_features[mask].mean(dim=0))
                    graph_features = torch.stack(pooled, dim=0)

                all_features.append(graph_features.cpu().numpy())
                all_labels.append(batch.y.detach().cpu().numpy())

                # Touch logits to keep forward contract explicit and validated.
                _ = logits_12

        self.remove_hooks()
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', {}).get('model', {})

    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=model_cfg.get('cnn_hidden_dim', 128),
        cnn_kernel_sizes=tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7])),
        cnn_dropout=model_cfg.get('cnn_dropout', 0.1),
        gcn_hidden_dim=model_cfg.get('gcn_hidden_dim', 256),
        gcn_out_channels=model_cfg.get('gcn_out_channels', 256),
        gcn_num_layers=model_cfg.get('gcn_num_layers', 3),
        gcn_dropout=model_cfg.get('gcn_dropout', 0.3),
        num_classes=model_cfg.get('num_classes', 12),
        num_attn_heads=model_cfg.get('num_attn_heads', 8),
        attn_dropout=model_cfg.get('attn_dropout', 0.1),
        use_simple_pooling=model_cfg.get('use_simple_pooling', False),
        use_hierarchical=model_cfg.get('use_hierarchical', True),
        use_layer_norm=model_cfg.get('use_layer_norm', True),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def extract_dataset_features(model, dataset, indices, device, batch_size=64, layer_name='gcn_output'):
    """Extract features from a dataset using the model."""
    loader = PyGDataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    extractor = FeatureExtractor(model, device)
    return extractor.extract_features(loader, layer_name=layer_name)
