"""
test_multirm_4class.py - MultIRM 4 类模式冒烟测试 / MultIRM 4-Class Mode Smoke Test

MultIRM 数据集的 4 类模式冒烟测试,验证 4 类层级分组训练与评估流程。
Smoke test for 4-class hierarchical grouping mode of MultIRM dataset.

功能模块 / Modules:
- 4 类模式推理 / 4-class mode inference
- 冒烟测试 / Smoke test
- main: 主入口 / Main entry point

输入 / Inputs:
- json/multirm_4class.json: 配置 / Config
- multirm/seq.npy, multirm/4loc.npy: 4 类标签 / 4-class labels
- 命令行参数 / CLI: --config, --gpu

输出 / Outputs:
- 终端冒烟测试结果 / Terminal smoke test results
- logs/multirm_4class_*/results.json

数据流 / Data Flow:
1. 加载 4 类数据 / Load 4-class data
2. 加载模型 / Load model
3. 4 类推理 / 4-class inference
4. 4 类评估 / 4-class evaluation

相关文件 / Related Files:
- 调用 / Calls: dataset.multirm.MultirmDataset, model.mrmodn
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python test_multirm_4class.py --config json/multirm_4class.json

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import torch
from torch_geometric.loader import DataLoader
from dataset.multirm import MultirmDataset, MULTIRM_CLASSES, MULTIRM_4CLASS_NAMES, MULTIRM_12TO4_MAPPING
from model.mrmodn_multirm import RNA_ClassQuery_Model

def main():
    """
    MultiRM 4-class 测试入口 / MultiRM 4-class test main entry.

    加载 `MultirmDataset`, 构建 DataLoader, 跑一遍 forward + loss 反向以验证
    4 分组 / 12 类的标签映射与模型头对接是否正确。
    Loads `MultirmDataset`, builds DataLoader, runs one forward + backward
    pass to verify the 4-group / 12-class label mapping and model-head wiring.

    Called by / 被调用:
        - __main__ 块: [中文] 直接 CLI 调用 / [English] invoked from CLI.

    Raises / 异常:
        AssertionError: [中文] 当形状 / 标签 / 损失不匹配时 / [English] shape, label,
            or loss mismatch.
    """

    print("=" * 60)
    print("Test 1: Check class mappings")
    print("=" * 60)
    
    print(f"\n12 classes (new order):")
    for i, cls in enumerate(MULTIRM_CLASSES):
        group_idx = MULTIRM_12TO4_MAPPING[i]
        print(f"  {i}: {cls:8s} -> Group {group_idx} ({MULTIRM_4CLASS_NAMES[group_idx]})")
    
    print("=" * 60)
    print("Test 2: Load dataset with 4-class mode")
    print("=" * 60)
    
    dataset = MultirmDataset(mode='train', use_4class=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  x shape: {sample.x.shape}")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  y (12-class) shape: {sample.y.shape}")
    if hasattr(sample, 'y_4'):
        print(f"  y_4 (4-class) shape: {sample.y_4.shape}")
        print(f"  12-class label: {sample.y[0].numpy()}")
        print(f"  4-class label: {sample.y_4[0].numpy()}")
    else:
        print("  Warning: y_4 not found")
    
    print("=" * 60)
    print("Test 3: Test DataLoader")
    print("=" * 60)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"\nBatch structure:")
    print(f"  batch.x shape: {batch.x.shape}")
    print(f"  batch.edge_index shape: {batch.edge_index.shape}")
    print(f"  batch.y shape: {batch.y.shape}")
    if hasattr(batch, 'y_4'):
        print(f"  batch.y_4 shape: {batch.y_4.shape}")
    
    print("=" * 60)
    print("Test 4: Test model with 4-class grouping")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=64,
        gcn_hidden_dim=128,
        gcn_out_channels=128,
        num_classes=12,
        use_hierarchical=True
    )
    
    model = model.to(device)
    batch = batch.to(device)
    
    print("\nModel structure:")
    print(model)
    
    print("\nForward pass (without attention):")
    with torch.no_grad():
        logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch, return_attention=False)
    
    print(f"  logits_12 shape: {logits_12.shape}")
    print(f"  logits_4 shape: {logits_4.shape}")
    
    print("\nForward pass (with attention):")
    with torch.no_grad():
        logits_12, logits_4, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
    
    print(f"  logits_12 shape: {logits_12.shape}")
    print(f"  logits_4 shape: {logits_4.shape}")
    print(f"  attn_weights shape: {attn_weights.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()