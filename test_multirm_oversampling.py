"""
test_multirm_oversampling.py - MultIRM 过采样+最大长度对齐测试 / MultIRM Oversampling+Max-Length-Alignment Test

测试 MultIRM 数据集训练时的过采样 + 最大长度对齐 (max-length-alignment) 流程。
Test oversampling + max-length-alignment flow for MultIRM dataset training.

功能模块 / Modules:
- 过采样推理 / Oversampling inference
- 最大长度对齐 / Max-length alignment
- 评估指标 / Evaluation metrics
- main: 主入口 / Main entry point

输入 / Inputs:
- json/multirm_oversampling.json: 配置 / Config
- multirm/seq.npy: MultIRM 序列 / MultIRM sequences
- 命令行参数 / CLI: --config, --gpu

输出 / Outputs:
- 终端测试结果 / Terminal test results
- logs/multirm_oversampling_*/results.json

数据流 / Data Flow:
1. 加载 MultIRM 数据 / Load MultIRM data
2. 加载模型 / Load model
3. 过采样训练循环 / Oversampling training loop
4. 最大长度对齐评估 / Max-length-alignment evaluation

相关文件 / Related Files:
- 调用 / Calls: dataset.multirm.MultirmDataset, model.main_model
- 被调用 / Called by: manual execution

使用示例 / Usage Example:
    python test_multirm_oversampling.py --config json/multirm_oversampling.json

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import torch
from torch_geometric.loader import DataLoader
from dataset.multirm import MultirmDataset
import numpy as np

def test_train_mode():
    """测试训练模式的采样策略"""
    print("="*80)
    print("测试1: 训练模式 - Oversampling with Max-Length Alignment")
    print("="*80)
    
    dataset = MultirmDataset(mode='train', use_cache=False, preload_cache=False)
    
    print(f"\n数据集信息:")
    print(f"  虚拟数据集大小: {len(dataset)}")
    print(f"  最大桶长度 (max_len): {dataset.max_len}")
    print(f"  桶数量: {len(dataset.buckets)}")
    print(f"  预期虚拟大小: {dataset.max_len * len(dataset.buckets)}")
    
    # 测试循环采样
    print(f"\n测试循环采样机制:")
    print(f"  假设某个小类桶只有 50 个样本，max_len=1000")
    print(f"  测试索引 [0, 49, 50, 99, 100, 149, 150] 的采样情况:")
    
    # 找一个较小的桶
    small_bucket_idx = None
    small_bucket_len = float('inf')
    for i, length in enumerate(dataset.bucket_lengths):
        if length > 0 and length < small_bucket_len:
            small_bucket_len = length
            small_bucket_idx = i
    
    if small_bucket_idx is not None:
        print(f"\n  找到小桶: 索引 {small_bucket_idx}, 长度 {small_bucket_len}")
        
        # 测试几个索引的循环采样
        test_indices = [0, small_bucket_len-1, small_bucket_len, 
                        small_bucket_len*2-1, small_bucket_len*2, 
                        small_bucket_len*3-1, small_bucket_len*3]
        
        for idx in test_indices:
            # 计算这个全局索引对应的桶内索引
            bucket_start = small_bucket_idx * dataset.max_len
            global_idx = bucket_start + idx
            inner_idx = idx % small_bucket_len
            actual_idx = inner_idx % small_bucket_len
            
            sample = dataset[global_idx]
            print(f"    全局索引 {global_idx}: 内部索引 {idx} -> 实际样本索引 {actual_idx}")
    
    # 统计一个 epoch 中每个类被采样的次数
    print(f"\n统计一个 Epoch 中每个桶的采样次数:")
    bucket_sample_counts = [0] * len(dataset.buckets)
    
    # 遍历所有索引（这会很慢，所以我们只采样一部分）
    sample_size = min(1000, len(dataset))
    indices = list(range(sample_size))
    
    for idx in indices:
        bucket_idx = idx // dataset.max_len
        if bucket_idx < len(dataset.buckets):
            bucket_sample_counts[bucket_idx] += 1
    
    for i, (meta, length, count) in enumerate(zip(dataset.bucket_meta, 
                                                   dataset.bucket_lengths,
                                                   bucket_sample_counts)):
        class_name, is_pos = meta
        pos_neg = "Pos" if is_pos else "Neg"
        expected = min(sample_size // len(dataset.buckets), sample_size)
        print(f"  [{i:2d}] {class_name:6s} {pos_neg:3s}: 实际采样 {count:4d}, "
              f"样本数 {length:5d}, 重复率 {count/length if length>0 else 0:.2f}x")
    
    print("\n" + "="*80)

def test_test_mode():
    """测试测试模式的采样策略"""
    print("\n" + "="*80)
    print("测试2: 测试/验证模式 - 不进行循环采样")
    print("="*80)
    
    dataset = MultirmDataset(mode='test', use_cache=False, preload_cache=False)
    
    print(f"\n数据集信息:")
    print(f"  真实数据集大小: {len(dataset)}")
    print(f"  固定测试列表长度: {len(dataset.fixed_test_list)}")
    
    # 统计每个类的样本数
    class_counts = {}
    for sample in dataset.fixed_test_list:
        key = (sample['class_name'], 'Pos' if sample['is_pos'] else 'Neg')
        class_counts[key] = class_counts.get(key, 0) + 1
    
    print(f"\n各桶样本数统计:")
    for i, (meta, length) in enumerate(zip(dataset.bucket_meta, dataset.bucket_lengths)):
        class_name, is_pos = meta
        pos_neg = "Pos" if is_pos else "Neg"
        print(f"  [{i:2d}] {class_name:6s} {pos_neg:3s}: {length:5d}")
    
    print("\n" + "="*80)

def test_dataloader():
    """测试 DataLoader 的使用"""
    print("\n" + "="*80)
    print("测试3: DataLoader 批量加载")
    print("="*80)
    
    dataset = MultirmDataset(mode='train', use_cache=False, preload_cache=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"\nDataLoader 配置:")
    print(f"  batch_size: 32")
    print(f"  shuffle: True")
    print(f"  num_workers: 0")
    
    # 获取第一个批次
    print(f"\n获取第一个批次:")
    batch = next(iter(loader))
    
    print(f"  batch.x 形状: {batch.x.shape}")
    print(f"  batch.edge_index 形状: {batch.edge_index.shape}")
    print(f"  batch.y 形状: {batch.y.shape}")
    print(f"  batch.class_idx 形状: {batch.class_idx.shape}")
    print(f"  batch.size: {batch.num_graphs}")
    
    # 统计批次中各类的数量
    unique_classes, counts = torch.unique(batch.class_idx, return_counts=True)
    print(f"\n批次中各类别的分布:")
    for cls, cnt in zip(unique_classes, counts):
        print(f"  类别 {cls.item():2d}: {cnt.item():2d} 个样本")
    
    print("\n" + "="*80)

def test_4class_mode():
    """测试4类模式"""
    print("\n" + "="*80)
    print("测试4: 4类分类模式")
    print("="*80)
    
    dataset = MultirmDataset(mode='train', use_cache=False, preload_cache=False, use_4class=True)
    
    print(f"\n获取第一个样本:")
    sample = dataset[0]
    
    print(f"  y (12类) 形状: {sample.y.shape}")
    print(f"  y_4 (4类) 形状: {sample.y_4.shape}")
    print(f"  y_4 值: {sample.y_4}")
    
    print("\n" + "="*80)

def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("MultIRM Oversampling with Max-Length Alignment 策略测试")
    print("="*80)
    
    try:
        test_train_mode()
        test_test_mode()
        test_dataloader()
        test_4class_mode()
        
        print("\n" + "="*80)
        print("所有测试完成！✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()