"""
AC4C数据集使用示例

本文件展示如何使用AC4CDataset类加载和访问AC4C RNA修饰数据集。
"""

from torch_geometric.loader import DataLoader
from dataset.ac4c import AC4CDataset


def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("\n" + "=" * 60)
    print("示例1: 基本使用方法")
    print("=" * 60)
    
    # 加载balanced版本的训练集（默认）
    train_dataset = AC4CDataset(
        mode='train',
        data_dir='npy/ac4c_processed/balanced_ac4c'
    )
    
    # 加载balanced版本的测试集
    test_dataset = AC4CDataset(
        mode='test',
        data_dir='npy/ac4c_processed/balanced_ac4c'
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 获取单个样本
    sample = train_dataset[0]
    print(f"\n样本信息:")
    print(f"  节点特征形状: {sample.x.shape}")
    print(f"  边索引形状: {sample.edge_index.shape}")
    print(f"  标签 (12类): {sample.y}")
    print(f"  标签 (4类): {sample.y_4class}")


def example_2_unbalanced_version():
    """示例2: 使用unbalanced版本"""
    print("\n" + "=" * 60)
    print("示例2: 使用unbalanced版本")
    print("=" * 60)
    
    # 加载unbalanced版本
    train_dataset = AC4CDataset(
        mode='train',
        data_dir='npy/ac4c_processed/unbalanced_ac4c'
    )
    
    print(f"Unbalanced训练集大小: {len(train_dataset)}")


def example_3_dataloader():
    """示例3: 使用PyG DataLoader进行批处理"""
    print("\n" + "=" * 60)
    print("示例3: 使用DataLoader进行批处理")
    print("=" * 60)
    
    # 创建数据集
    dataset = AC4CDataset(mode='train')
    
    # 创建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # 设置为0进行单进程调试
    )
    
    print(f"DataLoader创建成功")
    print(f"  批次大小: 32")
    print(f"  总批次数: {len(loader)}")
    
    # 获取一个批次
    batch = next(iter(loader))
    print(f"\n批次信息:")
    print(f"  batch.x形状: {batch.x.shape}")
    print(f"  batch.edge_index形状: {batch.edge_index.shape}")
    print(f"  batch.y形状: {batch.y.shape}")


def example_4_few_shot_split():
    """示例4: Few-shot学习数据划分"""
    print("\n" + "=" * 60)
    print("示例4: Few-shot学习数据划分")
    print("=" * 60)
    
    dataset = AC4CDataset(mode='train')
    
    # 5-shot学习：从10%的池中采样每类5个样本
    support_indices, test_indices = dataset.get_few_shot_split(
        k_shots=5,
        valid_classes=[5, 8, 9],  # 指定类别
        test_ratio=0.9,
        seed=42
    )
    
    print(f"Few-shot划分结果:")
    print(f"  支持集大小: {len(support_indices)}")
    print(f"  测试集大小: {len(test_indices)}")
    print(f"  支持集索引 (前10个): {support_indices[:10]}")


def example_5_cache_precomputation():
    """示例5: 预计算二级结构并缓存"""
    print("\n" + "=" * 60)
    print("示例5: 预计算二级结构并缓存")
    print("=" * 60)
    
    dataset = AC4CDataset(mode='train')
    
    # 获取缓存统计
    stats = dataset.get_cache_stats()
    print(f"当前缓存状态:")
    print(f"  批量缓存存在: {stats['batch_cache']['exists']}")
    print(f"  单文件缓存数量: {stats['single_file_cache']['total_files']}")
    
    print("\n注意: 预计算二级结构需要较长时间")
    print("如需预计算，请取消以下代码的注释:")
    print("# dataset.precompute_all_structures(batch_size=100, num_workers=4)")


def example_6_cache_management():
    """示例6: 缓存管理"""
    print("\n" + "=" * 60)
    print("示例6: 缓存管理")
    print("=" * 60)
    
    dataset = AC4CDataset(mode='train')
    
    # 查看缓存统计
    stats = dataset.get_cache_stats()
    print(f"缓存统计:")
    if stats['batch_cache']['exists']:
        cache = stats['batch_cache']
        print(f"  批量缓存:")
        print(f"    路径: {cache['path']}")
        print(f"    大小: {cache['size_mb']:.2f} MB")
        print(f"    已加载到内存: {cache['loaded_in_memory']}")
    
    print(f"  单文件缓存:")
    print(f"    文件数量: {stats['single_file_cache']['total_files']}")
    print(f"    总大小: {stats['single_file_cache']['total_size_mb']:.2f} MB")
    
    print("\n注意: 清除缓存需要取消以下代码的注释:")
    print("# dataset.clear_cache()")


if __name__ == "__main__":
    # 运行所有示例
    example_1_basic_usage()
    example_2_unbalanced_version()
    example_3_dataloader()
    example_4_few_shot_split()
    example_5_cache_precomputation()
    example_6_cache_management()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)