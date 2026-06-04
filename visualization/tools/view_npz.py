"""
view_npz.py - 检查 npy/npz 缓存文件工具 / Inspect npy/npz Cache Files Utility

检查 npy/cache/zero_batch_cache.npz 等缓存文件,验证 edge_indices、node features 等数据完整性。
Inspects npy/npz cache files (e.g., zero_batch_cache.npz), validating edge_indices and node features integrity.

功能模块 / Modules:
- inspect_npz: 检查 npz 文件 / Inspect npz file
- 节点索引验证 (0-1000) / Node index validation (0-1000)
- edge_indices 分析 / edge_indices analysis
- main: 主入口 / Main entry point

输入 / Inputs:
- npy/cache/zero_batch_cache.npz: NumPy 压缩格式 / NumPy compressed format
- 路径配置: FILE_PATH, MAX_ALLOWED_INDEX / Path config

输出 / Outputs:
- 终端详细报告 / Detailed terminal report
- 错误诊断 (节点索引超界) / Error diagnostics (out-of-bounds indices)

数据流 / Data Flow:
1. 加载 npz 文件 / Load npz file
2. 验证 keys / Validate keys
3. 检查 edge_indices 形状 / Check edge_indices shape
4. 报告节点索引范围 / Report node index range

相关文件 / Related Files:
- 调用 / Calls: numpy
- 被调用 / Called by: manual execution for cache inspection

使用示例 / Usage Example:
    python view_npz.py

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

import numpy as np
import os
import sys

# 设定目标文件路径
FILE_PATH = 'npy/cache/zero_batch_cache.npz'
# 设定最大允许的节点索引 (RNA长度为1001，索引应为 0-1000)
MAX_ALLOWED_INDEX = 1000 

def inspect_npz(path):
    print(f"{'='*60}")
    print(f"正在检查文件: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"❌ 错误: 文件不存在 -> {path}")
        return

    try:
        # allow_pickle=True 是必须的，因为 edge_indices 是 object 数组
        with np.load(path, allow_pickle=True) as data:
            print(f"✅ 文件加载成功!")
            print(f"包含的 Keys: {list(data.keys())}")
            
            # 1. 检查 edge_indices
            if 'edge_indices' in data:
                edge_indices = data['edge_indices']
                total_samples = len(edge_indices)
                print(f"\n[edge_indices 分析]")
                print(f"  - 数组形状: {edge_indices.shape}")
                print(f"  - 数据类型: {edge_indices.dtype}")
                print(f"  - 样本总数: {total_samples}")

                # --- 深度扫描 ---
                print(f"\n>>> 正在进行深度扫描 (查找 None 和 越界索引)...")
                
                none_count = 0
                out_of_bounds_count = 0
                empty_graph_count = 0
                max_idx_found = -1
                bad_indices_examples = []

                # 遍历所有样本
                for i in range(total_samples):
                    item = edge_indices[i]
                    
                    # 检查 None
                    if item is None:
                        none_count += 1
                        if none_count <= 5: print(f"  ⚠️ 发现 None 值，索引: {i}")
                        continue
                    
                    # 检查是否为 Numpy 数组
                    if not isinstance(item, np.ndarray):
                        print(f"  ⚠️ 索引 {i} 类型异常: {type(item)}")
                        continue

                    # 检查空图
                    if item.size == 0:
                        empty_graph_count += 1
                        continue
                    
                    # 检查最大索引 (CUDA 报错的元凶)
                    current_max = item.max()
                    if current_max > max_idx_found:
                        max_idx_found = current_max
                    
                    if current_max > MAX_ALLOWED_INDEX:
                        out_of_bounds_count += 1
                        if len(bad_indices_examples) < 5:
                            bad_indices_examples.append((i, current_max))

                # --- 扫描报告 ---
                print(f"\n[扫描结果报告]")
                
                # 报告 1: None 值 (卡顿原因)
                if none_count > 0:
                    print(f"❌ 严重警告: 发现 {none_count} 个 None 值！")
                    print(f"   -> 这会导致训练时触发 CPU 兜底计算，造成严重卡顿。")
                else:
                    print(f"✅ 检查通过: 没有发现 None 值。")

                # 报告 2: 越界索引 (CUDA 报错原因)
                if out_of_bounds_count > 0:
                    print(f"❌ 严重警告: 发现 {out_of_bounds_count} 个样本包含越界节点索引！")
                    print(f"   -> 最大允许索引: {MAX_ALLOWED_INDEX}")
                    print(f"   -> 实际发现最大值: {max_idx_found}")
                    print(f"   -> 错误样本示例 (索引, 该样本最大节点): {bad_indices_examples}")
                    print(f"   -> 这会导致 'scatter gather kernel index out of bounds' 错误。")
                else:
                    print(f"✅ 检查通过: 所有节点索引均在合法范围内 (<= {MAX_ALLOWED_INDEX})。")

                print(f"\n  - 空图数量 (无边): {empty_graph_count}")

            # 2. 检查其他 Key (如 num_samples)
            if 'num_samples' in data:
                print(f"\n[Metadata]")
                print(f"  - num_samples (记录值): {data['num_samples']}")

    except Exception as e:
        print(f"\n❌ 读取文件时发生异常: {e}")

if __name__ == "__main__":
    inspect_npz(FILE_PATH)