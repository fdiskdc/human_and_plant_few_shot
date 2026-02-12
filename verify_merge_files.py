"""
Verify the generated merge files in split directory
"""
import numpy as np
import os

merge_dir = "npy/multirm/51split/split"
prefixes = ["train", "test", "valid"]

print("=" * 60)
print("验证合并文件")
print("=" * 60)

for prefix in prefixes:
    print(f"\n{prefix}:")
    
    in_path = os.path.join(merge_dir, f"{prefix}_in.npy")
    out_12_path = os.path.join(merge_dir, f"{prefix}_out_12class.npy")
    out_4_path = os.path.join(merge_dir, f"{prefix}_out_4class.npy")
    
    if not all(os.path.exists(p) for p in [in_path, out_12_path, out_4_path]):
        print(f"  缺少文件!")
        continue
    
    data_in = np.load(in_path, allow_pickle=True)
    data_out_12 = np.load(out_12_path, allow_pickle=True)
    data_out_4 = np.load(out_4_path, allow_pickle=True)
    
    print(f"  序列形状: {data_in.shape}")
    print(f"  12类标签形状: {data_out_12.shape}")
    print(f"  4类标签形状: {data_out_4.shape}")
    
    # 检查序列长度是否为51
    if data_in.shape[1] == 51:
        print(f"  ✓ 序列长度正确: 51")
    else:
        print(f"  ✗ 序列长度错误: {data_in.shape[1]} (应为51)")
    
    # 检查样本数量一致
    if len(data_in) == len(data_out_12) == len(data_out_4):
        print(f"  ✓ 样本数量一致: {len(data_in)}")
    else:
        print(f"  ✗ 样本数量不一致!")
    
    # 检查前几个样本的4类标签
    print(f"  前3个样本:")
    for i in range(min(3, len(data_in))):
        pos_classes_12 = np.where(data_out_12[i] == 1)[0]
        pos_class_4 = np.argmax(data_out_4[i])
        
        if len(pos_classes_12) > 0:
            class_names = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Psi','AtoI']
            pos_class_names = [class_names[c] for c in pos_classes_12]
            group_names = ['A', 'C', 'G', 'U']
            print(f"    样本{i}: 12类={pos_class_names} -> 4类={group_names[pos_class_4]}")
        else:
            print(f"    样本{i}: 负样本 -> 4类={data_out_4[i]}")

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)