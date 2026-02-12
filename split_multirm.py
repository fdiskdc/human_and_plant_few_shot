'''
Author: Chao Deng && chaodeng987@outlook.com
Date: 2026-02-12 18:33:18
LastEditors: Chao Deng && chaodeng987@outlook.com
LastEditTime: 2026-02-12 22:58:53
FilePath: /rgcnformer_sum/split_multirm.py
Description: 
那只是一场游戏一场梦
 
https://orcid.org/0009-0009-8520-1656
DOI: 10.3390/app15158626
DOI: 10.3390/rs17142354
Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
'''
import numpy as np
import os

def split_dataset(crop_length=1001):
    """
    切分多修饰数据集，并可选择从中心剪裁序列
    
    Args:
        crop_length: 剪裁后的序列长度，默认为1001（不剪裁）
                    序列将从中心开始剪裁为指定长度
    """
    # 基础路径配置
    raw_dir = "npy/multirm/raw_npy"
    base_split_dir = "npy/multirm/51split"
    
    # 12 种修饰的名称顺序 (Index 0 到 11)
    mods = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Psi','AtoI']
    
    # 4类分组规则
    # A (Adenine): m6A(7), m1A(4), m6Am(8), Am(0), AtoI(11)
    # C (Cytosine): m5C(5), Cm(1)
    # G (Guanine): m7G(9), Gm(2)
    # U (Uracil): m5U(6), Psi(10), Um(3)
    group_4class_names = ['A', 'C', 'G', 'U']
    
    # 12类到4类的映射
    mod_to_group = {
        0: 0,   # Am -> A
        1: 1,   # Cm -> C
        2: 2,   # Gm -> G
        3: 3,   # Um -> U
        4: 0,   # m1A -> A
        5: 1,   # m5C -> C
        6: 3,   # m5U -> U
        7: 0,   # m6A -> A
        8: 0,   # m6Am -> A
        9: 2,   # m7G -> G
        10: 3,  # Psi -> U
        11: 0   # AtoI -> A
    }
    
    # 待处理的数据前缀
    prefixes = ["train", "test", "valid"]

    # 创建目录结构
    for mod in mods:
        for folder in ["pos", "neg"]:
            os.makedirs(os.path.join(base_split_dir, mod, folder), exist_ok=True)

    # 合并文件的输出目录
    merge_dir = os.path.join(base_split_dir, "split")
    os.makedirs(merge_dir, exist_ok=True)

    print(f"开始切分数据集，目标目录: {base_split_dir}")
    print(f"合并文件目录: {merge_dir}")
    print(f"剪裁长度: {crop_length} (原始长度: 1001)")
    print(f"4类分组规则:")
    for group_idx, group_name in enumerate(group_4class_names):
        mod_indices = [idx for idx, group in mod_to_group.items() if group == group_idx]
        mod_names = [mods[idx] for idx in sorted(mod_indices)]
        print(f"  {group_name} ({'Adenine' if group_name == 'A' else 'Cytosine' if group_name == 'C' else 'Guanine' if group_name == 'G' else 'Uracil'}): {', '.join(mod_names)}")
    print("-" * 50)

    for prefix in prefixes:
        in_file = os.path.join(raw_dir, f"{prefix}_in_nucleo.npy")
        out_file = os.path.join(raw_dir, f"{prefix}_out.npy")

        if not os.path.exists(in_file) or not os.path.exists(out_file):
            print(f"[Skip] 未找到 {prefix} 相关文件，跳过。")
            continue

        print(f"正在读取 {prefix} 数据...")
        # 序列数据 (n, 1001), 类型 S1
        data_in = np.load(in_file)
        # 标签数据 (n, 12), 类型 int
        data_out = np.load(out_file)
        
        n_samples = data_out.shape[0]
        original_seq_len = data_in.shape[1]
        
        # 检查剪裁长度是否有效
        if crop_length > original_seq_len:
            print(f"  [Warning] 剪裁长度 {crop_length} 大于原始序列长度 {original_seq_len}，将使用原始长度")
            crop_length = original_seq_len
        
        # 计算剪裁边界（从中心开始）
        if crop_length < original_seq_len:
            padding = (original_seq_len - crop_length) // 2
            start_idx = padding
            end_idx = start_idx + crop_length
            print(f"  剪裁范围: [{start_idx}, {end_idx})")
        else:
            start_idx = 0
            end_idx = original_seq_len

        # 剪裁序列
        if crop_length < original_seq_len:
            data_in_cropped = data_in[:, start_idx:end_idx]
        else:
            data_in_cropped = data_in

        # 生成4类标签
        data_out_4class = np.zeros((data_out.shape[0], 4), dtype=np.int8)
        for row_idx in range(data_out.shape[0]):
            # 找到当前样本的正类索引（如果有）
            pos_classes = np.where(data_out[row_idx] == 1)[0]
            if len(pos_classes) > 0:
                for class_idx in pos_classes:
                    group_idx = mod_to_group[class_idx]
                    data_out_4class[row_idx, group_idx] = 1
        
        # 保存合并文件（剪裁后的）
        np.save(os.path.join(merge_dir, f"{prefix}_in.npy"), data_in_cropped)
        np.save(os.path.join(merge_dir, f"{prefix}_out_12class.npy"), data_out)
        np.save(os.path.join(merge_dir, f"{prefix}_out_4class.npy"), data_out_4class)
        print(f"  已保存合并文件: {len(data_in_cropped)} 样本 (剪裁至{crop_length}bp)")

        # 遍历 12 个类别进行切分
        for i, mod_name in enumerate(mods):
            # 1. 提取当前类别的正样本 (Index i 为 1)
            pos_indices = np.where(data_out[:, i] == 1)[0]
            
            if len(pos_indices) == 0:
                print(f"  - {mod_name}: 无正样本")
                continue

            # 提取正样本块（使用剪裁后的数据）
            pos_in_block = data_in_cropped[pos_indices]
            pos_out_block = data_out[pos_indices]

            # 2. 生成4类标签
            # pos_out_4: 对于正样本，只有一个4类标签为1
            pos_out_4 = np.zeros((len(pos_indices), 4), dtype=np.int8)
            for row_idx in range(len(pos_indices)):
                # 找到当前样本的正类索引
                pos_class_idx = np.where(pos_out_block[row_idx] == 1)[0][0]
                # 映射到4类
                group_idx = mod_to_group[pos_class_idx]
                pos_out_4[row_idx, group_idx] = 1

            # 3. 提取紧随其后的负样本块 (全零行)
            # 根据块状规律，负样本起始于正样本块结束后的第一行
            neg_start = pos_indices[-1] + 1
            
            # 寻找负样本结束位置：直到遇到下一个有 1 的行，或文件结束
            neg_end = neg_start
            while neg_end < n_samples and np.all(data_out[neg_end] == 0):
                neg_end += 1
            
            # 提取负样本块（使用剪裁后的数据）
            neg_in_block = data_in_cropped[neg_start:neg_end]
            neg_out_block = data_out[neg_start:neg_end]

            # 4. 生成负样本的4类标签（全0）
            neg_out_4 = np.zeros((len(neg_in_block), 4), dtype=np.int8)

            # 5. 保存文件
            # 文件名格式: {prefix}_in.npy / {prefix}_out.npy
            save_path_pos = os.path.join(base_split_dir, mod_name, "pos")
            save_path_neg = os.path.join(base_split_dir, mod_name, "neg")

            np.save(os.path.join(save_path_pos, f"{prefix}_in.npy"), pos_in_block)
            np.save(os.path.join(save_path_pos, f"{prefix}_out.npy"), pos_out_block)
            np.save(os.path.join(save_path_pos, f"{prefix}_out_4class.npy"), pos_out_4)
            
            np.save(os.path.join(save_path_neg, f"{prefix}_in.npy"), neg_in_block)
            np.save(os.path.join(save_path_neg, f"{prefix}_out.npy"), neg_out_block)
            np.save(os.path.join(save_path_neg, f"{prefix}_out_4class.npy"), neg_out_4)

            print(f"  - {mod_name:4}: 已切分 (Pos: {len(pos_indices)}, Neg: {len(neg_in_block)})")

    print("-" * 50)
    print("所有数据切分完成！")

if __name__ == "__main__":
    split_dataset(crop_length=51)