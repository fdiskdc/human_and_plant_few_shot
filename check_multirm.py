import numpy as np
import os
import argparse
from collections import defaultdict

def analyze_multirm_structure(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"[Error] 无法读取 {file_path}: {e}")
        return

    n_samples, n_labels = data.shape
    
    # 存储每个 class 的统计信息
    # pos_count: 正样本数, total_zeros_after: 正样本后紧跟的全零行总数
    stats = {c: {"pos": 0, "neg_padding": 0, "m_list": []} for c in range(12)}
    
    print(f"\n文件: {os.path.basename(file_path)} (总行数: {n_samples})")
    
    i = 0
    while i < n_samples:
        row = data[i]
        
        if np.any(row == 1):
            # 找到正样本
            active_idx = np.where(row == 1)[0][0]
            stats[active_idx]["pos"] += 1
            
            # 开始往后数连续的全零行 (即该样本的 m)
            m_count = 0
            temp_j = i + 1
            while temp_j < n_samples and np.all(data[temp_j] == 0):
                # 预警：如果遇到下一个正样本，停止计数
                if temp_j < n_samples and np.any(data[temp_j] == 1):
                    break
                m_count += 1
                temp_j += 1
            
            stats[active_idx]["neg_padding"] += m_count
            stats[active_idx]["m_list"].append(m_count)
            
            # 跳过已统计的正样本及其填充
            i = temp_j
        else:
            # 这里的全零行如果不紧跟在正样本后，可能属于纯背景负样本
            i += 1

    # 打印分析结果
    header = f"{'Class':<8} | {'Pos':<6} | {'Neg(Pad)':<8} | {'m (Detected)':<12} | {'Ratio (N/P)':<10}"
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for c in range(12):
        s = stats[c]
        if s["pos"] > 0:
            # 取出现次数最多的 m 作为代表值
            m_val = max(set(s["m_list"]), key=s["m_list"].count) if s["m_list"] else 0
            # 检查 m 是否统一
            m_str = f"{m_val}" if len(set(s["m_list"])) == 1 else f"Mixed! {list(set(s['m_list']))}"
            
            ratio = s["neg_padding"] / s["pos"]
            print(f"Index {c:<3} | {s['pos']:<6} | {s['neg_padding']:<8} | {m_str:<12} | {ratio:<10.2f}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="npy/multirm/raw_npy")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"路径不存在: {args.dir}")
        return

    files = sorted([f for f in os.listdir(args.dir) if f.endswith("_out.npy")])
    for f in files:
        analyze_multirm_structure(os.path.join(args.dir, f))

if __name__ == "__main__":
    main()