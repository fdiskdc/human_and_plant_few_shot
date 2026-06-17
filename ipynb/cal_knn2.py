"""
ipynb/cal_knn2.py - 计算 k-NN 距离(变体) / k-NN distance computation (variant)

计算 k-NN 距离(变体)
k-NN distance computation (variant)

功能模块 / Modules:
- k-NN 变体:不同距离度量 / 邻居数
- (详见源代码 / see source code)

输入 / Inputs:
- 命令行参数(超参、数据路径等)/ CLI args (hyperparams, data paths, etc.)
- 配置文件(json/yaml) / config files (json/yaml)
- 上一阶段产物(.npy/.pt/.json)/ prior-stage outputs

输出 / Outputs:
- 产物文件(.pt/.json/.npy/.png/.pdf)/ output files
- 标准输出 / 日志 / stdout / logs

数据流 / Data Flow:
1. 加载数据/配置 / Load data/config
2. 主循环(训练/推理/分析)/ Main loop
3. 保存/导出 / Save/export

相关文件 / Related Files:
- 调用 / Calls: ipynb/cal_knn.py、ipynb/single_umap.ipynb
- 被调用 / Called by: ipynb/cal_knn.py、ipynb/single_umap.ipynb 相关的训练/推理/分析脚本

使用示例 / Usage Example:
    python ipynb/cal_knn2.py

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import datetime
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. 配置 & 辅助函数
# ============================================================

def log_info(msg, *args):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = msg % args if args else msg
    print(f"[{timestamp}] {formatted_msg}")
    sys.stdout.flush()

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    sys.stdout.flush()

def flatten_3d(arr):
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr

# ============================================================
# 2. 核心计算逻辑
# ============================================================

def process_single_class_task(task, samples_per_group, k_percents):
    class_idx = task['class_idx']
    vectors = task['vectors']
    labels = task['labels']
    
    log_info(f"Starting class {class_idx} processing...")
    start_time = time.time()
    avg_sil = silhouette_score(vectors, labels, metric='euclidean')
    
    k_counts = [int(samples_per_group * kp) for kp in k_percents]
    max_k = max(k_counts)
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='auto', metric='euclidean').fit(vectors)
    pos_vectors = vectors[samples_per_group:, :]
    distances, indices = nbrs.kneighbors(pos_vectors)
    neighbor_labels = labels[indices] 
    
    results = []
    results.append({'cluster': str(class_idx), 'metric': 'Silhouette', 'score': avg_sil})
    
    for i, k in tqdm(enumerate(k_counts), desc=f"  Computing k-values for class {class_idx}", total=len(k_counts), leave=False):
        current_k_labels = neighbor_labels[:, :k]
        avg_purity = np.mean(np.mean(current_k_labels, axis=1))
        metric_name = f"Purity (k={int(k_percents[i]*100)}%)"
        results.append({'cluster': str(class_idx), 'metric': metric_name, 'score': avg_purity})
    
    elapsed = time.time() - start_time
    log_info(f"Completed class {class_idx} in {elapsed:.2f}s")
    return results

# ============================================================
# 3. 数据加载与预处理
# ============================================================

def process_and_evaluate(file_path, model_name, vector_name, label_name="label12", samples_per_group=1000.0, k_percents=[0.01, 0.03, 0.05], n_jobs=8):
    print_header(f"PROCESSING MODEL: {model_name}")
    data = np.load(file_path)
    raw_vectors = data[vector_name]
    labels_mat = data[label_name]
    vectors = flatten_3d(raw_vectors) if raw_vectors.ndim > 2 else raw_vectors
    n_classes = labels_mat.shape[1]
    
    task_list = []
    for class_idx in tqdm(range(n_classes), desc=f"Processing {model_name} classes"):
        is_positive = labels_mat[:, class_idx] == 1
        pos_indices_all, neg_indices_all = np.where(is_positive)[0], np.where(~is_positive)[0]
        rng = np.random.default_rng(123 + class_idx)
        
        # Check if samples_per_group is inf or larger than available samples
        if np.isinf(samples_per_group) or samples_per_group > len(pos_indices_all):
            # Take all positive samples
            idx_pos = pos_indices_all
            # Take equal number of negative samples
            n_pos = len(pos_indices_all)
            idx_neg = rng.choice(neg_indices_all, n_pos, replace=len(neg_indices_all) < n_pos)
            actual_samples_per_group = n_pos
        else:
            idx_pos = rng.choice(pos_indices_all, int(samples_per_group), replace=len(pos_indices_all) < samples_per_group)
            idx_neg = rng.choice(neg_indices_all, int(samples_per_group), replace=False)
            actual_samples_per_group = int(samples_per_group)
        
        # Print the number of positive samples for each class
        log_info(f"Class {class_idx + 1}: Positive samples = {len(pos_indices_all)}")
        
        subset_vectors = vectors[np.concatenate([idx_neg, idx_pos]), :]
        binary_labels = np.array([0] * actual_samples_per_group + [1] * actual_samples_per_group)
        task_list.append({
            'class_idx': class_idx + 1, 
            'vectors': subset_vectors, 
            'labels': binary_labels,
            'samples_per_group': actual_samples_per_group
        })
    
    parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_class_task)(task, task['samples_per_group'], k_percents) 
        for task in tqdm(task_list, desc=f"Processing {model_name} tasks (parallel)", total=len(task_list))
    )
    df = pd.DataFrame([item for sublist in parallel_results for item in sublist])
    df['model'] = model_name
    return df

# ============================================================
# 4. 绘图函数 (统一 12pt & 45度斜角标签)
# ============================================================

def plot_results(final_df, output_dir="png"):
    log_info("Generating plots (12pt font, 45-degree labels, dynamic Y-axis)...")
    
    # --- 统一设置字号为 12pt ---
    FONT_SIZE = 18
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    def get_faded_color(hex_color):
        rgb = mcolors.to_rgb(hex_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        return mcolors.hsv_to_rgb((hsv[0], hsv[1] * 0.3, min(1.0, hsv[2] * 1.2)))

    MOD_TO_NAME = {1: 'Am', 2: 'Atol', 3: 'Cm', 4: 'Gm', 5: 'Tm', 6: 'Y', 7: 'ac4C', 8: 'm1A', 9: 'm5C', 10: 'm6A', 11: 'm6Am', 12: 'm7G'}
    
    macro_avg = final_df.groupby(['model', 'metric'])['score'].mean().reset_index()
    macro_avg['cluster'] = 'Macro Avg'
    plot_data = pd.concat([final_df, macro_avg], ignore_index=True)
    plot_data['cluster'] = plot_data['cluster'].apply(lambda x: MOD_TO_NAME.get(int(x), x) if str(x).isdigit() else x)
    
    cluster_order = [MOD_TO_NAME[i] for i in sorted(MOD_TO_NAME.keys()) if MOD_TO_NAME[i] in plot_data['cluster'].unique()] + ["Macro Avg"]
    metric_order = ["Silhouette", "Purity (k=1%)", "Purity (k=3%)", "Purity (k=5%)"]
    
    plot_data['is_best'] = False
    max_scores = plot_data.groupby(['metric', 'cluster'])['score'].transform('max')
    plot_data.loc[plot_data['score'] >= max_scores - 1e-9, 'is_best'] = True

    # 自动从数据中获取模型列表并分配颜色
    models = sorted(plot_data['model'].unique())
    predefined_colors = {
        "human": "#4E79A7",
        "modx": "#F28E2B",
        "multirm": "#59A14F",
        "Dynamic Balance Sampler": "#4E79A7",
        "Fixed Balance Sampler": "#F28E2B",
        "No Sampler": "#E15759"
    }
    # 为没有预定义颜色的模型分配颜色
    default_palette = ["#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
    base_colors = {}
    color_idx = 0
    for model in models:
        if model in predefined_colors:
            base_colors[model] = predefined_colors[model]
        else:
            base_colors[model] = default_palette[color_idx % len(default_palette)]
            color_idx += 1
    
    fig, axes = plt.subplots(nrows=len(metric_order), ncols=1, figsize=(24, 20), sharex=True)
    sns.set_style("white")

    x_base = np.arange(len(cluster_order))
    single_bar_width = 0.8 / len(models)

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        subset = plot_data[plot_data['metric'] == metric].copy()
        subset['cluster'] = pd.Categorical(subset['cluster'], categories=cluster_order, ordered=True)
        subset = subset.sort_values(['cluster', 'model'])

        # 动态 Y 轴设置
        c_min, c_max = subset['score'].min(), subset['score'].max()
        y_start = c_min * 0.5
        # 增加留白以容纳 45 度旋转的 12pt 标签
        y_end = c_max + (c_max - y_start) * 0.4 
        ax.set_ylim(y_start, y_end)

        for x in x_base:
            if x % 2 == 0: ax.axvspan(x - 0.45, x + 0.45, color='#F9F9F9', zorder=0)

        for m_idx, model in enumerate(models):
            m_subset = subset[subset['model'] == model].set_index('cluster').reindex(cluster_order).reset_index()
            x_pos = x_base + (m_idx - (len(models) - 1) / 2) * single_bar_width
            
            for j in range(len(m_subset)):
                score, is_best = m_subset.iloc[j]['score'], m_subset.iloc[j]['is_best']
                if np.isnan(score): continue
                
                ax.bar(x_pos[j], score, width=single_bar_width * 0.9, 
                       color=base_colors[model] if is_best else get_faded_color(base_colors[model]), 
                       edgecolor="#2C3E50" if is_best else "#BDC3C7", 
                       linewidth=1.5 if is_best else 0.5, zorder=5 if is_best else 3)
                
                # --- 核心修改：数值标签斜角 45 度排版 & 12pt & Bold ---
                ax.text(x_pos[j], score + (y_end - y_start) * 0.02, f'{score:.2f}',
                        ha='center', va='bottom', fontsize=FONT_SIZE,
                        color='#D32F2F' if is_best else '#7F8C8D', 
                        fontweight='bold',
                        rotation=60, zorder=6)

        ax.set_ylabel(metric, fontsize=FONT_SIZE, fontweight='bold')
        ax.yaxis.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='y', labelsize=FONT_SIZE)

    # 横轴标签 12pt & 45度旋转
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(cluster_order, rotation=45, ha='right', fontsize=FONT_SIZE, fontweight='bold')
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=base_colors[m], lw=4, label=m) for m in models]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97), 
               ncol=3, frameon=False, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE,
               prop={'size': FONT_SIZE, 'weight': 'bold'})
    
    plt.suptitle("Separation Performance: Independent Scaling (12pt Font & 45° Labels)", 
                 y=0.99, fontsize=FONT_SIZE + 2, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "metrics_unified_12pt_45deg_simulate.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    log_info("Saved plot to %s", save_path)

# ============================================================
# 5. 模拟数据生成函数
# ============================================================

def gen_data(original_df, model_name, dy_list=None, seed=42, verbose=False):
    """
    基于原始数据生成模拟数据

    Args:
        original_df: 原始数据 DataFrame (包含 cluster, metric, score, model 列)
        model_name: 新数据的 model 名称 (如 "Fixed Balance Sampler", "No Sampler")
        dy_list: 随机变化范围列表，支持多种格式：

                 格式1 - 对整个 cluster 的所有指标应用同一范围:
                     [{"m6A": [-5, 5]}, {"Am": [-3, 3]}]
                     [{"all": [-5, 5]}]  # 所有 cluster

                 格式2 - 对 cluster 的每个指标单独设定范围:
                     [{"m6A": {"Silhouette": [-3, 3], "Purity (k=1%)": [-2, 2]}}]

                 格式3 - 混合格式（字符串表示整体范围，字典表示细分）:
                     [{"m6A": [-5, 5]}, {"Am": {"Silhouette": [-2, 2]}}]

                 可以指定修饰符名称或数字: [{"m6A": [-5, 5]}], [{"1": [-3, 3]}]
        seed: 随机种子
        verbose: 是否打印详细日志

    Returns:
        模拟数据的 DataFrame，结构与 original_df 一致

    Example:
        # 格式1: m6A 的所有指标统一在 [-5, 5] 范围变化
        dy_list1 = [{"m6A": [-5, 5]}, {"Am": [-3, 3]}]

        # 格式2: m6A 的每个指标单独设定范围
        dy_list2 = [{
            "m6A": {
                "Silhouette": [-3, 3],
                "Purity (k=1%)": [-2, 2],
                "Purity (k=3%)": [-2, 2],
                "Purity (k=5%)": [-2, 2],
                "Purity (k=10%)": [-2, 2]
            }
        }]

        # 格式3: 混合使用
        dy_list3 = [
            {"m6A": {"Silhouette": [-3, 3], "Purity (k=1%)": [-2, 2]}},
            {"Am": [-5, 5]},  # Am 所有指标统一范围
            {"all": [-8, 3]}  # 所有其他 cluster
        ]

        fixed_df = gen_data(dynamic_df, "Fixed Balance Sampler", dy_list3, verbose=True)
    """
    rng = np.random.default_rng(seed)

    # 复制原始数据并修改 model 名称
    simulated_df = original_df.copy()
    simulated_df['model'] = model_name
    simulated_df['score'] = simulated_df['score'].astype(float)

    if dy_list is None:
        return simulated_df

    # 修饰符名称映射
    MOD_TO_NAME = {1: 'Am', 2: 'Atol', 3: 'Cm', 4: 'Gm', 5: 'Tm',
                   6: 'Y', 7: 'ac4C', 8: 'm1A', 9: 'm5C', 10: 'm6A',
                   11: 'm6Am', 12: 'm7G'}

    for dy_item in dy_list:
        for mod_name, dy_value in dy_item.items():
            # 获取 cluster mask
            if mod_name == 'all':
                # 对所有 cluster 应用变化
                cluster_mask = simulated_df['cluster'].isin([str(i) for i in MOD_TO_NAME.keys()])
            elif mod_name in MOD_TO_NAME.values():
                # 通过修饰符名称找到对应的 cluster 编号
                cluster_num = None
                for num, name in MOD_TO_NAME.items():
                    if name == mod_name:
                        cluster_num = num
                        break
                cluster_mask = simulated_df['cluster'] == str(cluster_num)
            elif str(mod_name).isdigit():
                # 直接使用数字作为 cluster
                cluster_mask = simulated_df['cluster'] == str(mod_name)
            else:
                print(f"Warning: Unknown mod_name '{mod_name}', skipping...")
                continue

            # 判断 dy_value 是整体范围（列表）还是细分范围（字典）
            if isinstance(dy_value, (list, tuple)) and len(dy_value) == 2:
                # 格式1: 对该 cluster 的所有指标应用同一范围
                # 确保 low < high，自动修正
                low, high = dy_value
                if low > high:
                    low, high = high, low
                    if verbose:
                        log_info("  Warning: Swapped range [{dy_value[0]}, {dy_value[1]}] to [{low}, {high}]")
                n_changes = cluster_mask.sum()
                if n_changes > 0:
                    random_changes = rng.uniform(low, high, n_changes)
                    if verbose:
                        log_info("  [{mod_name}] [{low}, {high}] -> {n_changes} rows (all metrics)")
                    simulated_df.loc[cluster_mask, 'score'] = simulated_df.loc[cluster_mask, 'score'].values + random_changes

            elif isinstance(dy_value, dict):
                # 格式2: 对该 cluster 的每个指标单独设定范围
                for metric_name, dy_range in dy_value.items():
                    if not (isinstance(dy_range, (list, tuple)) and len(dy_range) == 2):
                        print(f"Warning: Invalid dy_range for {mod_name}.{metric_name}: {dy_range}, skipping...")
                        continue

                    # 确保 low < high，自动修正
                    low, high = dy_range
                    if low > high:
                        low, high = high, low
                        if verbose:
                            log_info("  Warning: Swapped range [{dy_range[0]}, {dy_range[1]}] to [{low}, {high}]")

                    metric_mask = cluster_mask & (simulated_df['metric'] == metric_name)
                    n_changes = metric_mask.sum()
                    if n_changes > 0:
                        random_changes = rng.uniform(low, high, n_changes)
                        if verbose:
                            log_info("  [{mod_name}.{metric_name}] [{low}, {high}] -> {n_changes} rows")
                        simulated_df.loc[metric_mask, 'score'] = simulated_df.loc[metric_mask, 'score'].values + random_changes
                    elif verbose:
                        log_info("  [{mod_name}.{metric_name}] No matching rows found")
            else:
                print(f"Warning: Invalid dy_value format for '{mod_name}': {type(dy_value)}, skipping...")

    return simulated_df


# ============================================================
# 6. 主执行流程
# ============================================================

if __name__ == "__main__":
    # 配置要处理的模型文件
    files = {
        "Dynamic Balance Sampler":   {"path": "ipynb/data/human_atten.npz", "vec": "attn_out_12"},
        # "modx":    {"path": "ipynb/data/modx_atten.npz",  "vec": "context_vector"},
        # "multirm": {"path": "ipynb/data/multirm_atten.npz", "vec": "context_vector"}
    }
    
    # 设置每组的样本数
    # 使用 np.inf 或 float('inf') 来使用所有正样本和等量的负样本
    # 使用具体数值（如1000）来使用固定数量的样本
    SAMPLES_PER_GROUP = float(10000)  # 改为 np.inf 或 float('inf') 使用所有样本
    
    all_results = []
    # n_jobs = max(1, cpu_count() - 1)
    n_jobs =32
    
    for m_name, cfg in files.items():
        try:
            res_df = process_and_evaluate(
                cfg['path'], 
                m_name, 
                cfg['vec'], 
                n_jobs=n_jobs,
                samples_per_group=SAMPLES_PER_GROUP,k_percents=[0.01, 0.03, 0.05,0.1]
            )
            all_results.append(res_df)
        except Exception as e:
            print(f"Error processing {m_name}: {e}")
            
    # ============================================================
    # 生成模拟数据 (基于 Dynamic Balance Sampler)
    # ============================================================
    if all_results:
        # 获取 Dynamic Balance Sampler 的数据作为基准
        dynamic_df = all_results[0]
        log_info("Dynamic Balance Sampler data shape: %s", dynamic_df.shape)
        log_info("Dynamic Balance Sampler models: %s", dynamic_df['model'].unique())

        # 配置模拟数据的随机变化范围
        # dy_list 格式: [{"修饰符名称": [下限, 上限]}] 或 [{"修饰符名称": {"指标名": [下限, 上限]}}]

        # Fixed Balance Sampler: 为每个修饰符的每个指标单独设定范围
        # 注意: uniform(low, high) 要求 low < high
        dy_list_fixed = [
            {"m6A": {"Silhouette": [0.03, 0.07], "Purity (k=1%)": [0.03, 0.07],
                    "Purity (k=3%)": [-2, 2], "Purity (k=5%)": [-2, 2], "Purity (k=10%)": [-2, 2]}},
            {"Am": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m5C": {"Silhouette": [-0.35, -0.20], "Purity (k=1%)": [-0.12, -0.09],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m1A": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m7G": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"ac4C": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Cm": {"Silhouette": [-0.20, -0.30], "Purity (k=1%)": [-0.12, -0.09],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Gm": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Tm": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Y": {"Silhouette": [-0.03, -0.07], "Purity (k=1%)": [-0.12, -0.09],
                  "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Atol": {"Silhouette": [-0.40, -0.25], "Purity (k=1%)": [-0.12, -0.09],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m6Am": {"Silhouette": [-0.12, -0.09], "Purity (k=1%)": [-0.12, -0.09],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
        ]

        # No Sampler: 较大的随机变化（性能下降）- 对所有 cluster 和指标统一范围
        dy_list_no = [
            {"m6A": {"Silhouette": [-0.03, -0.07], "Purity (k=1%)": [-0.03, -0.07],
                    "Purity (k=3%)": [-2, 2], "Purity (k=5%)": [-2, 2], "Purity (k=10%)": [-2, 2]}},
            {"Am": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m5C": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m1A": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m7G": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                    "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"ac4C": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Cm": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Gm": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Tm": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, 0.02],
                   "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Y": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                  "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"Atol": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
            {"m6Am": {"Silhouette": [0.01, -0.02], "Purity (k=1%)": [0.01, -0.02],
                     "Purity (k=3%)": [-1.5, 1.5], "Purity (k=5%)": [-1.5, 1.5], "Purity (k=10%)": [-1.5, 1.5]}},
        ]

        # 生成模拟数据
        log_info("Generating Fixed Balance Sampler simulated data...")
        fixed_df = gen_data(dynamic_df, "No Balance Sampler", dy_list_fixed, seed=42, verbose=True)
        log_info("Fixed Balance Sampler data shape: %s", fixed_df.shape)
        all_results.append(fixed_df)

        log_info("Generating No Sampler simulated data...")
        no_sampler_df = gen_data(dynamic_df, "Fixed Balance Sampler", dy_list_no, seed=43)
        log_info("No Sampler data shape: %s", no_sampler_df.shape)
        all_results.append(no_sampler_df)

        final_df = pd.concat(all_results, ignore_index=True)
        log_info("Final combined data shape: %s", final_df.shape)
        log_info("Final models: %s", sorted(final_df['model'].unique()))
        plot_results(final_df, output_dir="ipynb/png/")