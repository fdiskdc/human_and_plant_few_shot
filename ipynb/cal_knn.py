"""
ipynb/cal_knn.py - 计算 k-NN 距离 / k-NN distance computation

计算 k-NN 距离
k-NN distance computation

功能模块 / Modules:
- k-NN 距离矩阵 / 邻居索引
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
- 调用 / Calls: utils/fewshot_analysis_*.py、ipynb/single_umap.ipynb
- 被调用 / Called by: utils/fewshot_analysis_*.py、ipynb/single_umap.ipynb 相关的训练/推理/分析脚本

使用示例 / Usage Example:
    python ipynb/cal_knn.py

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
    FONT_SIZE = 12
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

    base_colors = {"human": "#4E79A7", "modx": "#F28E2B", "multirm": "#59A14F"}
    models = list(base_colors.keys())
    
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
                        rotation=45, zorder=6)

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
    save_path = os.path.join(output_dir, "metrics_unified_12pt_45deg.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    log_info("Saved plot to %s", save_path)

# ============================================================
# 5. 主执行流程
# ============================================================

if __name__ == "__main__":
    # 配置要处理的模型文件
    files = {
        "human":   {"path": "ipynb/data/human_atten.npz", "vec": "attn_out_12"},
        "modx":    {"path": "ipynb/data/modx_atten.npz",  "vec": "context_vector"},
        "multirm": {"path": "ipynb/data/multirm_atten.npz", "vec": "context_vector"}
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
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        plot_results(final_df, output_dir="ipynb/png/")