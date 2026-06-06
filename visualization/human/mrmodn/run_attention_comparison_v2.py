"""
atten_comp/run_attention_comparison_v2.py - 长序列注意力对比实验 v2 / Long-Sequence Attention Comparison v2

v2 版本流水线:1) 选代表性序列 2) mRModN 全长推理 3) MultiRM 51nt 滑动窗口 4) modX 1001nt 全长 (无滑动窗口) 5) EvoRMD 41nt 滑动窗口
6) 生成 v2 对比图 (背景色块)。
V2 pipeline: 1) Select sequences 2) mRModN full 3) MultiRM 51nt 4) modX 1001nt full 5) EvoRMD 41nt 6) Generate v2 figures.

功能模块 / Modules:
- 选代表性序列 / Select representative sequences
- 4 模型推理 (modX 改为全长) / 4-model inference (modX full-length)
- v2 对比图生成 / V2 comparison figure generation
- main: 主入口 / Main entry point

输入 / Inputs:
- json/human.json: 推理配置 / Inference config
- 4 个模型 checkpoints / 4 model checkpoints
- 命令行参数 / CLI: --output_dir, --top_n

输出 / Outputs:
- figs_atten/xxx/seq_XXXX.pdf, seq_XXXX.png: v2 对比图 / V2 comparison figures
- 4 模型 npz 注意力 / 4-model npz attention

数据流 / Data Flow:
1. 选代表性序列 / Select representative sequences
2. 4 模型推理 (modX 改为全长) / 4-model inference (modX full)
3. 拼接注意力 / Stitch attention
4. 生成 v2 对比图 (背景色块) / Generate v2 figures (background blocks)

相关文件 / Related Files:
- 调用 / Calls: select_representative_sequences, inference_*, atten_comp/inference_modx_full, atten_comp/visualize_attention_comparison_v2
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python atten_comp/run_attention_comparison_v2.py

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

#!/usr/bin/env python3
"""
Run All v2: Long-Sequence Attention Comparison Experiment

Executes the full pipeline:
  1. Select representative sequences (top-N by modification density)
  2. mRModN full-length 1001nt inference
  3. MultiRM 51nt sliding window inference
  4. modX full-length 1001nt inference (no sliding window)
  5. EvoRMD 41nt sliding window inference
  6. Visualize attention comparison v2 (background blocks for window boundaries)

Usage:
    python atten_comp/run_attention_comparison_v2.py
    python atten_comp/run_attention_comparison_v2.py --output_dir figs_atten/my_exp --top_n 50
"""

import argparse
import subprocess
import sys
import time
import os

PYTHON = sys.executable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_scripts(top_n, low_n, output_dir):
    """
    生成 v2 注意力对比 pipeline 的脚本序列 / Build the v2 attention-comparison script list.

    v2 在 v1 基础上加入 modx 全长推理, 同时选序时区分 top / low 修饰密度。
    v2 adds modx full-length inference and picks representatives across both
    top and low modification density buckets.

    Args / 参数:
        top_n (int): [中文] 高修饰密度代表序列数 / [English] top-density sequence count.
        low_n (int): [中文] 低修饰密度代表序列数 / [English] low-density sequence count.
        output_dir (str): [中文] 可视化输出目录 / [English] visualization output dir.

    Returns / 返回:
        List[Tuple[str, List[str]]]: [中文] `(脚本, 参数)` 列表 / [English] `(script, args)` list.
    """

    return [
        ("select_representative_sequences.py", ["--top_n", str(top_n), "--low_n", str(low_n)]),
        ("inference_mrmodn_full.py",            []),
        ("inference_multirm_segmented.py",      ["--window_size", "51", "--stride", "1"]),
        ("atten_comp/inference_modx_full.py",   []),
        ("inference_evormd_segmented.py",       ["--window_size", "41", "--stride", "1"]),
        ("atten_comp/visualize_attention_comparison_v2.py", ["--output_dir", output_dir]),
    ]


def main(output_dir='fig/attention_comparison', top_n=100, low_n=100):
    """
    注意力对比 v2 实验主入口 / Attention-comparison v2 experiment main entry.

    Args / 参数:
        output_dir (str, optional): [中文] 可视化输出目录 / [English] output dir.
            Defaults to 'fig/attention_comparison'.
        top_n (int, optional): [中文] 高修饰密度代表序列数 / [English] top-density count.
            Defaults to 100.
        low_n (int, optional): [中文] 低修饰密度代表序列数 / [English] low-density count.
            Defaults to 100.

    Called by / 被调用:
        - __main__ 块: [中文] 命令行直接调用 / [English] invoked from CLI.
    """

    scripts = build_scripts(top_n, low_n, output_dir)
    total_start = time.time()

    print(f"{'='*60}")
    print("Long-Sequence Attention Comparison Experiment v2")
    print(f"{'='*60}")
    print(f"Python:    {PYTHON}")
    print(f"Top-N:     {top_n}")
    print(f"Output:    {os.path.abspath(output_dir)}/")
    print(f"Steps:     {len(scripts)}\n")

    for i, (script, args) in enumerate(scripts, 1):
        step_start = time.time()
        cmd = [PYTHON, script] + args
        print(f"\n[Step {i}/{len(scripts)}] {script}")
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  {'-'*50}")

        result = subprocess.run(cmd, cwd=ROOT_DIR)

        elapsed = time.time() - step_start
        if result.returncode != 0:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            print(f"  Stopping pipeline.")
            sys.exit(result.returncode)

        print(f"  Done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All {len(scripts)} steps completed in {total_elapsed:.1f}s")
    print(f"Output: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full attention comparison pipeline v2")
    parser.add_argument('--output_dir', type=str, default='figs_atten/attention_comparison',
                        help='Directory to save output figures (default: figs_atten/attention_comparison)')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of high-m6A-density sequences (default: 100)')
    parser.add_argument('--low_n', type=int, default=100,
                        help='Number of low-m6A-density sequences (default: 100)')
    args = parser.parse_args()
    main(args.output_dir, args.top_n, args.low_n)
