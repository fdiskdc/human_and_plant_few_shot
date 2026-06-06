"""
run_attention_comparison.py - 长序列注意力对比实验完整流水线 / Long-Sequence Attention Comparison Experiment Pipeline

执行完整流水线:1) 选代表性序列 (top-N by m6A 修饰密度) 2) mRModN 全长 1001nt 推理 3) MultiRM 51nt 滑动窗口
4) modx 101nt 滑动窗口 5) EvoRMD 41nt 滑动窗口 6) 生成对比图表。
Executes full pipeline: 1) Select representative sequences 2) mRModN full inference 3) MultiRM/modx/EvoRMD sliding
window 4) Generate comparison figures.

功能模块 / Modules:
- 选代表性序列 / Select representative sequences
- 4 模型注意力推理 / 4-model attention inference
- 对比图表生成 / Comparison figure generation
- main: 主入口 / Main entry point

输入 / Inputs:
- json/human.json: 推理配置 / Inference config
- 4 个模型 checkpoints / 4 model checkpoints
- human3/1001loc.npy: m6A 修饰密度 / m6A density
- 命令行参数 / CLI: --output_dir, --top_n

输出 / Outputs:
- figs_atten/xxx/seq_XXXX.pdf, seq_XXXX.png: 对比图 / Comparison figures
- 4 模型 npz 注意力 / 4-model npz attention
- 时间戳子目录 / Timestamped subdirectory

数据流 / Data Flow:
1. 选代表性序列 / Select representative sequences
2. mRModN 全长推理 / mRModN full inference
3. MultiRM/modx/EvoRMD 滑动窗口推理 / Sliding window inference
4. 拼接注意力 / Stitch attention
5. 生成对比图 / Generate comparison figures

相关文件 / Related Files:
- 调用 / Calls: select_representative_sequences, inference_*, visualize_attention_comparison
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python run_attention_comparison.py
    python run_attention_comparison.py --output_dir figs_atten/my_exp --top_n 50

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

#!/usr/bin/env python3
"""
Run All: Long-Sequence Attention Comparison Experiment

Executes the full pipeline:
  1. Select representative sequences (top-N by modification density)
  2. mRModN full-length 1001nt inference
  3. MultiRM 51nt sliding window inference
  4. modx 101nt sliding window inference
  5. EvoRMD 41nt sliding window inference
  6. Visualize attention comparison (figures)

Usage:
    python run_attention_comparison.py
    python run_attention_comparison.py --output_dir figs_atten/my_exp --top_n 50
"""

import argparse
import subprocess
import sys
import time
import os

PYTHON = sys.executable


def build_scripts(top_n, output_dir):
    """
    生成注意力对比 pipeline 的脚本序列 / Build the script list for the attention-comparison pipeline.

    顺序: 选代表性序列 -> mRModN 全长推理 -> MultiRM / modx / EvoRMD 滑窗推理 -> 可视化。
    Sequence: select representatives -> mRModN full inference -> MultiRM /
    modx / EvoRMD sliding-window inference -> visualize.

    Args / 参数:
        top_n (int): [中文] 选 top 序列数 / [English] number of top sequences to pick.
        output_dir (str): [中文] 可视化输出目录 / [English] visualization output dir.

    Returns / 返回:
        List[Tuple[str, List[str]]]: [中文] `(脚本名, 参数列表)` 元组列表 /
            [English] list of `(script, args)` tuples.
    """

    return [
        ("select_representative_sequences.py", ["--top_n", str(top_n)]),
        ("inference_mrmodn_full.py",            []),
        ("inference_multirm_segmented.py",      ["--window_size", "51", "--stride", "1"]),
        ("inference_modx_segmented.py",         ["--window_size", "101", "--stride", "1"]),
        ("inference_evormd_segmented.py",       ["--window_size", "41", "--stride", "1"]),
        ("visualize_attention_comparison.py",   ["--output_dir", output_dir]),
    ]


def main(output_dir='fig/attention_comparison', top_n=100):
    """
    注意力对比实验主入口 / Attention-comparison experiment main entry.

    流程: 构造脚本序列 -> 依次串行执行 -> 每个脚本失败立即终止。
    Pipeline: build script list -> run them sequentially -> abort on first failure.

    Args / 参数:
        output_dir (str, optional): [中文] 可视化输出目录 / [English] output dir.
            Defaults to 'fig/attention_comparison'.
        top_n (int, optional): [中文] 选 top 序列数 / [English] top-N sequences.
            Defaults to 100.

    Called by / 被调用:
        - __main__ 块: [中文] 命令行直接调用 / [English] invoked from CLI.
    """

    scripts = build_scripts(top_n, output_dir)
    total_start = time.time()

    print(f"{'='*60}")
    print("Long-Sequence Attention Comparison Experiment")
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

        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

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
    parser = argparse.ArgumentParser(description="Run full attention comparison pipeline")
    parser.add_argument('--output_dir', type=str, default='figs_atten/attention_comparison',
                        help='Directory to save output figures (default: figs_atten/attention_comparison)')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top sequences by modification density (default: 100)')
    args = parser.parse_args()
    main(args.output_dir, args.top_n)
