"""
atten_comp/run_attention_comparison_v3.py - 长序列注意力对比实验 v3 / Long-Sequence Attention Comparison v3

v3 版本流水线:1) 选代表性序列 2) mRModN 全长推理 3) MultiRM 51nt 滑动窗口 4) modX 1001nt 全长 (无滑动窗口) 5) EvoRMD 41nt 滑动窗口
6) 生成 v3 对比图 (共享 Y 轴上限)。
V3 pipeline: 1) Select sequences 2) mRModN full 3) MultiRM 51nt 4) modX full 5) EvoRMD 41nt 6) Generate v3 figures with a shared per-sequence y-axis cap.

功能模块 / Modules:
- 选代表性序列 / Select representative sequences
- 4 模型推理 (modX 改为全长) / 4-model inference (modX full-length)
- v3 对比图生成 / V3 comparison figure generation
- main: 主入口 / Main entry point

输入 / Inputs:
- json/human.json: 推理配置 / Inference config
- 4 个模型 checkpoints / 4 model checkpoints
- 命令行参数 / CLI: --output_dir, --top_n

输出 / Outputs:
- figs_atten/xxx/seq_XXXX.pdf, seq_XXXX.png: v3 对比图 / V3 comparison figures
- 4 模型 npz 注意力 / 4-model npz attention

数据流 / Data Flow:
1. 选代表性序列 / Select representative sequences
2. 4 模型推理 (modX 改为全长) / 4-model inference (modX full)
3. 拼接注意力 / Stitch attention
4. 生成 v3 对比图 (共享 Y 轴上限) / Generate v2 figures (background blocks)

相关文件 / Related Files:
- 调用 / Calls: select_representative_sequences, inference_*, atten_comp/inference_modx_full, atten_comp/visualize_attention_comparison_v3
- 被调用 / Called by: shell scripts, manual CLI

使用示例 / Usage Example:
    python atten_comp/run_attention_comparison_v3.py

作者 / Author: RGCNFormer Project
日期 / Date: 2026-06-03
版本 / Version: 1.0
"""

#!/usr/bin/env python3
"""
Run All v3: Long-Sequence Attention Comparison Experiment

Executes the full pipeline:
  1. Select representative sequences (top-N by modification density)
  2. mRModN full-length 1001nt inference
  3. MultiRM 51nt sliding window inference
  4. modX full-length 1001nt inference (no sliding window)
  5. EvoRMD 41nt sliding window inference
  6. Visualize attention comparison v3 (shared per-sequence y-axis cap)

Usage:
    python atten_comp/run_attention_comparison_v3.py
    python atten_comp/run_attention_comparison_v3.py --output_dir figs_atten/my_exp --top_n 50
"""

import argparse
import subprocess
import sys
import time
import os

PYTHON = sys.executable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_scripts(top_n, low_n, output_dir, clip_quantile):
    return [
        ("select_representative_sequences.py", ["--top_n", str(top_n), "--low_n", str(low_n)]),
        ("inference_mrmodn_full.py",            []),
        ("inference_multirm_segmented.py",      ["--window_size", "51", "--stride", "1"]),
        ("atten_comp/inference_modx_full.py",   []),
        ("inference_evormd_segmented.py",       ["--window_size", "41", "--stride", "1"]),
        ("atten_comp/visualize_attention_comparison_v3.py", ["--output_dir", output_dir, "--clip_quantile", str(clip_quantile)]),
    ]


def main(output_dir='fig/attention_comparison', top_n=100, low_n=100, clip_quantile=0.99):
    scripts = build_scripts(top_n, low_n, output_dir, clip_quantile)
    total_start = time.time()

    print(f"{'='*60}")
    print("Long-Sequence Attention Comparison Experiment v3")
    print(f"{'='*60}")
    print(f"Python:    {PYTHON}")
    print(f"Top-N:     {top_n}")
    print(f"Output:    {os.path.abspath(output_dir)}/" )
    print(f"Y cap:     per-sequence quantile {clip_quantile:g}")
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
    parser = argparse.ArgumentParser(description="Run full attention comparison pipeline v3")
    parser.add_argument('--output_dir', type=str, default='figs_atten/attention_comparison',
                        help='Directory to save output figures (default: figs_atten/attention_comparison)')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of high-m6A-density sequences (default: 100)')
    parser.add_argument('--low_n', type=int, default=100,
                        help='Number of low-m6A-density sequences (default: 100)')
    parser.add_argument('--clip_quantile', type=float, default=0.99,
                        help='Per-sequence quantile used as the shared y-axis cap (default: 0.99)')
    args = parser.parse_args()
    main(args.output_dir, args.top_n, args.low_n, args.clip_quantile)
