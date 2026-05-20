'''
Author: Chao Deng && chaodeng987@outlook.com
Date: 2026-05-10 09:49:13
LastEditors: Chao Deng && chaodeng987@outlook.com
LastEditTime: 2026-05-10 11:08:14
FilePath: /rgcnformer_sum/atten_comp/run_attention_comparison_v2.py
Description: 
那只是一场游戏一场梦
 
https://orcid.org/0009-0009-8520-1656
DOI: 10.3390/app15158626
DOI: 10.3390/rs17142354
Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
'''
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
    return [
        ("select_representative_sequences.py", ["--top_n", str(top_n), "--low_n", str(low_n)]),
        ("inference_mrmodn_full.py",            []),
        ("inference_multirm_segmented.py",      ["--window_size", "51", "--stride", "1"]),
        ("atten_comp/inference_modx_full.py",   []),
        ("inference_evormd_segmented.py",       ["--window_size", "41", "--stride", "1"]),
        ("atten_comp/visualize_attention_comparison_v2.py", ["--output_dir", output_dir]),
    ]


def main(output_dir='fig/attention_comparison', top_n=100, low_n=100):
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
