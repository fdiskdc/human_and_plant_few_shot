"""
Few-shot Analysis Utilities

This module contains utility functions used across the few-shot analysis pipeline.
"""

import csv
import json
import os
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.fewshot_analysis_constants import MORANDI_GRID


def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def build_output_dir(base='output', prefix='zero_fewshot_analysis'):
    """Create and return a timestamped output directory path."""
    ts = get_timestamp()
    out_dir = os.path.join(base, f'{prefix}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, base_path, dpi=300):
    """Save figure as both PNG and PDF."""
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{base_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def apply_plot_style(ax):
    """Apply consistent plot styling."""
    ax.grid(True, color=MORANDI_GRID, linewidth=0.7, alpha=0.8)
    ax.set_facecolor('#FBF8F3')
    for spine in ax.spines.values():
        spine.set_color('#D7CFC4')
    ax.tick_params(colors='#6E675F')


def stable_mean(values):
    """Calculate mean, returning NaN for empty lists."""
    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))


def convert_for_json(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    return obj


def save_json(data, path):
    """Save data as JSON file."""
    with open(path, 'w') as f:
        json.dump(convert_for_json(data), f, indent=2)


def call_r_script(input_dir, output_dir, logger, conda_env='learn-new'):
    """Call R plotting script using conda environment."""
    r_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plot_zero_fewshot_analysis.R')
    if not os.path.exists(r_script_path):
        logger.warning(f"R script not found at {r_script_path}, skipping R plots.")
        return False
    try:
        # Convert to absolute paths to ensure R script can find files
        abs_input_dir = os.path.abspath(input_dir)
        abs_output_dir = os.path.abspath(output_dir)
        abs_r_script_path = os.path.abspath(r_script_path)
        
        # Use conda run to execute Rscript in the specified environment
        cmd = ['conda', 'run', '-n', conda_env, 'Rscript', abs_r_script_path,
               '--input_dir', abs_input_dir, '--output_dir', abs_output_dir]
        logger.info(f"Running R script with conda env '{conda_env}': {' '.join(cmd)}")
        logger.info(f"  Input dir (absolute): {abs_input_dir}")
        logger.info(f"  Output dir (absolute): {abs_output_dir}")
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"R script failed with return code {result.returncode}")
            logger.warning(f"R stderr: {result.stderr}")
            return False
        logger.info("R plotting script completed successfully.")
        if result.stdout:
            logger.info(f"R stdout: {result.stdout}")
        return True
    except FileNotFoundError:
        logger.warning("Rscript not found in PATH. Skipping R plots.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("R script timed out after 300 seconds.")
        return False
    except Exception as e:
        logger.warning(f"Failed to run R script: {e}")
        return False
