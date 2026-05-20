"""
Attention Comparison Visualization v2 — m6A Focus (Background Blocks)

Generates per-sequence figures comparing m6A attention patterns across 4 models:
  - mRModN (per-class MHA, full 1001nt)
  - MultiRM (shared Bahdanau, 51nt sliding window)
  - modX (shared Bahdanau, full 1001nt inference, 101-window boundary visualization)
  - EvoRMD (shared scalar, 41nt sliding window)

Each figure: 4 subplots (one per model), m6A attention curve + true sites + window region shading.
Window boundaries are shown as alternating background color blocks (ax.axvspan) instead of
dashed vertical lines, avoiding visual confusion with True m6A site markers.

Output: fig/attention_comparison/seq_XXXX.pdf + seq_XXXX.png

Style: Times New Roman 18pt, Morandi palette, journal quality.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.font_manager as fm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

M6A_INDEX = 9
M6A_LABEL = M6A_INDEX + 1

MORANDI = {
    'mRModN':   '#8DA9C4',
    'MultiRM':  '#B5838D',
    'modx':     '#A3B18A',
    'EvoRMD':   '#DDB892',
    'true_site':'#6B705C',
    'boundary': '#9E9E9E',
    'bg':       '#FAFAF8',
}

MODEL_META = [
    ('mRModN',  'mRModN',   MORANDI['mRModN'],  None),
    ('MultiRM', 'MultiRM',  MORANDI['MultiRM'], list(range(50, 1001, 51))),
    ('modX',    'modX',     MORANDI['modx'],    list(range(100, 1001, 101))),
    ('EvoRMD',  'EvoRMD',   MORANDI['EvoRMD'],  list(range(40, 1001, 41))),
]


def setup_journal_style():
    font_path = None
    for p in ['/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
              '/usr/share/fonts/TTF/Times_New_Roman.ttf',
              '/usr/share/fonts/truetype/times.ttf']:
        if os.path.exists(p):
            font_path = p
            break

    props = {'family': 'serif', 'size': 18}
    if font_path:
        fm.fontManager.addfont(font_path)
        props['serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'mathtext.fontset': 'stix',
    })


def compute_topk_recall(attn, sites, k=50):
    """
    Top-K Recall: fraction of true m6A sites captured by top-K attention positions.

    Uses a fixed K=50 for fair cross-model comparison regardless of
    per-sequence m6A site count.

    Args:
        attn: [1001] attention weights for m6A (L1-normalized)
        sites: [1001] site labels (0 = no modification, M6A_LABEL = m6A)
        k: number of top attention positions to consider (default 50)

    Returns:
        recall: float in [0, 1]
    """
    true_pos = np.where(sites == M6A_LABEL)[0]
    n_true = len(true_pos)
    if n_true == 0:
        return 0.0
    topk_pos = np.argsort(attn)[-k:]
    hits = np.intersect1d(true_pos, topk_pos)
    return len(hits) / min(n_true, k)


def visualize_sequence(seq_idx, rank, mrmodn_attn, multirm_attn,
                       modx_attn, evormd_attn, sites_1001, output_dir):

    fig, axes = plt.subplots(4, 1, figsize=(20//2, 16//2), sharex=True)
    fig.patch.set_facecolor(MORANDI['bg'])
    fig.subplots_adjust(hspace=0.50, top=0.93, bottom=0.10, left=0.10, right=0.92)

    attns = [mrmodn_attn, multirm_attn, modx_attn, evormd_attn]
    n_true = int(np.sum(sites_1001 == M6A_LABEL))
    true_positions = np.where(sites_1001 == M6A_LABEL)[0]

    fig.suptitle(
        f'Sequence #{rank + 1} (index={seq_idx})  —  m6A sites: {n_true}',
        fontsize=20, fontweight='bold', y=0.97
    )

    for ax, (key, label, color, boundaries) in zip(axes, MODEL_META):
        ax.set_facecolor(MORANDI['bg'])

        attn = attns[MODEL_META.index((key, label, color, boundaries))]
        x = np.arange(len(attn))

        ax.fill_between(x, attn, alpha=0.20, color=color)
        ax.plot(x, attn, color=color, linewidth=1.8)

        for tp in true_positions:
            ax.axvline(x=tp, color=MORANDI['true_site'], linewidth=1.4,
                       linestyle='--', alpha=0.85, zorder=3)

        if boundaries:
            prev = 0
            for i, bw in enumerate(boundaries + [1001]):
                if i % 2 == 1:
                    ax.axvspan(prev, bw, alpha=0.10, color=MORANDI['boundary'], zorder=0)
                prev = bw

        topk = compute_topk_recall(attn, sites_1001, k=50)
        ax.text(0.99, 0.92, f'Top-50 Recall = {topk:.1%}',
                transform=ax.transAxes, fontsize=14, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85))

        ax.set_ylabel('Attention', fontsize=16)
        ax.set_xlim(0, 1000)
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.text(0.01, 0.92, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', color=color)

    axes[-1].set_xlabel('Sequence Position (nt)', fontsize=18)

    legend_elements = [
        Line2D([0], [0], color=MORANDI['true_site'], linewidth=1.4,
               linestyle='--', label=f'True m6A site ({n_true})'),
        Patch(facecolor=MORANDI['boundary'], alpha=0.10, label='Window region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=14, framealpha=0.9,
               bbox_to_anchor=(0.52, 0.01))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f'seq_{rank + 1:04d}_idx{seq_idx}')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    fig.savefig(f'{base}.pdf', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return f'{base}.png'


def main(mrmodn_path='npy/mrmodn_full_atten.npz',
         multirm_path='npy/multirm_segmented_atten.npz',
         modx_path='npy/modx_full_atten.npz',
         evormd_path='npy/evormd_segmented_atten.npz',
         output_dir='fig/attention_comparison'):

    setup_journal_style()

    print(f"{'='*60}")
    print("m6A Attention Comparison Visualization v2 (Background Blocks)")
    print(f"{'='*60}")

    print("\nLoading attention data...")
    mrmodn = np.load(mrmodn_path, allow_pickle=True)
    multirm = np.load(multirm_path, allow_pickle=True)
    modx = np.load(modx_path, allow_pickle=True)
    evormd = np.load(evormd_path, allow_pickle=True)

    indices = mrmodn['indices']
    sites = mrmodn['sites']
    n_seq = len(indices)
    print(f"Sequences: {n_seq}")

    mrmodn_attn  = mrmodn['attn_weights']
    multirm_attn = multirm['attn_weights']
    modx_attn    = modx['attn_weights']
    evormd_attn  = evormd['attn_weights']

    multirm_attn = multirm_attn / (multirm_attn.sum(axis=-1, keepdims=True) + 1e-8)
    evormd_attn = evormd_attn / (evormd_attn.sum(axis=-1, keepdims=True) + 1e-8)

    has_m6a = np.sum(sites == M6A_LABEL, axis=1) > 0
    if not np.all(has_m6a):
        missing = np.where(~has_m6a)[0]
        print(f"  Warning: {len(missing)} sequences have no m6A sites: {missing}")

    print(f"\nGenerating {n_seq} figures (PDF + PNG)...")
    for rank in range(n_seq):
        path = visualize_sequence(
            seq_idx=indices[rank],
            rank=rank,
            mrmodn_attn=mrmodn_attn[rank, M6A_INDEX, :],
            multirm_attn=multirm_attn[rank, M6A_INDEX, :],
            modx_attn=modx_attn[rank, M6A_INDEX, :],
            evormd_attn=evormd_attn[rank, M6A_INDEX, :],
            sites_1001=sites[rank],
            output_dir=output_dir,
        )
        print(f"  [{rank+1}/{n_seq}] {path}")

    print(f"\nSaved {n_seq} figures (PDF + PNG) to {os.path.abspath(output_dir)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mrmodn',  type=str, default='npy/mrmodn_full_atten.npz')
    parser.add_argument('--multirm', type=str, default='npy/multirm_segmented_atten.npz')
    parser.add_argument('--modx',    type=str, default='npy/modx_full_atten.npz')
    parser.add_argument('--evormd',  type=str, default='npy/evormd_segmented_atten.npz')
    parser.add_argument('--output_dir', type=str, default='figs_atten/attention_comparison')
    args = parser.parse_args()
    main(args.mrmodn, args.multirm, args.modx, args.evormd, args.output_dir)
