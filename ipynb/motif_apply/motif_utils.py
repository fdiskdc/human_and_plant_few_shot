#!/usr/bin/env python3
"""
motif_utils.py — Shared motif analysis utilities for localized IG motif pipeline.

Ported / adapted from:
  - /home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/util_cm_treex_local_fixed_pwm.py
  - /home/dc/vscode/vscode20260406/DtreeInMultiRM/draw_motif/utils.py
"""

import os
import subprocess
import numpy as np


# ---------------------------------------------------------------------------
# Sliding-window selection (highest_x / highest_score)
# ---------------------------------------------------------------------------

def highest_score(a, w):
    """Return (best_sum, best_start, best_end_inclusive) for a sliding window
    of width *w* over 1-D array *a*.
    """
    assert len(a) >= w
    best = -20000
    best_idx_start = 0
    best_idx_end = 0
    for i in range(len(a) - w + 1):
        tmp = np.sum(a[i:i + w])
        if tmp > best:
            best = tmp
            best_idx_start = i
            best_idx_end = i + w - 1
    return best, best_idx_start, best_idx_end


def highest_x(a, w, p=1, top_k=5):
    """Select up to *top_k* non-overlapping (with padding *p*) windows of
    width *w* from 1-D score array *a*.

    Returns
    -------
    result : dict
        ``{rank: (score, start_inclusive, end_inclusive), ...}``
        Ranks are 1-based.
    """
    lists = [{k: v for (k, v) in zip(range(len(a)), a)}]
    result = {}
    max_idx = len(a) - 1
    count = 1
    condition = [True]
    while any(con is True for con in condition) and count <= top_k:
        starts, ends, bests = [], [], []
        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())
            start_idx = idx[0]
            if len(values) >= w:
                highest, highest_idx_start, highest_idx_end = highest_score(values, w)
                starts.append(highest_idx_start + start_idx)
                ends.append(highest_idx_end + start_idx)
                bests.append(highest)
        if len(bests) == 0:
            break
        best_idx = max(zip(bests, range(len(bests))))[1]
        cut_value = bests[best_idx]
        cut_idx_start = max(starts[best_idx] - p, 0)
        cut_idx_end = min(ends[best_idx] + p, max_idx)
        result[count] = (cut_value, starts[best_idx], ends[best_idx])
        copy = lists.copy()
        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())
            start_idx_e, end_idx_e = idx[0], idx[-1]
            if len(values) < w:
                copy.remove(ele)
            else:
                if (cut_idx_end < start_idx_e) or (cut_idx_start > end_idx_e):
                    pass
                elif (cut_idx_start < start_idx_e) and (cut_idx_end >= start_idx_e):
                    copy.remove(ele)
                    values = values[cut_idx_end - start_idx_e + 1:]
                    idx = idx[cut_idx_end - start_idx_e + 1:]
                    ele = {k: v for (k, v) in zip(idx, values)}
                    if ele != {}:
                        copy.append(ele)
                elif (cut_idx_start >= start_idx_e) and (cut_idx_end <= end_idx_e):
                    copy.remove(ele)
                    values_1 = values[:cut_idx_start - start_idx_e]
                    idx_1 = idx[:cut_idx_start - start_idx_e]
                    ele_1 = {k: v for (k, v) in zip(idx_1, values_1)}
                    values_2 = values[cut_idx_end - start_idx_e + 1:]
                    idx_2 = idx[cut_idx_end - start_idx_e + 1:]
                    ele_2 = {k: v for (k, v) in zip(idx_2, values_2)}
                    if ele_1 != {}:
                        copy.append(ele_1)
                    if ele_2 != {}:
                        copy.append(ele_2)
                elif (cut_idx_start <= end_idx_e) and (cut_idx_end > end_idx_e):
                    copy.remove(ele)
                    values = values[:cut_idx_start - start_idx_e]
                    idx = idx[:cut_idx_start - start_idx_e]
                    ele = {k: v for (k, v) in zip(idx, values)}
                    if ele != {}:
                        copy.append(ele)
        lists = copy
        count += 1
        condition = [len(i) >= w for i in lists]
    return result


# ---------------------------------------------------------------------------
# PFM / PWM / one-hot helpers
# ---------------------------------------------------------------------------

def pfm(seqs):
    """Position frequency matrix from a list of equal-length strings.
    Returns (4, L).  Rows: A, C, G, T/U."""
    length = len(seqs[0])
    out = np.zeros((4, length))
    for seq in seqs:
        for i in range(length):
            c = seq[i]
            if c == 'A':
                out[0, i] += 1
            elif c == 'C':
                out[1, i] += 1
            elif c == 'G':
                out[2, i] += 1
            elif c in ('T', 'U'):
                out[3, i] += 1
            # gap '-' → skip
    return out


def pwm_from_pfm(pfm_mat, pseudocount=1.0):
    """Normalise a (4, L) PFM to a probability PWM with pseudocount."""
    pfm_mat = np.asarray(pfm_mat, dtype=float)
    col_totals = np.sum(pfm_mat, axis=0, keepdims=True)
    p = (pfm_mat + pseudocount) / (col_totals + 4.0 * pseudocount)
    return p


def to_onehot(seq):
    """Convert a nucleotide string to a (4, L) one-hot matrix."""
    length = len(seq)
    out = np.zeros((4, length))
    for i in range(length):
        c = seq[i]
        if c == 'A':
            out[0, i] = 1
        elif c == 'C':
            out[1, i] = 1
        elif c == 'G':
            out[2, i] = 1
        elif c in ('T', 'U'):
            out[3, i] = 1
        elif c == '-':
            out[:, i] = 0.25
    return out


# ---------------------------------------------------------------------------
# Consensus motif (DBSCAN clustering + PWM)
# ---------------------------------------------------------------------------

def cal_consensus_motif_2(seqs, scores, eps=0.3, min_samples=10, random_state=666):
    """Cluster *aligned* sequences with DBSCAN on UMAP-reduced one-hot,
    then compute per-cluster PWMs sorted by mean IG score.

    Parameters
    ----------
    seqs : list[str] or ndarray
        Aligned motif-window sequences.
    scores : list[float] or ndarray
        Importance scores for each sequence.
    eps : float
        DBSCAN neighborhood radius on the 2D UMAP/PCA embedding.
    min_samples : int
        DBSCAN minimum samples per dense neighborhood.
    random_state : int
        Random seed for UMAP/PCA reduction.

    Returns
    -------
    consensus_motif : ndarray (n_clusters, 4, L)
    ig_score : list[float]
    """
    from sklearn.cluster import DBSCAN
    try:
        import umap as umap_lib
        _has_umap = True
    except ImportError:
        _has_umap = False

    if isinstance(seqs, list):
        seqs = np.array(seqs)
    if isinstance(scores, list):
        scores = np.array(scores)

    data = [to_onehot(s).T.flatten() for s in seqs]
    import pandas as pd
    df = pd.DataFrame(data=data, index=list(range(len(seqs))))

    if _has_umap:
        Y = umap_lib.UMAP(n_components=2, random_state=random_state).fit_transform(df)
    else:
        from sklearn.decomposition import PCA
        Y = PCA(n_components=2).fit_transform(df)

    min_samples = max(2, int(min_samples))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(Y)
    class_labels = clustering.labels_

    valid_labels = np.unique(class_labels[class_labels != -1])
    if len(valid_labels) == 0:
        class_labels = np.zeros_like(class_labels)

    index_dict, scores_dict = {}, {}
    for label in np.unique(class_labels):
        idx_list = list(df.index[class_labels == label])
        index_dict[label] = idx_list
        scores_dict[label] = float(np.mean(scores[idx_list])) if len(idx_list) > 0 else 0.0

    index_sorted = sorted(scores_dict, key=scores_dict.__getitem__, reverse=True)

    seqs_dict = {}
    for label in index_sorted:
        seqs_dict[label] = seqs[index_dict[label]]

    pwm_weights, ig_score = [], []
    for idx in index_sorted:
        pwm_weights.append(np.expand_dims(pwm_from_pfm(pfm(seqs_dict[idx])), axis=0))
        ig_score.append(scores_dict[idx])

    consensus_motif = np.concatenate(pwm_weights, axis=0)
    return consensus_motif, ig_score


# ---------------------------------------------------------------------------
# MEME export
# ---------------------------------------------------------------------------

def export_to_meme(pwm_list, output_path, motif_names=None, bg_frequencies=None):
    """Export a list of PWM matrices (each L×4, rows=positions, cols=ACGT)
    to MEME format.
    """
    if bg_frequencies is None:
        bg_frequencies = [0.295, 0.205, 0.205, 0.295]
    if motif_names is None:
        motif_names = [f"Model_Motif_{i+1}" for i in range(len(pwm_list))]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write(f"Background letter frequencies\n")
        f.write(f"A {bg_frequencies[0]:.3f} C {bg_frequencies[1]:.3f} "
                f"G {bg_frequencies[2]:.3f} T {bg_frequencies[3]:.3f}\n\n")
        for i, pwm_mat in enumerate(pwm_list):
            pwm_mat = np.asarray(pwm_mat, dtype=float)
            # Ensure shape (L, 4) – if (4, L), transpose
            if pwm_mat.shape[0] == 4 and pwm_mat.shape[1] != 4:
                pwm_mat = pwm_mat.T
            L = pwm_mat.shape[0]
            # Normalise rows
            row_sums = pwm_mat.sum(axis=1, keepdims=True)
            pwm_norm = np.where(row_sums > 0, pwm_mat / row_sums, 0.25)
            f.write(f"MOTIF {motif_names[i]}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {L} nsites= 20 "
                    f"E= 0\n")
            for row in pwm_norm:
                f.write(f"  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# STREME / Tomtom wrappers
# ---------------------------------------------------------------------------

def run_streme(output_dir, positive_fasta, negative_fasta=None,
               minw=5, maxw=15, dna=False, rna=False, conda_env='meme_env'):
    """Run STREME motif discovery via conda environment."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'conda', 'run', '-n', conda_env,
        'streme',
        '--p', positive_fasta,
        '--minw', str(minw),
        '--maxw', str(maxw),
        '--oc', output_dir,
    ]
    if negative_fasta:
        cmd.extend(['--n', negative_fasta])
    if dna:
        cmd.append('--dna')
    if rna:
        cmd.append('--rna')
    print(f"[STREME] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[STREME] STDERR:\n{result.stderr}")
        raise RuntimeError(f"STREME failed (rc={result.returncode})")
    print("[STREME] completed successfully")
    return output_dir


def run_tomtom_validation(query_meme, target_meme, output_dir,
                          conda_env='meme_env'):
    """Run Tomtom to compare *query_meme* against *target_meme*."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'conda', 'run', '-n', conda_env,
        'tomtom',
        '-no-ssc',
        '-dist', 'pearson',
        '-oc', output_dir,
        query_meme,
        target_meme,
    ]
    print(f"[Tomtom] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Tomtom] STDERR:\n{result.stderr}")
        raise RuntimeError(f"Tomtom failed (rc={result.returncode})")
    print("[Tomtom] completed successfully")
    tsv_path = os.path.join(output_dir, 'tomtom.tsv')
    return tsv_path


# ---------------------------------------------------------------------------
# Motif logo drawing (Liquid Glass style, ported from old project)
# ---------------------------------------------------------------------------

def trim(motif, threshold=0.2):
    """Trim edge positions with low information content. motif: (L, 4)"""
    motif = np.array(motif)
    ic = np.log2((motif + 1e-10) / 0.25) * motif
    total_ic = np.sum(ic, axis=1)
    above = np.where(total_ic > threshold)[0]
    if len(above) == 0:
        return motif
    return motif[above[0]:above[-1] + 1]


def draw_motif_logos(consensus_motif, ig_scores, mod_name, res_dir=None,
                     filename=None, save_only=True, file_format='png',
                     p_values=None, motif_labels=None):
    """Draw motif logos using logomaker with Liquid Glass visual style.

    Parameters
    ----------
    consensus_motif : ndarray (n_motifs, 4, L)
    ig_scores : list[float]
    p_values : list[float | str | None]
        Optional Tomtom/STREME comparison p-values to show in titles. When
        provided, titles display p-value instead of IG score.
    motif_labels : list[str] | None
        Optional per-panel labels, e.g. original MEME/Tomtom query IDs such as
        Model_Motif_17. When omitted, panels are labeled Motif 1..N.
    mod_name : str
    res_dir : str
    filename : str  (without extension)
    file_format : 'png' or 'pdf'
    """
    try:
        import logomaker
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        from matplotlib.colors import to_rgba
    except ImportError as e:
        print(f"[draw_motif_logos] missing dependency: {e}")
        return

    n_motifs = consensus_motif.shape[0]
    if save_only and res_dir is None:
        return

    os.makedirs(res_dir, exist_ok=True)
    output_path = os.path.join(res_dir, f"{filename or mod_name}_motif.{file_format}")

    n_cols = min(4, n_motifs)
    n_rows = (n_motifs + n_cols - 1) // n_cols

    # Liquid Glass Color Scheme — Morandi colors with transparency
    color_scheme = {
        'A': to_rgba('#9BA4B5', 0.80),   # Muted blue-gray
        'C': to_rgba('#C8B8C0', 0.80),   # Muted mauve/pink-gray
        'G': to_rgba('#A8B5A2', 0.80),   # Muted sage green-gray
        'U': to_rgba('#D4C4B0', 0.80),   # Muted sand/beige-gray
    }

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.5 * n_rows),
                             facecolor='#F5F7FA')
    if n_motifs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_motifs):
        ax = axes[i]
        motif_label = (str(motif_labels[i])
                       if motif_labels is not None and i < len(motif_labels)
                       else f"Motif {i+1}")
        pwm = consensus_motif[i].T          # (L, 4)
        pwm_trimmed = trim(pwm, threshold=0.1)
        df = pd.DataFrame(pwm_trimmed, columns=['A', 'C', 'G', 'U'])

        ax.set_facecolor('#EBEDF0')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        try:
            logo = logomaker.Logo(df, ax=ax, color_scheme=color_scheme)
            for glyph in logo.ax.get_children():
                if hasattr(glyph, 'get_path') or isinstance(glyph, plt.Polygon):
                    shadow = path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.3)
                    highlight = path_effects.Stroke(linewidth=1.5, foreground='white', alpha=0.5)
                    glow = path_effects.Stroke(linewidth=2.5, foreground='#8B9DAD', alpha=0.2)
                    glass_effects = [glow, shadow, path_effects.Normal(), highlight]
                    try:
                        glyph.set_path_effects(glass_effects)
                    except (AttributeError, TypeError):
                        pass
                elif isinstance(glyph, plt.Text):
                    text_glow = path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.3)
                    try:
                        glyph.set_path_effects([text_glow])
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            print(f"Warning: Could not draw motif {motif_label}: {e}")
            ax.set_facecolor('#EBEDF0')
            ax.text(0.5, 0.5, f"{motif_label}\n(Error)", ha='center', va='center',
                    fontsize=12, color='#5A6B7C', fontfamily='sans-serif')

        if p_values is not None and i < len(p_values):
            pv = p_values[i]
            if pv is None or pv == "":
                value_str = "  (p=NA)"
            else:
                try:
                    value_str = f"  (p={float(pv):.2e})"
                except (TypeError, ValueError):
                    value_str = f"  (p={pv})"
        else:
            value_str = f"  (score={ig_scores[i]:.4f})" if i < len(ig_scores) else ""
        ax.set_title(f"{mod_name} {motif_label}{value_str}",
                     fontsize=13, fontweight='bold', fontfamily='sans-serif',
                     color='#3D4752', pad=12, loc='left')
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#D1D9E6')
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_alpha(0.5)

    for i in range(n_motifs, len(axes)):
        axes[i].axis('off')
        axes[i].set_facecolor('#F5F7FA')

    plt.tight_layout(pad=1.5, w_pad=1.2, h_pad=1.8)
    save_kwargs = {'bbox_inches': 'tight', 'facecolor': '#F5F7FA', 'edgecolor': 'none'}
    if file_format in ('png', 'jpg', 'jpeg', 'tiff'):
        save_kwargs['dpi'] = 200
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    print(f"[MotifLogo] saved to {output_path}")


# ---------------------------------------------------------------------------
# R alignment helper
# ---------------------------------------------------------------------------

def run_alignment_r(input_fasta, output_fasta, r_script_path=None):
    """Call an R script for sequence alignment.

    If *r_script_path* is None, skip alignment and just copy the input.
    """
    if r_script_path is None or not os.path.isfile(r_script_path):
        print(f"[Alignment] R script not found or not provided; skipping.")
        return input_fasta
    cmd = ['Rscript', r_script_path, input_fasta, output_fasta]
    print(f"[Alignment] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Alignment] STDERR:\n{result.stderr}")
        return input_fasta
    return output_fasta
