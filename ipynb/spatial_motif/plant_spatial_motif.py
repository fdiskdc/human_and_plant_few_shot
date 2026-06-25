#!/usr/bin/env python3
"""
plant_spatial_motif.py
Spatial motif analysis for plant RNA modification model.
Combines localized IG motif analysis (from plant_localized_ig_motif.py) with
latent-space clustering and sequence logo generation (from SpatialMotif_nobackground.py).

双管线架构:
  管线 A (默认): localized IG → STREME/Tomtom 分析 (原 plant_localized_ig_motif.py 逻辑)
  管线 B (--spatial_motif): GCN嵌入 → PCA+KMeans聚类 → 全局 IG 归因 → 序列标志图

输出目录: ipynb/spatial_motif/output/plant/
"""

import argparse, json, os, sys, time, warnings, csv, re
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 项目路径初始化
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.main_model import RNA_ClassQuery_Model
from dataset.plant import PlantDataset, MOD_NAMES, LABEL_MAPPING

# 中英文：尝试导入 motif_apply 下的工具函数；如果当前目录结构不同则回退到直接 import。
try:
    from ipynb.motif_apply.motif_utils import (
        highest_x, cal_consensus_motif_2, export_to_meme,
        draw_motif_logos,
        run_streme, run_tomtom_validation, run_alignment_r,
    )
    from ipynb.motif_apply.apply_train import (
        ensure_converged_checkpoint,
        DEFAULT_INIT_CHECKPOINT as _DEFAULT_INIT_CKPT,
        DEFAULT_PLANT_CONFIG,
    )
except ImportError:
    # 回退：将 motif_apply 目录加入 sys.path
    _motif_apply_dir = os.path.join(PROJECT_ROOT, 'ipynb', 'motif_apply')
    if _motif_apply_dir not in sys.path:
        sys.path.insert(0, _motif_apply_dir)
    from motif_utils import (
        highest_x, cal_consensus_motif_2, export_to_meme,
        draw_motif_logos,
        run_streme, run_tomtom_validation, run_alignment_r,
    )
    from apply_train import (
        ensure_converged_checkpoint,
        DEFAULT_INIT_CHECKPOINT as _DEFAULT_INIT_CKPT,
        DEFAULT_PLANT_CONFIG,
    )

# Clustering imports (for spatial motif pipeline)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Optional: logomaker
try:
    import logomaker
    HAS_LOGOMAKER = True
except ImportError:
    HAS_LOGOMAKER = False
    print("Warning: logomaker not installed. Install with: pip install logomaker")

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Warning: captum not installed. Install with: pip install captum")

# Optional: matplotlib for spatial motif logos
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ===========================================================================
# RNA 字母表与 one-hot 编码 (移植自 plant_localized_ig_motif.py)
# ===========================================================================

def _create_byte_to_onehot_mapping():
    mapping = np.zeros((256, 4), dtype=np.float32)
    mapping[65] = [1., 0., 0., 0.]   # 'A'
    mapping[97] = [1., 0., 0., 0.]   # 'a'
    mapping[67] = [0., 1., 0., 0.]   # 'C'
    mapping[99] = [0., 1., 0., 0.]   # 'c'
    mapping[71] = [0., 0., 1., 0.]   # 'G'
    mapping[103] = [0., 0., 1., 0.]  # 'g'
    mapping[84] = [0., 0., 0., 1.]   # 'T'
    mapping[116] = [0., 0., 0., 1.]  # 't'
    mapping[85] = [0., 0., 0., 1.]   # 'U'
    mapping[117] = [0., 0., 0., 1.]  # 'u'
    mapping[78] = [0., 0., 0., 0.]   # 'N'
    mapping[110] = [0., 0., 0., 0.]  # 'n'
    return mapping

_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()


def normalize_rna_seq(seq):
    return str(seq).upper().replace('T', 'U')


def bytes_to_onehot(row):
    raw = np.asarray(row)
    if raw.dtype.kind == 'S':
        if raw.ndim == 0:
            byte_array = np.frombuffer(raw.item(), dtype=np.uint8)
        else:
            byte_array = raw.view(np.uint8).reshape(-1)
    else:
        if raw.ndim == 0:
            byte_array = np.frombuffer(raw.item(), dtype=np.uint8)
        else:
            byte_array = raw.astype(np.uint8, copy=False).reshape(-1)
    return _BYTE_TO_ONEHOT_MAPPING[byte_array].copy()


def bytes_to_seqstr(row):
    raw = np.asarray(row)
    if raw.ndim == 0:
        seq = raw.item().decode('ascii') if isinstance(raw.item(), bytes) else str(raw.item())
    else:
        seq = b''.join(raw[i] for i in range(len(raw))).decode('ascii')
    return normalize_rna_seq(seq)


# ===========================================================================
# 修饰名称映射
# ===========================================================================
MOD_NAME_TO_CLASS = {v: k for k, v in MOD_NAMES.items()}
PLANT_DEFAULT_MOD_INDICES = (5, 8, 9)  # Y, m5C, m6A
PLANT_DEFAULT_MODS = ','.join(MOD_NAMES[i] for i in PLANT_DEFAULT_MOD_INDICES)

# DBSCAN eps defaults (Plant 仅含 Y, m5C, m6A)
offset = 16
DEFAULT_CLUSTER_EPS_BY_MOD = {
    'Y': 2.1/offset,
    'm5C': 1.9/offset,
    'm6A': 2.0/offset,
}


def parse_mod_list(mod_str):
    mods = []
    for name in mod_str.split(','):
        name = name.strip()
        if name not in MOD_NAME_TO_CLASS:
            warnings.warn(f"Unknown mod name '{name}', skipping.")
            continue
        mods.append((name, MOD_NAME_TO_CLASS[name]))
    return mods


# ===========================================================================
# 配置与模型加载
# ===========================================================================

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_model(cfg, checkpoint_path, device):
    mc = cfg['model']
    model = RNA_ClassQuery_Model(
        cnn_hidden_dim=mc.get('cnn_hidden_dim', 64),
        cnn_kernel_sizes=tuple(mc.get('cnn_kernel_sizes', [1, 3, 5, 7])),
        cnn_dropout=mc.get('cnn_dropout', 0.1),
        gcn_hidden_dim=mc.get('gcn_hidden_dim', 128),
        gcn_out_channels=mc.get('gcn_out_channels', 128),
        gcn_num_layers=mc.get('gcn_num_layers', 3),
        gcn_dropout=mc.get('gcn_dropout', 0.3),
        num_classes=mc.get('num_classes', 12),
        num_attn_heads=mc.get('num_attn_heads', 4),
        attn_dropout=mc.get('attn_dropout', 0.1),
        use_simple_pooling=mc.get('use_simple_pooling', False),
        use_hierarchical=mc.get('use_hierarchical', False),
        use_layer_norm=mc.get('use_layer_norm', True),
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ===========================================================================
# 管线 A: Localized IG motif 分析 (原 plant_localized_ig_motif.py 逻辑)
# ===========================================================================

def collect_events_from_dataset(dataset, indices, target_mods):
    events = []
    class_to_modidx = {v: k for k, v in LABEL_MAPPING.items()}
    for idx in indices:
        y_site = np.asarray(dataset.full_labels[idx])
        for mod_name, class_idx in target_mods:
            mod_index = class_to_modidx[class_idx]
            for pos in np.where(y_site == mod_index)[0]:
                if 25 <= pos <= 975:
                    events.append((int(idx), int(pos), class_idx))
    return events


def generate_negatives_matched_offsets(records, zero_seq_path, motif_w,
                                       negative_ratio=1.0, seed=666,
                                       allow_no_negative=False):
    zero_seq = np.load(zero_seq_path, mmap_mode='r')
    n_zero = zero_seq.shape[0]
    if n_zero == 0:
        if allow_no_negative:
            return [], [], []
        raise RuntimeError(f"No zero sequences in {zero_seq_path}")
    rng = np.random.RandomState(seed)
    n_pos = len(records)
    n_neg = max(1, int(round(n_pos * negative_ratio)))
    neg_seqs, neg_hdrs, neg_records = [], [], []
    for i in range(n_neg):
        rec = records[i % n_pos]
        motif_start_51 = rec['motif_start_51']
        m51e = motif_start_51 + motif_w
        if m51e > 51:
            m51e = 51
        abs_start = 475 + motif_start_51
        abs_end = 475 + m51e
        if abs_end > 1001:
            abs_end = 1001
        zidx = rng.randint(0, n_zero)
        z_seq = bytes_to_seqstr(zero_seq[zidx])
        sub = z_seq[abs_start:abs_end]
        neg_seqs.append(sub)
        neg_hdrs.append(f"neg_{zidx}_{abs_start}_{abs_end}")
        neg_records.append(dict(
            zero_idx=int(zidx), abs_start=int(abs_start), abs_end=int(abs_end),
            motif_start_51=int(motif_start_51), seq=sub,
            paired_sample_idx=rec.get('sample_idx', ''),
            paired_site_pos=rec.get('site_pos', ''),
        ))
    return neg_seqs, neg_hdrs, neg_records


class LocalizedIGWrapper(torch.nn.Module):
    def __init__(self, model, single_edge_index, seq_len, class_idx):
        super().__init__()
        self.model = model
        self.register_buffer('single_edge_index', single_edge_index)
        self.seq_len = seq_len
        self.class_idx = class_idx

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        bs = x.size(0)
        offsets = torch.arange(bs, device=x.device).view(1, -1, 1) * self.seq_len
        ei = self.single_edge_index.unsqueeze(1)
        ei = ei + offsets
        ei = ei.permute(1, 0, 2).reshape(2, -1)
        bv = torch.arange(bs, device=x.device).repeat_interleave(self.seq_len)
        out = self.model(x, ei, bv)
        logits_12 = out[0] if isinstance(out, tuple) else out
        return logits_12[:, self.class_idx].view(bs, 1)


def compute_localized_ig(model, input_full, edge_index, batch_vec,
                         class_idx, site_pos, ig_steps=32, device='cuda'):
    from captum.attr import IntegratedGradients as IG
    input_full = input_full.to(device)
    edge_index_dev = edge_index.to(device)
    batch_vec_dev = batch_vec.to(device)
    with torch.no_grad():
        out = model(input_full.unsqueeze(0), edge_index_dev, batch_vec_dev)
        logits_12 = out[0] if isinstance(out, tuple) else out
        logit_val = (logits_12[0, class_idx] if logits_12.dim() == 2
                     else logits_12[class_idx]).item()
    pred_prob = torch.sigmoid(torch.tensor(logit_val)).item()
    s0, s1 = site_pos - 25, site_pos + 26
    baseline = input_full.clone()
    baseline[s0:s1, :] = 0.0
    wrapper = LocalizedIGWrapper(model, edge_index_dev, 1001, class_idx).to(device)
    wrapper.eval()
    ig = IG(wrapper)
    attr = ig.attribute(input_full.unsqueeze(0), baselines=baseline.unsqueeze(0),
                        n_steps=ig_steps, target=0)
    attr_np = attr.squeeze(0).detach().cpu().numpy()
    score_all = np.abs(attr_np).sum(axis=-1)
    score_51 = score_all[s0:s1]
    assert score_51.shape[0] == 51
    return score_51, pred_prob, logit_val


def extract_windows(score_51, seq_str, site_pos, motif_w=8, top_k=5, skip_n=True):
    w51s, w51e = site_pos - 25, site_pos + 26
    seq51 = seq_str[w51s:w51e]
    assert len(seq51) == 51
    result = highest_x(score_51, w=motif_w, p=1, top_k=top_k)
    windows = []
    for rank in sorted(result.keys()):
        score, si, ei = result[rank]
        m51s, m51e = si, ei + 1
        if m51e - m51s != motif_w:
            continue
        s8 = seq51[m51s:m51e]
        if skip_n and ('N' in s8 or 'N' in seq51):
            continue
        windows.append(dict(
            window51_start=w51s, window51_end=w51e,
            motif_start=w51s + m51s, motif_end=w51s + m51e,
            motif_start_51=m51s, motif_end_51=m51e,
            window_score=float(score), seq_8nt=s8, seq_51nt=seq51, rank=rank))
    return windows


_WINDOWS_HDR = (
    "sample_idx\tsite_pos\tclass_idx\tmod_name\t"
    "window51_start\twindow51_end\tmotif_start\tmotif_end\t"
    "window_score\tseq_8nt\tseq_51nt\tmotif_start_51\tmotif_end_51\t"
    "rank\tpred_prob\tlogit\n"
)

_NEGATIVE_WINDOWS_HDR = (
    "zero_idx\tabs_start\tabs_end\tmotif_start_51\tseq\t"
    "paired_sample_idx\tpaired_site_pos\n"
)


def write_windows_tsv(path, records):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_WINDOWS_HDR)
        for r in records:
            f.write(f"{r['sample_idx']}\t{r['site_pos']}\t{r['class_idx']}\t"
                    f"{r['mod_name']}\t{r['window51_start']}\t{r['window51_end']}\t"
                    f"{r['motif_start']}\t{r['motif_end']}\t"
                    f"{r['window_score']:.6f}\t{r['seq_8nt']}\t{r['seq_51nt']}\t"
                    f"{r['motif_start_51']}\t{r['motif_end_51']}\t"
                    f"{r['rank']}\t{r['pred_prob']:.6f}\t{r['logit']:.6f}\n")


def write_negative_windows_tsv(path, neg_records):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_NEGATIVE_WINDOWS_HDR)
        for r in neg_records:
            f.write(f"{r['zero_idx']}\t{r['abs_start']}\t{r['abs_end']}\t"
                    f"{r['motif_start_51']}\t{r['seq']}\t"
                    f"{r['paired_sample_idx']}\t{r['paired_site_pos']}\n")


def write_fasta(path, seqs, headers=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for i, s in enumerate(seqs):
            h = headers[i] if headers else f"seq_{i}"
            f.write(f">{h}\n{normalize_rna_seq(s)}\n")


def rewrite_meme_alphabet_to_rna(path):
    if not path or not os.path.isfile(path):
        return
    out_lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('ALPHABET='):
                out_lines.append('ALPHABET= ACGU\n')
            elif re.match(r'^A\s+[-+0-9.eE]+\s+C\s+', line):
                out_lines.append(re.sub(r'\bT\s+([-+0-9.eE]+)', r'U \1', line))
            else:
                out_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(out_lines)


def _consensus_from_pwm_rows(rows):
    letters = ['A', 'C', 'G', 'U']
    consensus = []
    for row in rows:
        if not row:
            continue
        consensus.append(letters[int(np.argmax(row))])
    return ''.join(consensus)


def parse_meme_like_motifs(path, source):
    motifs = []
    if not path or not os.path.isfile(path):
        return motifs
    current = None
    collecting = False
    remaining_rows = 0
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith('MOTIF '):
                if current is not None:
                    current['consensus_from_pwm'] = _consensus_from_pwm_rows(current.pop('_rows', []))
                    if not current.get('consensus'):
                        current['consensus'] = current['consensus_from_pwm']
                    motifs.append(current)
                motif_text = line.split(None, 1)[1]
                motif_id = motif_text.split()[0]
                consensus = ''
                if source == 'streme' and '-' in motif_id:
                    consensus = motif_id.split('-', 1)[1]
                current = {
                    'source': source, 'motif_id': motif_id, 'motif_name': motif_text,
                    'consensus': consensus, 'width': None, 'nsites': None,
                    'p_value': None, '_rows': [],
                }
                collecting = False
                remaining_rows = 0
                continue
            if current is None:
                continue
            if line.startswith('letter-probability'):
                width_match = re.search(r'\bw=\s*(\d+)', line)
                nsites_match = re.search(r'\bnsites=\s*(\d+)', line)
                p_match = re.search(r'\bP=\s*([0-9.eE+-]+)', line)
                if width_match:
                    current['width'] = int(width_match.group(1))
                    remaining_rows = current['width']
                    collecting = True
                if nsites_match:
                    current['nsites'] = int(nsites_match.group(1))
                if p_match:
                    current['p_value'] = p_match.group(1)
                continue
            if collecting and remaining_rows > 0 and line:
                parts = line.split()
                try:
                    row = [float(x) for x in parts[:4]]
                except ValueError:
                    continue
                current['_rows'].append(row)
                remaining_rows -= 1
                if remaining_rows == 0:
                    collecting = False
    if current is not None:
        current['consensus_from_pwm'] = _consensus_from_pwm_rows(current.pop('_rows', []))
        if not current.get('consensus'):
            current['consensus'] = current['consensus_from_pwm']
        motifs.append(current)
    return motifs


def parse_tomtom_tsv(path):
    if not path or not os.path.isfile(path):
        return []
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader((line for line in f if not line.startswith('#')),
                                delimiter='\t')
        for row in reader:
            if row.get('Query_ID'):
                rows.append(row)
    return rows


def best_tomtom_pvalues_by_query(tomtom_rows, n_motifs):
    best = {}
    for row in tomtom_rows:
        qid = row.get('Query_ID', '')
        pval = row.get('p-value', '')
        try:
            p_float = float(pval)
        except (TypeError, ValueError):
            continue
        if qid not in best or p_float < best[qid]:
            best[qid] = p_float
    return [best.get(f'Model_Motif_{i+1}') for i in range(n_motifs)]


def filter_motifs_with_pvalues(consensus_motif, ig_scores, p_values):
    keep = []
    for idx, pv in enumerate(p_values):
        if pv is None or pv == '':
            continue
        try:
            pv_float = float(pv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(pv_float):
            keep.append(idx)
    if not keep:
        return None, [], [], []
    filtered_motifs = consensus_motif[keep]
    filtered_scores = [ig_scores[i] for i in keep]
    filtered_pvalues = [p_values[i] for i in keep]
    return filtered_motifs, filtered_scores, filtered_pvalues, keep


def remove_logo_files(mod_dir, mod_name):
    for ext in ('png', 'pdf'):
        path = os.path.join(mod_dir, f"{mod_name}_motif_motif.{ext}")
        if os.path.exists(path):
            os.remove(path)


def build_mod_summary(mod_name, mod_dir):
    streme_txt = os.path.join(mod_dir, 'streme_out', 'streme.txt')
    model_meme = os.path.join(mod_dir, 'model_motifs.meme')
    tomtom_tsv = os.path.join(mod_dir, 'tomtom_out', 'tomtom.tsv')
    streme_motifs = parse_meme_like_motifs(streme_txt, source='streme')
    model_motifs = parse_meme_like_motifs(model_meme, source='model')
    tomtom_rows = parse_tomtom_tsv(tomtom_tsv)
    streme_by_id = {m['motif_id']: m for m in streme_motifs}
    model_by_id = {m['motif_id']: m for m in model_motifs}
    match_rows = []
    matched_model_ids = set()
    matched_streme_ids = set()
    for row in tomtom_rows:
        qid = row.get('Query_ID', '')
        tid = row.get('Target_ID', '')
        model_m = model_by_id.get(qid, {})
        streme_m = streme_by_id.get(tid, {})
        matched_model_ids.add(qid)
        matched_streme_ids.add(tid)
        match_rows.append({
            'mod_name': mod_name, 'matched': True,
            'model_motif_id': qid,
            'model_consensus': row.get('Query_consensus') or model_m.get('consensus', ''),
            'streme_motif_id': tid,
            'streme_consensus': row.get('Target_consensus') or streme_m.get('consensus', ''),
            'streme_p_value': streme_m.get('p_value', ''),
            'tomtom_p_value': row.get('p-value', ''),
            'tomtom_E_value': row.get('E-value', ''),
            'tomtom_q_value': row.get('q-value', ''),
            'overlap': row.get('Overlap', ''),
            'orientation': row.get('Orientation', ''),
            'optimal_offset': row.get('Optimal_offset', ''),
        })
    for motif in model_motifs:
        if motif['motif_id'] not in matched_model_ids:
            match_rows.append({
                'mod_name': mod_name, 'matched': False,
                'model_motif_id': motif['motif_id'],
                'model_consensus': motif.get('consensus', ''),
                'streme_motif_id': '', 'streme_consensus': '', 'streme_p_value': '',
                'tomtom_p_value': '', 'tomtom_E_value': '', 'tomtom_q_value': '',
                'overlap': '', 'orientation': '', 'optimal_offset': '',
            })
    for motif in streme_motifs:
        if motif['motif_id'] not in matched_streme_ids:
            match_rows.append({
                'mod_name': mod_name, 'matched': False,
                'model_motif_id': '', 'model_consensus': '',
                'streme_motif_id': motif['motif_id'],
                'streme_consensus': motif.get('consensus', ''),
                'streme_p_value': motif.get('p_value', ''),
                'tomtom_p_value': '', 'tomtom_E_value': '', 'tomtom_q_value': '',
                'overlap': '', 'orientation': '', 'optimal_offset': '',
            })
    streme_rows = [
        {
            'mod_name': mod_name,
            'streme_motif_id': m.get('motif_id', ''),
            'streme_consensus': m.get('consensus', ''),
            'streme_consensus_from_pwm': m.get('consensus_from_pwm', ''),
            'width': m.get('width', ''), 'nsites': m.get('nsites', ''),
            'streme_p_value': m.get('p_value', ''),
        }
        for m in streme_motifs
    ]
    model_rows = [
        {
            'mod_name': mod_name,
            'model_motif_id': m.get('motif_id', ''),
            'model_consensus': m.get('consensus', ''),
            'width': m.get('width', ''), 'nsites': m.get('nsites', ''),
        }
        for m in model_motifs
    ]
    return match_rows, streme_rows, model_rows, tomtom_rows


def write_tomtom_summary_excel(output_dir, match_rows, streme_rows, model_rows):
    if not match_rows and not streme_rows and not model_rows:
        print("[Summary] no motif summary rows to write")
        return None
    import pandas as _pd
    out_path = os.path.join(output_dir, 'tomtom_summary.xlsx')
    with _pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        _pd.DataFrame(match_rows).to_excel(writer, sheet_name='tomtom_matches', index=False)
        _pd.DataFrame(streme_rows).to_excel(writer, sheet_name='streme_motifs', index=False)
        _pd.DataFrame(model_rows).to_excel(writer, sheet_name='model_motifs', index=False)
    print(f"[Summary] wrote {out_path}")
    return out_path


# ===========================================================================
# 管线 B: Spatial Motif 分析 (移植自 SpatialMotif_nobackground.py)
# ===========================================================================

# Constants
SEQ_LENGTH = 1001
NUCLEOTIDES = ['A', 'C', 'G', 'U']
MIN_REL_POS = -1000
MAX_REL_POS = 1000
REL_RANGE = (MAX_REL_POS - MIN_REL_POS) + 1  # 2001
CENTER_IDX = -MIN_REL_POS  # 1000
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
COL_TO_NUCLEOTIDE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
# 修饰位点的原始碱基映射 (m6A→A, m5C→C, Y→U 等)
CLASS_TO_NUCLEOTIDE = {
    0: 'A', 1: 'A',                          # Am, Atol → A
    2: 'C',                                  # Cm → C
    3: 'G',                                  # Gm → G
    4: 'U', 5: 'U',                          # Tm, Y → U
    6: 'C',                                  # ac4C → C
    7: 'A',                                  # m1A → A
    8: 'C',                                  # m5C → C
    9: 'A', 10: 'A',                         # m6A, m6Am → A
    11: 'G'                                  # m7G → G
}

# Morandi colors for logomaker
MORANDI_COLORS = {
    'A': (95/255, 158/255, 160/255, 0.80),
    'C': (188/255, 143/255, 143/255, 0.75),
    'G': (143/255, 188/255, 143/255, 0.80),
    'U': (218/255, 165/255, 32/255, 0.78),
}

NUC_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


class GlobalIGModelWrapper(nn.Module):
    """Wrapper for global IG attribution: returns single class logit from full sequence."""

    def __init__(self, model: RNA_ClassQuery_Model, target_class_idx: int, edge_index, batch):
        super().__init__()
        self.model = model
        self.target_class_idx = target_class_idx
        self.edge_index = edge_index
        self.batch = batch
        self.model.eval()

    def forward(self, x_flat):
        x = x_flat.view(-1, 4)
        if hasattr(self.model, 'use_hierarchical') and self.model.use_hierarchical:
            output = self.model(x, self.edge_index, self.batch, return_attention=True)
            logits_12 = output[0]
        else:
            output = self.model(x, self.edge_index, self.batch)
            if isinstance(output, tuple):
                logits_12 = output[0]
            else:
                logits_12 = output
        return logits_12[0, self.target_class_idx]


def extract_embeddings_spatial(
    model: RNA_ClassQuery_Model,
    dataset,
    indices: np.ndarray,
    device: torch.device,
    target_module_path: str = 'class_query_head.mha_12',
) -> np.ndarray:
    """Extract graph embeddings via forward hook for clustering.

    Parameters
    ----------
    model : RNA_ClassQuery_Model
    dataset : PlantDataset
    indices : np.ndarray
        Sample indices to extract embeddings for.
    device : torch.device
    target_module_path : str
        Model submodule name to hook (default: 'class_query_head.mha_12').

    Returns
    -------
    np.ndarray shape [num_samples, embedding_dim]
    """
    model.eval()
    embeddings_list = []
    idx_list = []

    activation = {}
    hook_handle = None

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation[name] = output[0].detach()
            else:
                activation[name] = output.detach()
        return hook

    target_module = None
    for name, module in model.named_modules():
        if name == target_module_path:
            target_module = module
            break

    if target_module is None:
        raise ValueError(
            f"Could not find module '{target_module_path}' in the model. "
            f"Run model.named_modules() to list available modules."
        )

    hook_handle = target_module.register_forward_hook(get_hook(target_module_path))

    print(f"\n{'='*60}")
    print(f"Extracting embeddings from '{target_module_path}'...")
    print(f"{'='*60}")

    try:
        with torch.no_grad():
            for idx in indices:
                try:
                    data = dataset[idx]
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
                    activation.clear()
                    _ = model(x, edge_index, batch)
                    node_features = activation.get(target_module_path)
                    if node_features is not None:
                        if node_features.dim() == 3:
                            sample_embedding = node_features.flatten().cpu().numpy()
                        else:
                            sample_embedding = node_features.mean(dim=0).cpu().numpy()
                        embeddings_list.append(sample_embedding)
                        idx_list.append(idx)
                    else:
                        print(f"Warning: No activation for sample {idx}")
                except Exception as e:
                    print(f"Warning: Failed to extract embedding for sample {idx}: {e}")
                    continue
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if len(embeddings_list) == 0:
        raise ValueError("No embeddings were successfully extracted!")

    embeddings = np.array(embeddings_list)
    print(f"Successfully extracted {len(embeddings)} embeddings, shape={embeddings.shape}")
    return embeddings


def cluster_samples_spatial(
    embeddings: np.ndarray,
    indices: np.ndarray,
    n_clusters: int = 5,
    pca_components: int = None,
    random_state: int = 42,
):
    """Cluster samples using PCA + KMeans on latent embeddings."""
    print(f"\n{'='*60}")
    print(f"Clustering {len(embeddings)} samples into {n_clusters} clusters...")
    print(f"{'='*60}")

    if pca_components is not None and pca_components < embeddings.shape[1]:
        print(f"Applying PCA: {embeddings.shape[1]} -> {pca_components} dimensions...")
        pca = PCA(n_components=pca_components, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.4f}")
        clustering_input = embeddings_reduced
    else:
        print("Skipping PCA (using original embeddings)...")
        clustering_input = embeddings

    print(f"Running K-Means with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(clustering_input)

    clusters_dict = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = indices[cluster_mask]
        clusters_dict[cluster_id] = cluster_indices
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} samples")

    print(f"Clustering complete. Inertia: {kmeans.inertia_:.4f}")
    return clusters_dict


def calculate_spatial_attribution(
    model: RNA_ClassQuery_Model,
    dataset,
    target_class_idx: int,
    target_indices: np.ndarray,
    device: torch.device,
    n_steps: int = 50,
    internal_batch_size: int = 512,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute global IG attribution for target samples, aligned around modification sites.

    Returns
    -------
    np.ndarray shape [2001, 4] — averaged attribution matrix (relative position x nucleotide).
    """
    original_label_id = REVERSE_LABEL_MAPPING.get(target_class_idx)
    aggregated_importance = np.zeros((REL_RANGE, 4), dtype=np.float32)

    print(f"\nComputing attribution for {len(target_indices)} samples...")
    if len(target_indices) == 0:
        print("Warning: No target indices provided!")
        return aggregated_importance

    processed_count = 0
    class_name = MOD_NAMES[target_class_idx]

    for i in range(0, len(target_indices), batch_size):
        batch_idxs = target_indices[i:i + batch_size]
        for idx in batch_idxs:
            try:
                data = dataset[idx]
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                y_site = torch.LongTensor(dataset.full_labels[idx].copy())
                batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

                anchor_indices = torch.where(y_site == original_label_id)[0].tolist()
                if len(anchor_indices) == 0:
                    continue

                model.eval()
                x_flat = x.flatten()
                x_attrib = x_flat.clone().detach().requires_grad_(True)

                sample_wrapper = GlobalIGModelWrapper(model, target_class_idx, edge_index, batch)
                sample_ig = IntegratedGradients(sample_wrapper)

                try:
                    attributions_flat = sample_ig.attribute(
                        x_attrib.unsqueeze(0),
                        n_steps=n_steps,
                        internal_batch_size=internal_batch_size,
                    )
                    attributions = attributions_flat.view(1001, 4)
                except Exception:
                    x_grad = x.clone().detach().requires_grad_(True)
                    fallback_wrapper = GlobalIGModelWrapper(model, target_class_idx, edge_index, batch)
                    output = fallback_wrapper(x_grad.flatten().unsqueeze(0))
                    output.backward()
                    attributions = x_grad.grad

                sample_attrib_matrix = attributions.cpu().detach().numpy()

                for anchor_pos in anchor_indices:
                    start_rel = MIN_REL_POS
                    end_rel = MAX_REL_POS
                    start_abs = max(0, anchor_pos + start_rel)
                    end_abs = min(1001, anchor_pos + end_rel + 1)
                    rel_idx_start = (start_abs - anchor_pos) - MIN_REL_POS
                    rel_idx_end = (end_abs - anchor_pos) - MIN_REL_POS
                    aggregated_importance[rel_idx_start:rel_idx_end, :] += sample_attrib_matrix[start_abs:end_abs, :]
                    processed_count += 1

            except Exception as e:
                print(f"Warning: Failed to process sample {idx}: {e}")
                continue

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if processed_count > 0:
        aggregated_importance /= processed_count
        print(f"Successfully processed {processed_count} samples.")
    else:
        print("Warning: No samples were successfully processed!")

    return aggregated_importance


def plot_top_k_logo(
    aggregated_importance: np.ndarray,
    class_idx: int,
    class_name: str,
    node_num: int = 10,
    output_dir: str = "motif_logo_clustered_noback",
) -> str:
    """Generate a sequence logo with Otsu-based hard-zero filtering.

    Adapted from SpatialMotif_nobackground.py.
    """
    if not HAS_LOGOMAKER or not HAS_MPL:
        print("Warning: logomaker/matplotlib not available; skipping logo generation.")
        return ""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Stage 1: Otsu per-position scoring ----
    saliency_matrix = np.abs(aggregated_importance)
    center_idx = CENTER_IDX
    num_positions = saliency_matrix.shape[0]
    position_otsu_scores = {}
    IDX_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

    for pos in range(num_positions):
        if pos == center_idx:
            continue
        base_values = saliency_matrix[pos, :].copy()
        base_sum = np.sum(base_values)
        if base_sum < 1e-9:
            continue
        normalized_values = base_values / base_sum
        sorted_indices = np.argsort(normalized_values)[::-1]
        sorted_values = normalized_values[sorted_indices]

        max_between_variance = -1.0
        best_signal_bases = []
        for split_point in [1, 2, 3]:
            signal_values = sorted_values[:split_point]
            background_values = sorted_values[split_point:]
            omega_0 = len(signal_values) / 4.0
            omega_1 = len(background_values) / 4.0
            mu_0 = np.mean(signal_values) if len(signal_values) > 0 else 0.0
            mu_1 = np.mean(background_values) if len(background_values) > 0 else 0.0
            between_variance = omega_0 * omega_1 * (mu_0 - mu_1) ** 2
            if between_variance > max_between_variance:
                max_between_variance = between_variance
                best_signal_bases = [IDX_TO_NUC[sorted_indices[i]] for i in range(split_point)]
        position_otsu_scores[pos] = (max_between_variance, best_signal_bases)

    # ---- Stage 2: Dynamic baseline filter ----
    if position_otsu_scores:
        all_variances = np.array([v[0] for v in position_otsu_scores.values()])
        baseline_noise_level = np.median(all_variances)
        valid_positions = [
            pos for pos, (variance, _) in position_otsu_scores.items()
            if variance > baseline_noise_level
        ]
    else:
        valid_positions = []

    # ---- Stage 3: Sort & top-k ----
    if valid_positions:
        valid_position_scores = [
            (pos, np.sum(saliency_matrix[pos, :])) for pos in valid_positions
        ]
        valid_position_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(node_num, len(valid_position_scores))
        top_indices = np.array([valid_position_scores[i][0] for i in range(top_k)])
    else:
        temp_scores = np.sum(saliency_matrix, axis=1)
        temp_scores[center_idx] = -1.0
        top_indices = np.argsort(temp_scores)[-node_num:]

    # ---- Stage 4: Hard zeroing ----
    all_indices = np.append(top_indices, center_idx)
    all_indices = np.sort(all_indices)
    logo_matrix_raw = saliency_matrix[all_indices]
    row_sums = np.sum(logo_matrix_raw, axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0
    logo_matrix_norm = logo_matrix_raw / row_sums

    significant_dict = {}
    for local_idx, global_pos in enumerate(all_indices):
        if global_pos == center_idx:
            significant_dict[local_idx] = ['A', 'C', 'G', 'U']
            continue
        if global_pos in position_otsu_scores:
            _, sig_bases = position_otsu_scores[global_pos]
            if sig_bases:
                significant_dict[local_idx] = sig_bases

    anchor_local_idx = np.where(all_indices == center_idx)[0][0]
    for local_idx in range(logo_matrix_norm.shape[0]):
        if local_idx == anchor_local_idx:
            continue
        if local_idx in significant_dict:
            sig_bases = significant_dict[local_idx]
            for nuc, col_idx in NUC_TO_INDEX.items():
                if nuc not in sig_bases:
                    logo_matrix_norm[local_idx, col_idx] = 0.0

    target_nuc = CLASS_TO_NUCLEOTIDE.get(class_idx, 'N')
    if target_nuc in NUC_TO_INDEX:
        target_col = NUC_TO_INDEX[target_nuc]
        logo_matrix_norm[anchor_local_idx, :] = 0
        logo_matrix_norm[anchor_local_idx, target_col] = 1.0

    logo_df = pd.DataFrame(logo_matrix_norm, columns=['A', 'C', 'G', 'U'])

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(max(10, node_num * 0.8), 6))
    fig.patch.set_facecolor('#eceff2')
    ax.set_facecolor('#eceff2')

    logo = logomaker.Logo(logo_df, ax=ax, color_scheme=MORANDI_COLORS,
                          font_name='DejaVu Sans', center_values=False)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_yticks([])
    ax.set_yticklabels([])
    logo.style_spines(spines=['left'], visible=False)

    highlight = path_effects.Stroke(linewidth=0.8, foreground=(1, 1, 1, 0.6), alpha=0.7)
    shadow = path_effects.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.4, rho=0.5)
    normal = path_effects.Normal()

    for glyph in logo.glyph_list:
        if hasattr(glyph, 'patch') and glyph.patch is not None:
            if glyph.p == anchor_local_idx:
                glyph.patch.set_path_effects([shadow, highlight, normal])
                continue
            is_significant = (glyph.p in significant_dict) and (glyph.c in significant_dict[glyph.p])
            if is_significant:
                glyph.patch.set_path_effects([shadow, highlight, normal])

    ax.set_title(f"{class_name}: Spatial Motif (Top {node_num} Context, Hard Zeroing)",
                 fontsize=30, fontfamily='sans-serif', fontweight='bold',
                 color='#2D3748', pad=15)
    ax.set_ylim(0, 1.05)

    real_rel_positions = [idx - CENTER_IDX for idx in all_indices]
    ax.set_xticks(range(len(all_indices)))
    ax.set_xticklabels(real_rel_positions,
                       rotation=90 if len(str(max(real_rel_positions))) > 3 else 0,
                       fontfamily='sans-serif', fontsize=20, color='#4A5568')

    anchor_plot_idx = int(np.where(all_indices == center_idx)[0][0])
    logo.highlight_position(p=anchor_plot_idx, color='#9E2A2B', alpha=0.6)

    x_labels = ax.get_xticklabels()
    if len(x_labels) > anchor_plot_idx:
        x_labels[anchor_plot_idx].set_fontweight('bold')
        x_labels[anchor_plot_idx].set_color('#2D3748')

    safe_name = class_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'motif_logo_{safe_name}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved logo to {output_path}")
    return output_path


# ===========================================================================
# 管线 B: Spatial Motif 主流程
# ===========================================================================

def run_spatial_motif_pipeline(args, model, dataset, indices, device):
    """Run the spatial motif analysis pipeline (embedding → cluster → IG → logo).

    Parameters
    ----------
    args : argparse.Namespace
    model : RNA_ClassQuery_Model
    dataset : PlantDataset
    indices : list[int]
    device : torch.device
    """
    target_mods = parse_mod_list(args.mods)
    if not target_mods:
        print("[SpatialMotif] No valid mods. Exiting.")
        return

    rng = np.random.RandomState(args.seed)
    output_dir = os.path.join(args.output_dir, 'spatial_motif')
    os.makedirs(output_dir, exist_ok=True)

    for class_name, class_idx in target_mods:
        print(f"\n{'='*80}")
        print(f"[SpatialMotif] Analyzing {class_name} (class_idx={class_idx})")
        print(f"{'='*80}")

        # Collect positive samples via full_labels
        all_labels = dataset.full_labels
        class_to_modidx = {v: k for k, v in LABEL_MAPPING.items()}
        mod_index = class_to_modidx[class_idx]
        positive_indices = []
        for idx in indices:
            y_site = np.asarray(all_labels[idx])
            if mod_index in y_site:
                positive_indices.append(idx)
        positive_indices = np.array(positive_indices)
        total_found = len(positive_indices)

        if total_found == 0:
            print(f"[SpatialMotif] Warning: No positive samples for {class_name}!")
            continue

        print(f"[SpatialMotif] Found {total_found} positive samples for {class_name}.")

        # Optional: limit samples
        rng.shuffle(positive_indices)
        if args.spatial_sample_size is not None and args.spatial_sample_size < total_found:
            positive_indices = positive_indices[:args.spatial_sample_size]
            print(f"[SpatialMotif] Using {len(positive_indices)} samples (randomly selected from {total_found}).")

        # Step 1: Extract embeddings
        print(f"\n[SpatialMotif] Step 1: Extracting latent embeddings...")
        try:
            embeddings = extract_embeddings_spatial(
                model, dataset, positive_indices, device,
                target_module_path=args.spatial_hook_target,
            )
        except ValueError as e:
            print(f"[SpatialMotif] Error during embedding extraction: {e}")
            continue

        # Step 2: Cluster
        print(f"\n[SpatialMotif] Step 2: Clustering samples...")
        try:
            clusters_dict = cluster_samples_spatial(
                embeddings, positive_indices,
                n_clusters=args.spatial_n_clusters,
                pca_components=args.spatial_pca_components,
                random_state=args.seed,
            )
        except Exception as e:
            print(f"[SpatialMotif] Error during clustering: {e}")
            continue

        # Step 3: Per-cluster IG + logo
        print(f"\n[SpatialMotif] Step 3: Computing cluster-specific attributions and logos...")
        for cluster_id, cluster_indices in clusters_dict.items():
            print(f"\n{'─'*60}")
            print(f"Cluster {cluster_id}: {len(cluster_indices)} samples")
            print(f"{'─'*60}")

            imp_matrix = calculate_spatial_attribution(
                model, dataset, class_idx, cluster_indices, device,
                n_steps=args.spatial_ig_steps,
                internal_batch_size=args.spatial_ig_internal_bs,
                batch_size=args.spatial_ig_batch_size,
            )

            cluster_name = f"{class_name}_Cluster_{cluster_id}"
            plot_top_k_logo(
                imp_matrix, class_idx, cluster_name,
                node_num=args.spatial_node_num,
                output_dir=output_dir,
            )

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n[SpatialMotif] Spatial motif analysis complete! Logos saved to: {output_dir}/")


# ===========================================================================
# 管线 A: Localized IG 主流程
# ===========================================================================

def run_localized_ig_pipeline(args, model, dataset, indices, device):
    """Run the original localized IG motif analysis pipeline."""
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    target_mods = parse_mod_list(args.mods)
    if not target_mods:
        print("[IG] No valid mods. Exiting.")
        return
    print(f"[Mods] {[m for m, _ in target_mods]}")

    # Save run config
    run_cfg = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'project_root': PROJECT_ROOT,
        'seed': args.seed,
        'device': str(device),
        'data_usage': 'train_only' if hasattr(args, 'train_indices_passthrough') else 'all',
        'dataset_name': 'plant',
        'n_used': len(indices),
    }
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # Collect events
    all_events = collect_events_from_dataset(dataset, indices, target_mods)
    print(f"[Events] total: {len(all_events)}")

    events_by_mod = defaultdict(list)
    for ev in all_events:
        events_by_mod[MOD_NAMES[ev[2]]].append(ev)

    all_match_rows, all_streme_rows, all_model_rows = [], [], []

    # Negative sampling setup
    neg_zero_seq_path = os.path.join(PROJECT_ROOT, args.negative_dir, 'zero_seq.npy')
    has_negative_source = os.path.isfile(neg_zero_seq_path)
    if not has_negative_source:
        msg = f"[Negative] zero_seq.npy not found at {neg_zero_seq_path}"
        if args.allow_no_negative:
            print(msg + " — continuing without negatives")
        else:
            raise FileNotFoundError(msg + " — use --allow_no_negative to skip")

    for mod_name, class_idx in target_mods:
        mod_events = events_by_mod.get(mod_name, [])
        print(f"\n{'='*60}")
        print(f"[{mod_name}] class_idx={class_idx}, events={len(mod_events)}")
        mod_dir = os.path.join(args.output_dir, mod_name)
        os.makedirs(mod_dir, exist_ok=True)

        cluster_eps = (args.cluster_eps if args.cluster_eps is not None
                       else DEFAULT_CLUSTER_EPS_BY_MOD.get(mod_name, 2.0))
        print(f"[{mod_name}] cluster_eps={cluster_eps}, cluster_min_samples={args.cluster_min_samples}")

        if args.sample_size > 0 and len(mod_events) > args.sample_size:
            sel = rng.choice(len(mod_events), size=args.sample_size, replace=False)
            sel.sort()
            mod_events = [mod_events[i] for i in sel]
            print(f"[{mod_name}] sampled {len(mod_events)}")

        if len(mod_events) < args.min_events_per_mod:
            print(f"[{mod_name}] WARNING: < {args.min_events_per_mod} events")

        records, fasta_seqs, fasta_hdrs, errors = [], [], [], []
        cm, ig_sc = None, None
        t0 = time.time()

        for ei, (sidx, spos, cidx) in enumerate(mod_events):
            try:
                data = dataset[sidx]
                inp = torch.FloatTensor(bytes_to_onehot(dataset.sequences[sidx]))
                edge_index = data.edge_index
                n_edges = edge_index.size(1)

                if ei == 0:
                    print(f"[{mod_name}] sample edge_index size: {n_edges} edges "
                          f"({'STRUCTURE' if n_edges > 2000 else 'SEQUENTIAL-ONLY'})")

                batch_vec = torch.zeros(inp.size(0), dtype=torch.long)
                seq_str = bytes_to_seqstr(dataset.sequences[sidx])

                score51, pprob, logit = compute_localized_ig(
                    model, inp, edge_index, batch_vec, cidx, spos,
                    ig_steps=args.ig_steps, device=device)

                if args.pred_threshold > 0 and pprob < args.pred_threshold:
                    continue

                wins = extract_windows(score51, seq_str, spos,
                                       motif_w=args.motif_w, top_k=args.top_k,
                                       skip_n=args.skip_n_windows)
                for w in wins:
                    rec = dict(sample_idx=sidx, site_pos=spos, class_idx=cidx,
                               mod_name=mod_name, pred_prob=pprob, logit=logit, **w)
                    records.append(rec)
                    fasta_seqs.append(w['seq_8nt'])
                    fasta_hdrs.append(f"{sidx}_{spos}_{w['rank']}")
            except Exception as e:
                errors.append(dict(sample_idx=sidx, site_pos=spos,
                                   class_idx=cidx, error=str(e)))
            if (ei + 1) % 100 == 0:
                print(f"  [{mod_name}] {ei+1}/{len(mod_events)} "
                      f"({time.time()-t0:.1f}s), wins={len(records)}")

        print(f"[{mod_name}] done: {len(records)} windows, {len(errors)} errors, {time.time()-t0:.1f}s")

        write_windows_tsv(os.path.join(mod_dir, 'windows.tsv'), records)
        if errors:
            with open(os.path.join(mod_dir, 'errors.tsv'), 'w') as f:
                f.write("sample_idx\tsite_pos\tclass_idx\terror\n")
                for e in errors:
                    f.write(f"{e['sample_idx']}\t{e['site_pos']}\t{e['class_idx']}\t{e['error']}\n")

        fasta_path = None
        if fasta_seqs:
            fasta_path = os.path.join(mod_dir, 'positives.fasta')
            write_fasta(fasta_path, fasta_seqs, fasta_hdrs)
            print(f"[{mod_name}] wrote {fasta_path} ({len(fasta_seqs)} seqs)")

        negative_fasta_path = None
        if has_negative_source and records:
            neg_seqs, neg_hdrs, neg_recs = generate_negatives_matched_offsets(
                records, neg_zero_seq_path, motif_w=args.motif_w,
                negative_ratio=args.negative_ratio, seed=args.negative_seed,
                allow_no_negative=args.allow_no_negative)
            if neg_seqs:
                negative_fasta_path = os.path.join(mod_dir, 'negatives.fasta')
                write_fasta(negative_fasta_path, neg_seqs, neg_hdrs)
                print(f"[{mod_name}] wrote {negative_fasta_path} ({len(neg_seqs)} neg seqs)")
                neg_tsv_path = os.path.join(mod_dir, 'negative_windows.tsv')
                write_negative_windows_tsv(neg_tsv_path, neg_recs)
                print(f"[{mod_name}] wrote {neg_tsv_path}")
                if len(neg_seqs) != len(fasta_seqs):
                    print(f"[{mod_name}] WARNING: positives={len(fasta_seqs)} vs negatives={len(neg_seqs)}")
        elif not has_negative_source and not args.allow_no_negative:
            print(f"[{mod_name}] no negative source available")

        if args.skip_external_tools:
            print(f"[{mod_name}] skipping external tools")
            continue
        if len(fasta_seqs) < args.min_events_per_mod:
            print(f"[{mod_name}] too few for STREME, skipping")
            continue

        # STREME
        try:
            streme_out = os.path.join(mod_dir, 'streme_out')
            run_streme(streme_out, fasta_path, negative_fasta=negative_fasta_path,
                       minw=args.streme_minw, maxw=args.streme_maxw, rna=True)
        except Exception as e:
            print(f"[{mod_name}] STREME failed: {e}")

        # Consensus motif
        try:
            if len(fasta_seqs) >= 4:
                cm, ig_sc = cal_consensus_motif_2(
                    fasta_seqs, [r['window_score'] for r in records],
                    eps=cluster_eps, min_samples=args.cluster_min_samples,
                    random_state=args.seed)
                pwm_list = [cm[i].T for i in range(cm.shape[0])]
                np.savez(os.path.join(mod_dir, 'treex_consensus_motifs.npz'),
                         pwms=pwm_list, ig_scores=ig_sc,
                         cluster_eps=cluster_eps,
                         cluster_min_samples=args.cluster_min_samples)
                meme_p = os.path.join(mod_dir, 'model_motifs.meme')
                export_to_meme(pwm_list, meme_p)
                rewrite_meme_alphabet_to_rna(meme_p)
                print(f"[{mod_name}] MEME exported to {meme_p}")
                remove_logo_files(mod_dir, mod_name)
        except Exception as e:
            print(f"[{mod_name}] consensus motif failed: {e}")

        # Tomtom
        streme_txt = os.path.join(streme_out, 'streme.txt')
        meme_p = os.path.join(mod_dir, 'model_motifs.meme')
        if os.path.isfile(streme_txt) and os.path.isfile(meme_p):
            try:
                tomtom_out = os.path.join(mod_dir, 'tomtom_out')
                tsv = run_tomtom_validation(meme_p, streme_txt, tomtom_out)
                print(f"[{mod_name}] Tomtom done: {tsv}")
                tomtom_rows_for_logo = parse_tomtom_tsv(tsv)
                if cm is not None and ig_sc is not None:
                    p_values = best_tomtom_pvalues_by_query(tomtom_rows_for_logo, cm.shape[0])
                    cm_plot, ig_plot, p_plot, keep_idx = filter_motifs_with_pvalues(cm, ig_sc, p_values)
                    try:
                        if cm_plot is None:
                            remove_logo_files(mod_dir, mod_name)
                            print(f"[{mod_name}] no Tomtom p-values for logos; removed motif logo files")
                        else:
                            motif_labels = [f"Model_Motif_{i + 1}" for i in keep_idx]
                            print(f"[{mod_name}] logo motifs kept after p-value filter: {motif_labels}")
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif", file_format='png',
                                             p_values=p_plot, motif_labels=motif_labels)
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif", file_format='pdf',
                                             p_values=p_plot, motif_labels=motif_labels)
                    except Exception as e_logo:
                        print(f"[{mod_name}] p-value motif logo failed: {e_logo}")
                rp = os.path.join(mod_dir, 'results.txt')
                with open(rp, 'w') as f:
                    f.write(f"Mod: {mod_name}\nWindows: {len(records)}\n\n")
                    if os.path.isfile(tsv):
                        with open(tsv, 'r') as tt:
                            for row in csv.DictReader(tt, delimiter='\t'):
                                f.write(f"  {row.get('Query_ID','')} vs {row.get('Target_ID','')}: "
                                        f"p={row.get('p-value','')} E={row.get('E-value','')} "
                                        f"q={row.get('q-value','')}\n")
            except Exception as e:
                print(f"[{mod_name}] Tomtom failed: {e}")

        match_rows, streme_rows, model_rows, _ = build_mod_summary(mod_name, mod_dir)
        all_match_rows.extend(match_rows)
        all_streme_rows.extend(streme_rows)
        all_model_rows.extend(model_rows)

    write_tomtom_summary_excel(args.output_dir, all_match_rows, all_streme_rows, all_model_rows)
    print(f"\n[IG] Localized IG motif analysis complete! Outputs in {args.output_dir}")


# ===========================================================================
# 调度入口（双管线）
# ===========================================================================

def run_pipeline(args):
    """Main entry: dispatch to localized IG and/or spatial motif pipelines."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Init] device={device}")

    cfg = load_config(args.config)
    print(f"[Config] loaded from {args.config}")

    # ---- Training / checkpoint resolution ----
    if hasattr(args, 'trained_checkpoint') and args.trained_checkpoint and args.trained_checkpoint != 'None':
        checkpoint_path = args.trained_checkpoint
        train_indices = None
        val_indices = None
        train_history_path = None
        print(f"[Checkpoint] using provided trained checkpoint: {checkpoint_path}")
    else:
        init_ckpt = getattr(args, 'init_checkpoint', _DEFAULT_INIT_CKPT)
        output_root = os.path.join(PROJECT_ROOT, 'ipynb', 'spatial_motif', 'checkpoints')
        train_result = ensure_converged_checkpoint(
            dataset_name='plant',
            config_path=args.config,
            init_checkpoint=init_ckpt,
            output_root=output_root,
            force_retrain=getattr(args, 'force_retrain', False),
            max_epochs=getattr(args, 'max_train_epochs', None),
            patience=getattr(args, 'early_stop_patience', 10),
            min_delta=getattr(args, 'early_stop_min_delta', 1e-4),
            val_ratio=getattr(args, 'val_ratio', 0.1),
            seed=getattr(args, 'seed', None),
            batch_size=getattr(args, 'train_batch_size', None),
            device=device,
        )
        checkpoint_path = train_result['checkpoint_path']
        train_indices = train_result['train_indices']
        val_indices = train_result['val_indices']
        train_history_path = train_result['history_path']
        print(f"[Checkpoint] converged checkpoint: {checkpoint_path}")

    model = load_model(cfg, checkpoint_path, device)
    print(f"[Model] loaded from {checkpoint_path}")

    # ---- Dataset ----
    data_dir = args.data_dir or os.path.join(
        PROJECT_ROOT, cfg['data'].get('plant_data_dir', 'npy/plant'))
    cache_dir = os.path.join(
        PROJECT_ROOT, cfg['data'].get('plant_cache_dir',
                                      cfg['data'].get('cache_dir', 'npy/cache')))
    print(f"[Data] loading PlantDataset from {data_dir}")
    dataset = PlantDataset(
        plant_dir=data_dir,
        cache_dir=cache_dir,
        use_cache=True,
        preload_cache=True,
    )
    print(f"[Data] dataset size: {len(dataset)}")

    if train_indices is not None:
        indices = train_indices
        print(f"[Data] using train indices only: {len(indices)} samples (val={len(val_indices)} held out)")
    else:
        indices = list(range(len(dataset)))
        print(f"[Data] using all plant samples: {len(indices)}")

    # ---- Dispatch ----
    if args.spatial_motif:
        print("\n" + "=" * 80)
        print("Running Spatial Motif Pipeline (管线 B)")
        print("=" * 80)
        run_spatial_motif_pipeline(args, model, dataset, indices, device)
    else:
        print("\n" + "=" * 80)
        print("Running Localized IG Pipeline (管线 A)")
        print("=" * 80)
        run_localized_ig_pipeline(args, model, dataset, indices, device)


# ===========================================================================
# CLI
# ===========================================================================

def int_or_none(v):
    if v.lower() == 'none':
        return None
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{v}' is not an integer or 'none'")


def main():
    p = argparse.ArgumentParser(description='Plant spatial motif analysis (dual pipeline)')
    p.add_argument('--config', default='json/plant.json')
    p.add_argument('--checkpoint', default=None,
                   help='(Legacy) direct checkpoint path. Use --trained_checkpoint instead.')
    p.add_argument('--init_checkpoint', default=_DEFAULT_INIT_CKPT)
    p.add_argument('--trained_checkpoint', default=None)
    p.add_argument('--force_retrain', action='store_true')
    p.add_argument('--max_train_epochs', type=int, default=None)
    p.add_argument('--early_stop_patience', type=int, default=10)
    p.add_argument('--early_stop_min_delta', type=float, default=1e-4)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--train_batch_size', type=int, default=None)
    p.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'ipynb', 'spatial_motif', 'output', 'plant'))
    p.add_argument('--data_dir', default=None)
    p.add_argument('--mods', default=PLANT_DEFAULT_MODS)
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--device', default='cuda')

    # ---- Pipeline mode ----
    p.add_argument('--spatial_motif', action='store_true',
                   help='Run spatial motif pipeline (管线 B) instead of localized IG (管线 A).')

    # ---- Localized IG args (管线 A) ----
    p.add_argument('--sample_size', type=int, default=5000)
    p.add_argument('--window_51', type=int, default=51)
    p.add_argument('--motif_w', type=int, default=8)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--ig_steps', type=int, default=32)
    p.add_argument('--baseline', default='zero')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--pred_threshold', type=float, default=0.0)
    p.add_argument('--positive_only', action='store_true')
    p.add_argument('--skip_n_windows', action='store_true', default=True)
    p.add_argument('--min_events_per_mod', type=int, default=10)
    p.add_argument('--skip_external_tools', action='store_true')
    p.add_argument('--alignment_r_script', default=None)
    p.add_argument('--conda_env', default='meme_env')
    p.add_argument('--negative_dir', default='npy/zero')
    p.add_argument('--negative_mode', default='matched_offsets')
    p.add_argument('--negative_ratio', type=float, default=1.0)
    p.add_argument('--negative_seed', type=int, default=666)
    p.add_argument('--allow_no_negative', action='store_true')
    p.add_argument('--streme_minw', type=int, default=5)
    p.add_argument('--streme_maxw', type=int, default=15)
    p.add_argument('--cluster_eps', type=float, default=None)
    p.add_argument('--cluster_min_samples', type=int, default=10)

    # ---- Spatial motif args (管线 B) ----
    p.add_argument('--spatial_hook_target', type=str, default='class_query_head.mha_12',
                   help='Model submodule to hook for embedding extraction.')
    p.add_argument('--spatial_n_clusters', type=int, default=5,
                   help='Number of KMeans clusters for spatial motif.')
    p.add_argument('--spatial_pca_components', type=int_or_none, default=50,
                   help='PCA components (None to skip).')
    p.add_argument('--spatial_node_num', type=int, default=10,
                   help='Number of top context positions in sequence logo.')
    p.add_argument('--spatial_sample_size', type=int, default=None,
                   help='Max positive samples per class (None = all).')
    p.add_argument('--spatial_ig_steps', type=int, default=50,
                   help='IG steps for global attribution.')
    p.add_argument('--spatial_ig_internal_bs', type=int, default=512,
                   help='IG internal batch size.')
    p.add_argument('--spatial_ig_batch_size', type=int, default=512,
                   help='Sample batch size for IG.')

    args = p.parse_args()
    if args.checkpoint and not args.trained_checkpoint:
        args.trained_checkpoint = args.checkpoint
    run_pipeline(args)


if __name__ == '__main__':
    main()