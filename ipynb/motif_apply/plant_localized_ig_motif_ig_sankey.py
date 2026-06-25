#!/usr/bin/env python3
"""
plant_localized_ig_motif_ig_sankey.py
Localized Integrated Gradients motif analysis for plant RNA modification model,
with IG-only Sankey and IG×Attention stage Sankey single-mod visualization.

Based on human_localized_ig_motif.py, adding per-modification Sankey diagrams:
  - legacy_ig_only: Input tensor -> M2D -> Attention -> Attention pooling -> Motif
  - ig_attention_stage: Input sequence -> M2D absolute_start..absolute_end ->
    Attention kept / Miss [range], with kept flow continuing to
    Pooling [range] -> Motif

v2 — Full-data loading via PlantDataset,
     real structure edge_index from LinearFold cache, negative sampling from npy/zero,
     and per-modification DBSCAN eps for consensus motifs.

Smoke test (legacy):
    python ipynb/motif/plant_localized_ig_motif_ig_sankey.py \
      --config json/plant.json \
      --checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
      --output_dir ipynb/motif/output/debug_plant_ig_sankey \
      --mods m6A --sample_size 20 --top_k 2 --ig_steps 8 \
      --skip_external_tools --batch_size 1 --device cuda \
      --enable_ig_sankey --sankey_mode legacy_ig_only

Smoke test (ig_attention_stage):
    python ipynb/motif/plant_localized_ig_motif_ig_sankey.py \
      --config json/plant.json \
      --checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
      --output_dir ipynb/motif/output/debug_plant_ig_attn_stage_sankey \
      --mods m6A --sample_size 20 --top_k 2 --ig_steps 8 \
      --skip_external_tools --batch_size 1 --device cuda \
      --enable_ig_sankey --sankey_mode ig_attention_stage
"""

import argparse, json, os, sys, time, warnings, csv, re
from collections import defaultdict, OrderedDict
from datetime import datetime
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 项目路径初始化：确保从仓库根目录导入模型、数据集和工具函数。
# Project path bootstrap: make model, dataset, and utilities importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.main_model import RNA_ClassQuery_Model
from dataset.plant import PlantDataset, MOD_NAMES, LABEL_MAPPING
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


# ---------------------------------------------------------------------------
# RNA 字母表与 one-hot 编码：输出统一使用 U，编码层让 T/U 共用第 4 通道。
# RNA alphabet and one-hot encoding: write U, and encode T/U into the same 4th channel.
# ---------------------------------------------------------------------------

def _create_byte_to_onehot_mapping():
    """创建字节值到 one-hot 编码的映射表，避免字符串处理开销。
    Create a byte-to-one-hot lookup table to avoid per-character string overhead.

    Returns
    -------
    np.ndarray
        形状为 (256, 4) 的映射表；A/C/G/U 为四个通道，T 与 U 等价。
        Mapping table with shape (256, 4); A/C/G/U are the four channels, with T=U.
    """
    mapping = np.zeros((256, 4), dtype=np.float32)

    # A (ASCII 65, 97)
    mapping[65] = [1., 0., 0., 0.]   # 'A'
    mapping[97] = [1., 0., 0., 0.]   # 'a'

    # C (ASCII 67, 99)
    mapping[67] = [0., 1., 0., 0.]   # 'C'
    mapping[99] = [0., 1., 0., 0.]   # 'c'

    # G (ASCII 71, 103)
    mapping[71] = [0., 0., 1., 0.]   # 'G'
    mapping[103] = [0., 0., 1., 0.]  # 'g'

    # T (ASCII 84, 116) - RNA 输出中会规范化为 U，但编码上与 U 相同。
    # T (ASCII 84, 116) - normalized to U for RNA output, same encoding as U.
    mapping[84] = [0., 0., 0., 1.]   # 'T'
    mapping[116] = [0., 0., 0., 1.]  # 't'

    # U (ASCII 85, 117) - RNA 中使用 U。
    # U (ASCII 85, 117) - preferred RNA alphabet.
    mapping[85] = [0., 0., 0., 1.]   # 'U'
    mapping[117] = [0., 0., 0., 1.]  # 'u'

    # N (ASCII 78, 110) - 未知核苷酸。
    # N (ASCII 78, 110) - unknown nucleotide.
    mapping[78] = [0., 0., 0., 0.]   # 'N'
    mapping[110] = [0., 0., 0., 0.]  # 'n'

    return mapping


_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()


def normalize_rna_seq(seq):
    """将序列规范化为大写 RNA 字母表，并把 T 统一替换为 U。
    Normalize a sequence to uppercase RNA alphabet and convert T to U.
    """
    return str(seq).upper().replace('T', 'U')


def bytes_to_onehot(row):
    """使用本脚本的 T/U 等价映射把字节序列编码为 one-hot。
    Encode a byte sequence with this script's T/U-equivalent one-hot table.
    """
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

# ---------------------------------------------------------------------------
# 修饰名称映射：把用户输入的修饰名映射到模型的 0-11 类别索引。
# Modification mapping: map user-facing names to model class indices 0-11.
# ---------------------------------------------------------------------------
MOD_NAME_TO_CLASS = {v: k for k, v in MOD_NAMES.items()}  # mod_name -> class_idx (0-11)
PLANT_DEFAULT_MOD_INDICES = (5, 8, 9)  # Y, m5C, m6A
PLANT_DEFAULT_MODS = ','.join(MOD_NAMES[i] for i in PLANT_DEFAULT_MOD_INDICES)

# ---------------------------------------------------------------------------
# 每类修饰的 DBSCAN eps 默认值：用于 IG 片段聚类生成模型侧共识 motif。
# Default DBSCAN eps per modification: used to cluster IG windows into model motifs.
# ---------------------------------------------------------------------------
offset=16
DEFAULT_CLUSTER_EPS_BY_MOD = {
    'Am': 1.8/offset,
    'Atol': 1.7/offset,
    'Cm': 1.8/offset,
    'Gm': 1.8/offset,
    'Tm': 2.1/offset,
    'Y': 2.1/offset,
    'ac4C': 1.9/offset,
    'm1A': 2.0/offset,
    'm5C': 1.9/offset,
    'm6A': 2.0/offset,
    'm6Am': 2.1/offset,
    'm7G': 2.0/offset,
}


# ---------------------------------------------------------------------------
# 命令行修饰列表解析：过滤未知修饰名，保留合法的 (name, class_idx)。
# CLI modification parsing: drop unknown names and keep valid (name, class_idx).
# ---------------------------------------------------------------------------
def parse_mod_list(mod_str):
    mods = []
    for name in mod_str.split(','):
        name = name.strip()
        if name not in MOD_NAME_TO_CLASS:
            warnings.warn(f"Unknown mod name '{name}', skipping.")
            continue
        mods.append((name, MOD_NAME_TO_CLASS[name]))
    return mods


# ---------------------------------------------------------------------------
# 配置加载：读取 json/plant.json 里的模型、数据和训练配置。
# Config loading: read model, data, and training settings from json/plant.json.
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 模型加载：按配置构建 RGCNFormer，并载入 checkpoint 权重。
# Model loading: instantiate RGCNFormer from config and restore checkpoint weights.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 序列辅助函数：从 PlantDataset 的字节数组恢复原始 RNA 序列字符串。
# Sequence helpers: recover raw RNA strings from PlantDataset byte arrays.
# ---------------------------------------------------------------------------

def bytes_to_seqstr(row):
    """将 (1001,) |S1/byte 数组转换为 RNA 序列字符串，T 统一写成 U。
    Convert a (1001,) |S1/byte array to an RNA sequence string, normalizing T to U.
    """
    raw = np.asarray(row)
    if raw.ndim == 0:
        seq = raw.item().decode('ascii') if isinstance(raw.item(), bytes) else str(raw.item())
    else:
        seq = b''.join(raw[i] for i in range(len(raw))).decode('ascii')
    return normalize_rna_seq(seq)


# ---------------------------------------------------------------------------
# 事件收集：从 dataset.full_labels 中提取指定修饰的位点事件。
# Event collection: extract target modification site events from dataset.full_labels.
# ---------------------------------------------------------------------------

def collect_events_from_dataset(dataset, indices, target_mods):
    """从数据集中收集 (sample_idx, site_pos, class_idx) 事件。
    Collect (sample_idx, site_pos, class_idx) events from the dataset.

    中文：直接使用 dataset.full_labels，即 1001loc.npy 中的 0-12 位点标签。
    English: uses dataset.full_labels directly, i.e. 0-12 labels from 1001loc.npy.
    """
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


# ---------------------------------------------------------------------------
# 负样本采样：从 npy/zero/zero_seq.npy 中按 positive 的相对窗口位置抽取背景片段。
# Negative sampling: sample background windows from npy/zero/zero_seq.npy with matched offsets.
# ---------------------------------------------------------------------------

def generate_negatives_matched_offsets(records, zero_seq_path, motif_w,
                                       negative_ratio=1.0, seed=666,
                                       allow_no_negative=False):
    """按相对 offset 从 zero_seq.npy 生成负样本序列。
    Generate negative sequences from zero_seq.npy using matched offsets.
    """
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


# ---------------------------------------------------------------------------
# Localized IG：固定图结构和 batch 向量，仅对输入 one-hot 序列积分求梯度。
# Localized IG: fix graph structure and batch vector, integrate gradients over input one-hot only.
# ---------------------------------------------------------------------------

class LocalizedIGWrapper(torch.nn.Module):
    """Captum IG 包装器：固定 edge_index/batch_vec，只让 x 可微。
    Captum IG wrapper: fixes edge_index/batch_vec and keeps only x differentiable.
    """

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
        ei = self.single_edge_index.unsqueeze(1)  # (2, 1, E)
        ei = ei + offsets  # (2, N, E)
        ei = ei.permute(1, 0, 2).reshape(2, -1)  # (2, N*E)
        bv = torch.arange(bs, device=x.device).repeat_interleave(self.seq_len)
        out = self.model(x, ei, bv)
        logits_12 = out[0] if isinstance(out, tuple) else out
        return logits_12[:, self.class_idx].view(bs, 1)


def compute_localized_ig(model, input_full, edge_index, batch_vec,
                         class_idx, site_pos, ig_steps=32, device='cuda'):
    """计算修饰位点周围 51nt 的 localized IG 贡献分数。
    Compute localized IG attribution scores for the 51nt window around a site.
    """
    from captum.attr import IntegratedGradients

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
    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(input_full.unsqueeze(0), baselines=baseline.unsqueeze(0),
                        n_steps=ig_steps, target=0)
    attr_np = attr.squeeze(0).detach().cpu().numpy()  # (1001, 4)
    score_all = np.abs(attr_np).sum(axis=-1)           # (1001,)
    score_51 = score_all[s0:s1]
    assert score_51.shape[0] == 51
    return score_51, pred_prob, logit_val


# ---------------------------------------------------------------------------
# 窗口提取与文件输出：从 51nt IG 分数中选 top-k 8nt 片段并写入 TSV/FASTA。
# Window extraction and I/O: select top-k 8nt windows from 51nt IG scores and write TSV/FASTA.
# ---------------------------------------------------------------------------

def extract_windows(score_51, seq_str, site_pos, motif_w=8, top_k=5, skip_n=True):
    """从 51nt 分数中提取非重叠高分短窗口。
    Extract non-overlapping high-scoring short windows from the 51nt score vector.
    """
    w51s, w51e = site_pos - 25, site_pos + 26
    seq51 = seq_str[w51s:w51e]
    assert len(seq51) == 51
    result = highest_x(score_51, w=motif_w, p=1, top_k=top_k)
    windows = []
    for rank in sorted(result.keys()):
        score, si, ei = result[rank]
        m51s, m51e = si, ei + 1  # half-open
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
    "rank\tpred_prob\tlogit\t"
    "window_ig_sum\twindow_attn_sum\twindow_attn_mean\twindow_ig_attn_score\t"
    "window_local_attn_sum\twindow_local_attn_mean\t"
    "window_attention_flow\twindow_miss_flow\n"
)

_NEGATIVE_WINDOWS_HDR = (
    "zero_idx\tabs_start\tabs_end\tmotif_start_51\tseq\t"
    "paired_sample_idx\tpaired_site_pos\n"
)


def write_windows_tsv(path, records):
    """写出 positive IG 窗口元数据。
    Write positive IG window metadata.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_WINDOWS_HDR)
        for r in records:
            f.write(f"{r['sample_idx']}\t{r['site_pos']}\t{r['class_idx']}\t"
                    f"{r['mod_name']}\t{r['window51_start']}\t{r['window51_end']}\t"
                    f"{r['motif_start']}\t{r['motif_end']}\t"
                    f"{r['window_score']:.6f}\t{r['seq_8nt']}\t{r['seq_51nt']}\t"
                    f"{r['motif_start_51']}\t{r['motif_end_51']}\t"
                    f"{r['rank']}\t{r['pred_prob']:.6f}\t{r['logit']:.6f}\t"
                    f"{r.get('window_ig_sum', 0.0):.6f}\t"
                    f"{r.get('window_attn_sum', 0.0):.6f}\t"
                    f"{r.get('window_attn_mean', 0.0):.6f}\t"
                    f"{r.get('window_ig_attn_score', 0.0):.6f}\t"
                    f"{r.get('window_local_attn_sum', 0.0):.6f}\t"
                    f"{r.get('window_local_attn_mean', 0.0):.6f}\t"
                    f"{r.get('window_attention_flow', 0.0):.6f}\t"
                    f"{r.get('window_miss_flow', 0.0):.6f}\n")


def write_negative_windows_tsv(path, neg_records):
    """写出 zero 负样本窗口元数据。
    Write zero negative window metadata.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_NEGATIVE_WINDOWS_HDR)
        for r in neg_records:
            f.write(f"{r['zero_idx']}\t{r['abs_start']}\t{r['abs_end']}\t"
                    f"{r['motif_start_51']}\t{r['seq']}\t"
                    f"{r['paired_sample_idx']}\t{r['paired_site_pos']}\n")


def write_fasta(path, seqs, headers=None):
    """写出 RNA FASTA 文件，并将 T 统一替换为 U。
    Write RNA FASTA and normalize T to U.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for i, s in enumerate(seqs):
            h = headers[i] if headers else f"seq_{i}"
            f.write(f">{h}\n{normalize_rna_seq(s)}\n")


def rewrite_meme_alphabet_to_rna(path):
    """将模型侧 MEME 文件的字母表从 A/C/G/T 精确改写为 A/C/G/U。
    Rewrite a model-side MEME file alphabet from A/C/G/T to A/C/G/U precisely.
    """
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


# ===========================================================================
# IG-only Sankey 可视化模块
# IG-only Sankey visualization module
# ===========================================================================

_SANKEY_TSV_HDR = (
    "mod_name\tsource\ttarget\tflow_weight\traw_window_count\t"
    "mean_window_score\tmean_pred_prob\tleaf_group\tleaf_label\tleaf_seq\n"
)

# Sankey 节点名称常量 / Sankey node name constants
_NODE_INPUT = "Input tensor"
_NODE_M2D = "M2D (CNN + RGCN)"
_NODE_ATTN = "Attention"
_NODE_POOL = "Attention pooling"


def load_sankey_motif_map(path):
    """加载用户自定义 motif 映射表 TSV。
    Load user-provided motif mapping TSV.

    Parameters
    ----------
    path : str
        TSV 文件路径，字段: mod_name, sample_idx, site_pos, rank, motif_id, motif_label

    Returns
    -------
    dict
        映射键 (mod_name, sample_idx, site_pos, rank) -> dict with motif_id, motif_label
    """
    motif_map = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['mod_name'], int(row['sample_idx']),
                   int(row['site_pos']), int(row['rank']))
            motif_map[key] = {
                'motif_id': row.get('motif_id', ''),
                'motif_label': row.get('motif_label', ''),
            }
    return motif_map


def _weighted_consensus(seqs, weights):
    """根据序列权重生成 A/C/G/U consensus。
    Build an A/C/G/U consensus sequence from weighted motif windows.
    """
    if not seqs:
        return ''
    letters = ['A', 'C', 'G', 'U']
    seqs = [normalize_rna_seq(s) for s in seqs]
    width = len(seqs[0])
    consensus = []
    for i in range(width):
        counts = {base: 0.0 for base in letters}
        for seq, weight in zip(seqs, weights):
            base = seq[i] if i < len(seq) else 'N'
            if base in counts:
                counts[base] += float(weight)
        consensus.append(max(letters, key=lambda b: counts[b]))
    return ''.join(consensus)


def _record_cluster_key(record):
    """生成单条窗口记录的稳定聚类键。
    Build a stable per-record key for Sankey motif clustering.
    """
    return (record.get('mod_name', ''), int(record.get('sample_idx', -1)),
            int(record.get('site_pos', -1)), int(record.get('rank', -1)))


def _seq_to_flat_onehot(seq):
    """将 8mer 转为与 motif_utils.to_onehot(s).T.flatten() 等价的向量。
    Convert an 8mer to a flattened one-hot vector, matching motif_utils.
    """
    seq = normalize_rna_seq(seq)
    mapping = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'U': [0.0, 0.0, 0.0, 1.0],
    }
    rows = [mapping.get(base, [0.0, 0.0, 0.0, 0.0]) for base in seq]
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def build_sankey_motif_cluster_map(records, eps=0.3, min_samples=10,
                                   random_state=666):
    """用 DBSCAN 构建 Sankey motif cluster 映射。
    Build a motif-cluster map for Sankey leaves using DBSCAN.

    中文：参照 human_localized_ig_motif.py / cal_consensus_motif_2：
    8mer -> one-hot 展平 -> UMAP(若可用)/PCA 降到 2D -> DBSCAN。
    本函数额外返回每条窗口的 cluster 归属，供 Sankey 聚合使用。
    """
    from sklearn.cluster import DBSCAN
    try:
        import umap as umap_lib
        has_umap = True
    except ImportError:
        has_umap = False

    clean_records = []
    for rec in records:
        seq = normalize_rna_seq(rec.get('seq_8nt', ''))
        if not seq:
            continue
        flow = rec.get('window_attention_flow',
                       rec.get('window_ig_attn_score',
                               rec.get('window_score', 0.0)))
        clean_records.append({
            'key': _record_cluster_key(rec),
            'seq': seq,
            'flow': float(flow),
        })

    if not clean_records:
        return {}

    X = np.asarray([_seq_to_flat_onehot(r['seq']) for r in clean_records],
                   dtype=np.float32)

    if len(clean_records) < 2:
        class_labels = np.zeros(len(clean_records), dtype=int)
    else:
        if has_umap:
            n_neighbors = max(2, min(15, len(clean_records) - 1))
            Y = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                              random_state=random_state).fit_transform(X)
        else:
            from sklearn.decomposition import PCA
            n_components = min(2, X.shape[0], X.shape[1])
            Y = PCA(n_components=n_components,
                    random_state=random_state).fit_transform(X)
            if n_components == 1:
                Y = np.column_stack([Y[:, 0], np.zeros(Y.shape[0])])

        min_samples = max(2, int(min_samples))
        clustering = DBSCAN(eps=float(eps), min_samples=min_samples).fit(Y)
        class_labels = clustering.labels_

        valid_labels = np.unique(class_labels[class_labels != -1])
        if len(valid_labels) == 0:
            class_labels = np.zeros_like(class_labels)

    cluster_info_by_label = {}
    for idx, label in enumerate(class_labels):
        info = cluster_info_by_label.setdefault(int(label), {
            'seqs': [], 'weights': [], 'flow': 0.0, 'count': 0, 'is_noise': int(label) == -1,
        })
        flow = clean_records[idx]['flow']
        info['seqs'].append(clean_records[idx]['seq'])
        info['weights'].append(flow)
        info['flow'] += flow
        info['count'] += 1

    cluster_infos = []
    for label, info in cluster_info_by_label.items():
        consensus = _weighted_consensus(info['seqs'], info['weights'])
        cluster_infos.append({
            'label': label,
            'consensus': consensus,
            'flow': info['flow'],
            'count': info['count'],
            'is_noise': info['is_noise'],
        })

    cluster_infos.sort(key=lambda x: x['flow'], reverse=True)
    cluster_map = {}
    label_to_display = {}
    for new_idx, info in enumerate(cluster_infos, start=1):
        if info['is_noise']:
            key = f"motif_cluster_noise_{info['consensus']}"
            label = f"DBSCAN noise: {info['consensus']}"
        else:
            key = f"motif_cluster_{new_idx:02d}_{info['consensus']}"
            label = f"Motif cluster {new_idx:02d}: {info['consensus']}"
        label_to_display[info['label']] = {
            'key': key,
            'label': label,
            'consensus': info['consensus'],
            'window_count': info['count'],
            'flow': info['flow'],
        }

    for rec, label in zip(clean_records, class_labels):
        cluster_map[rec['key']] = label_to_display[int(label)]
    return cluster_map


def assign_sankey_leaf(record, motif_map, group_by, motif_cluster_map=None):
    """为单条 record 确定桑基图叶子节点标识。
    Determine the Sankey leaf node identifier for a single record.

    Parameters
    ----------
    record : dict
        一条 IG 窗口记录，需含 mod_name, sample_idx, site_pos, rank, seq_8nt。
    motif_map : dict or None
        用户 motif 映射表（load_sankey_motif_map 的返回值）。
    group_by : str
        'motif_map'、'motif_cluster' 或 'seq_8nt'。

    Returns
    -------
    leaf_key : str
        用于聚合的叶子键。
    leaf_label : str
        用于显示的叶子标签。
    leaf_seq : str
        代表序列（seq_8nt）。
    """
    seq_8nt = record.get('seq_8nt', '')

    if group_by == 'motif_map' and motif_map is not None:
        key = (record['mod_name'], record['sample_idx'],
               record['site_pos'], record['rank'])
        if key in motif_map:
            m = motif_map[key]
            leaf_label = m.get('motif_label') or m.get('motif_id', seq_8nt)
            leaf_key = leaf_label
            return leaf_key, leaf_label, seq_8nt

    if group_by == 'motif_cluster' and motif_cluster_map is not None:
        cluster = motif_cluster_map.get(_record_cluster_key(record))
        if cluster is not None:
            return cluster['key'], cluster['label'], cluster['consensus']

    # 默认按 seq_8nt 聚合 / Default: group by seq_8nt
    leaf_key = seq_8nt
    leaf_label = f"{seq_8nt}"
    return leaf_key, leaf_label, seq_8nt


def build_ig_only_sankey_edges(records, mod_name, motif_map=None,
                               group_by='motif_cluster', top_n=20,
                               min_flow=0.0, motif_cluster_eps=0.3,
                               motif_cluster_min_samples=10,
                               motif_cluster_random_state=666):
    """从 IG 窗口记录构建桑基图边列表。
    Build Sankey edge list from IG window records.

    Parameters
    ----------
    records : list[dict]
        IG 窗口记录，每条需含 window_score, pred_prob, seq_8nt 等。
    mod_name : str
        当前修饰名称。
    motif_map : dict or None
        用户 motif 映射表。
    group_by : str
        叶子分组模式: 'motif_map' 或 'seq_8nt'。
    top_n : int
        保留 top-N 叶子，其余合并为 Other。
    min_flow : float
        最小流量阈值，低于此值的叶子合并到 Other。

    Returns
    -------
    edge_rows : list[dict]
        每个 dict 包含 Sankey TSV 所需字段。
    """
    motif_cluster_map = None
    if group_by == 'motif_cluster':
        motif_cluster_map = build_sankey_motif_cluster_map(
            records, eps=motif_cluster_eps,
            min_samples=motif_cluster_min_samples,
            random_state=motif_cluster_random_state)

    # ---- 第一步：按 leaf_key 聚合流量 ----
    # Step 1: aggregate flow by leaf_key
    leaf_agg = OrderedDict()  # leaf_key -> {flow, count, scores, probs, label, seq}
    for rec in records:
        leaf_key, leaf_label, leaf_seq = assign_sankey_leaf(
            rec, motif_map, group_by, motif_cluster_map=motif_cluster_map)
        ws = rec['window_score']
        if leaf_key not in leaf_agg:
            leaf_agg[leaf_key] = {
                'flow': 0.0, 'count': 0, 'scores': [], 'probs': [],
                'label': leaf_label, 'seq': leaf_seq,
            }
        leaf_agg[leaf_key]['flow'] += ws
        leaf_agg[leaf_key]['count'] += 1
        leaf_agg[leaf_key]['scores'].append(ws)
        if 'pred_prob' in rec:
            leaf_agg[leaf_key]['probs'].append(rec['pred_prob'])

    # ---- 第二步：按 flow 降序排列，保留 top-N，剩余合并为 Other ----
    # Step 2: sort by flow descending, keep top-N, merge rest into Other
    sorted_leaves = sorted(leaf_agg.items(), key=lambda x: x[1]['flow'], reverse=True)

    visible = []
    other_flow = 0.0
    other_count = 0
    other_scores = []
    other_probs = []
    for i, (lk, agg) in enumerate(sorted_leaves):
        if agg['flow'] < min_flow:
            other_flow += agg['flow']
            other_count += agg['count']
            other_scores.extend(agg['scores'])
            other_probs.extend(agg['probs'])
            continue
        if i >= top_n:
            other_flow += agg['flow']
            other_count += agg['count']
            other_scores.extend(agg['scores'])
            other_probs.extend(agg['probs'])
            continue
        visible.append((lk, agg))

    # 如果 Other 有流量，添加为合并叶子
    # If Other has flow, add it as a merged leaf
    if other_flow > 0:
        visible.append(('__other__', {
            'flow': other_flow, 'count': other_count,
            'scores': other_scores, 'probs': other_probs,
            'label': 'Other selected motif windows', 'seq': '',
        }))

    if not visible:
        return []

    # ---- 第三步：计算总流量 ----
    # Step 3: compute total flow
    total_flow = sum(agg['flow'] for _, agg in visible)

    # ---- 第四步：构建边 ----
    # Step 4: build edges
    edge_rows = []

    # 结构边：Input tensor -> M2D -> Attention -> Attention pooling
    # 所有结构边权重相同 = total_flow
    pipeline_edges = [
        (_NODE_INPUT, _NODE_M2D),
        (_NODE_M2D, _NODE_ATTN),
        (_NODE_ATTN, _NODE_POOL),
    ]
    for src, tgt in pipeline_edges:
        edge_rows.append({
            'mod_name': mod_name,
            'source': src,
            'target': tgt,
            'flow_weight': total_flow,
            'raw_window_count': len(records),
            'mean_window_score': np.mean([r['window_score'] for r in records]),
            'mean_pred_prob': np.mean([r.get('pred_prob', 0.0) for r in records]),
            'leaf_group': '__pipeline__',
            'leaf_label': '',
            'leaf_seq': '',
        })

    # 叶子边：Attention pooling -> each leaf
    for leaf_key, agg in visible:
        edge_rows.append({
            'mod_name': mod_name,
            'source': _NODE_POOL,
            'target': agg['label'],
            'flow_weight': agg['flow'],
            'raw_window_count': agg['count'],
            'mean_window_score': float(np.mean(agg['scores'])) if agg['scores'] else 0.0,
            'mean_pred_prob': float(np.mean(agg['probs'])) if agg['probs'] else 0.0,
            'leaf_group': leaf_key,
            'leaf_label': agg['label'],
            'leaf_seq': agg['seq'],
        })

    return edge_rows


def write_sankey_edges_tsv(path, edge_rows):
    """写出 Sankey 边 TSV。
    Write Sankey edges TSV.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_SANKEY_TSV_HDR)
        for r in edge_rows:
            f.write(f"{r['mod_name']}\t{r['source']}\t{r['target']}\t"
                    f"{r['flow_weight']:.6f}\t{r['raw_window_count']}\t"
                    f"{r['mean_window_score']:.6f}\t{r['mean_pred_prob']:.6f}\t"
                    f"{r['leaf_group']}\t{r['leaf_label']}\t{r['leaf_seq']}\n")


def draw_ig_only_sankey(edge_rows, output_html, output_png=None, title=None,
                        width=1200, height=700, font_size=18,
                        node_pad=20, node_thickness=25):
    """绘制 IG-only 桑基图并输出 HTML（可选 PNG）。
    Draw IG-only Sankey diagram and output HTML (optionally PNG).

    Parameters
    ----------
    edge_rows : list[dict]
        build_ig_only_sankey_edges 的返回值。
    output_html : str
        HTML 输出路径。
    output_png : str or None
        PNG 输出路径（需要 kaleido）。如果 kaleido 不可用则跳过。
    title : str or None
        图标题。
    """
    import plotly.graph_objects as go

    if not edge_rows:
        print("[Sankey] no edges to draw, skipping")
        return

    # ---- 构建节点列表（保持顺序） ----
    # Build node list preserving order
    node_order = [_NODE_INPUT, _NODE_M2D, _NODE_ATTN, _NODE_POOL]
    node_set = set(node_order)
    for r in edge_rows:
        if r['source'] == _NODE_POOL:
            # 这是叶子边，target 是叶子节点
            if r['target'] not in node_set:
                node_order.append(r['target'])
                node_set.add(r['target'])

    node_idx = {name: i for i, name in enumerate(node_order)}

    # ---- 构建 link ----
    sources = []
    targets = []
    values = []
    link_labels = []
    link_colors = []

    # 管道边用统一颜色，叶子边用渐变色
    pipeline_color = 'rgba(31, 119, 180, 0.4)'
    leaf_colors = [
        'rgba(255, 127, 14, 0.5)',
        'rgba(44, 160, 44, 0.5)',
        'rgba(214, 39, 40, 0.5)',
        'rgba(148, 103, 189, 0.5)',
        'rgba(140, 86, 75, 0.5)',
        'rgba(227, 119, 194, 0.5)',
        'rgba(127, 127, 127, 0.5)',
        'rgba(188, 189, 34, 0.5)',
        'rgba(23, 190, 207, 0.5)',
        'rgba(31, 119, 180, 0.5)',
    ]

    leaf_counter = 0
    for r in edge_rows:
        src_idx = node_idx[r['source']]
        tgt_idx = node_idx[r['target']]
        sources.append(src_idx)
        targets.append(tgt_idx)
        values.append(r['flow_weight'])

        if r['leaf_group'] == '__pipeline__':
            link_labels.append(f"{r['source']} → {r['target']}<br>"
                               f"Total IG flow: {r['flow_weight']:.4f}")
            link_colors.append(pipeline_color)
        else:
            label = r['leaf_label']
            link_labels.append(
                f"{label}<br>"
                f"IG flow: {r['flow_weight']:.4f}<br>"
                f"Windows: {r['raw_window_count']}<br>"
                f"Mean score: {r['mean_window_score']:.4f}<br>"
                f"Mean pred_prob: {r['mean_pred_prob']:.4f}"
            )
            color = leaf_colors[leaf_counter % len(leaf_colors)]
            link_colors.append(color)
            leaf_counter += 1

    # 节点颜色
    node_colors = [
        'rgba(31, 119, 180, 0.8)',   # Input tensor
        'rgba(255, 127, 14, 0.8)',   # M2D
        'rgba(44, 160, 44, 0.8)',    # Attention
        'rgba(214, 39, 40, 0.8)',    # Attention pooling
    ]
    # 叶子节点用灰色
    for _ in range(len(node_order) - 4):
        node_colors.append('rgba(180, 180, 180, 0.8)')

    if title is None:
        mod_name = edge_rows[0]['mod_name'] if edge_rows else ''
        title = f"IG-only Sankey — {mod_name} (flow = sum of IG window scores)"

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=node_pad,
            thickness=node_thickness,
            line=dict(color='black', width=0.5),
            label=node_order,
            color=node_colors,
            hovertemplate='%{label}<br>Total flow: %{value:.4f}<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate='%{label}<extra></extra>',
        ),
    )])

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=font_size),
        font=dict(size=font_size),
        width=width,
        height=height,
        annotations=[
            dict(
                text="IG-only: flow weights = sum(IG window scores). "
                     "Attention & Attention pooling are structural narrative nodes.",
                xref="paper", yref="paper",
                x=0.5, y=-0.08, showarrow=False,
                font=dict(size=font_size, color="gray"),
            )
        ],
    )

    os.makedirs(os.path.dirname(output_html) or '.', exist_ok=True)
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"[Sankey] wrote HTML: {output_html}")

    if output_png:
        try:
            fig.write_image(output_png, scale=2)
            print(f"[Sankey] wrote PNG: {output_png}")
        except Exception as e:
            print(f"[Sankey] PNG export failed (kaleido may not be installed): {e}")


# ===========================================================================
# Attention 提取函数
# Attention extraction function
# ===========================================================================

def extract_target_attention_51(model, input_full, edge_index, batch_vec,
                                class_idx, site_pos, device):
    """提取目标修饰类别的 class-query attention 在 51nt 窗口上的值。
    Extract the target class-query attention weights for the 51nt window around a site.

    Parameters
    ----------
    model : RNA_ClassQuery_Model
        已加载的模型。
    input_full : Tensor (1001, 4)
        one-hot 输入。
    edge_index : Tensor (2, E)
        图边索引。
    batch_vec : Tensor (1001,)
        批次向量。
    class_idx : int
        目标修饰类别索引 (0-11)。
    site_pos : int
        修饰位点绝对位置。
    device : torch.device
        推理设备。

    Returns
    -------
    attn_51 : np.ndarray (51,) or None
        目标 class 在 51nt 窗口上的 attention 权重；若拿不到则返回 None。
    """
    input_full = input_full.to(device)
    edge_index_dev = edge_index.to(device)
    batch_vec_dev = batch_vec.to(device)

    try:
        with torch.no_grad():
            out = model(input_full.unsqueeze(0), edge_index_dev, batch_vec_dev,
                        return_attention=True)

            # 兼容 hierarchical / simple_pooling 两种返回格式
            # Compatible with hierarchical: (logits_12, logits_4, attn_weights_12)
            # and simple_pooling: (logits, attn_weights)
            attn_weights = None
            if isinstance(out, tuple):
                if len(out) == 3:
                    # hierarchical: (logits_12, logits_4, attn_weights_12)
                    attn_weights = out[2]
                elif len(out) == 2:
                    # simple_pooling: (logits, attn_weights)
                    attn_weights = out[1]

            if attn_weights is None:
                return None

            # attn_weights: [B, num_classes, 1001]
            attn_full = attn_weights[0, class_idx].cpu().numpy()  # (1001,)
            s0 = site_pos - 25
            s1 = site_pos + 26
            attn_51 = attn_full[s0:s1]
            assert attn_51.shape[0] == 51, \
                f"attn_51 shape mismatch: {attn_51.shape[0]} != 51"
            return attn_51
    except Exception as e:
        print(f"[Attention] extraction failed: {e}")
        return None


# ===========================================================================
# IG×Attention Stage Sankey 可视化模块
# IG×Attention Stage Sankey visualization module
# ===========================================================================

_STAGE_SANKEY_TSV_HDR = (
    "mod_name\tsource\ttarget\tflow_weight\t"
    "ig_sum\tattention_sum\tig_attention_sum\t"
    "local_attention_sum\tattention_flow\tmiss_flow\t"
    "raw_window_count\tmean_window_score\tmean_pred_prob\t"
    "source_stage\ttarget_stage\tsource_range\ttarget_range\t"
    "leaf_group\tleaf_label\tleaf_seq\tmotif_start_rel\tmotif_end_rel\n"
)

# Sankey 阶段节点名称前缀 / Stage node name prefixes
_STAGE_INPUT = "Input sequence"
_STAGE_M2D = "M2D"
_STAGE_ATTN = "Attention"
_STAGE_POOL = "Pooling"
_STAGE_MOTIF = "Motif"
_STAGE_SUPPRESSED = "Miss"
_INPUT_SEQUENCE_RANGE = "0--1000"
_OTHER_RANGE_LABEL = "Miss"


def _coord_label(pos):
    """生成序列绝对坐标标签。
    Generate an absolute sequence coordinate label.
    """
    return str(int(pos))


def _range_label(abs_start, abs_end):
    """生成绝对坐标标签，例如 496..503。
    Generate an inclusive absolute-coordinate label, e.g. 496..503.
    """
    abs_end_inclusive = int(abs_end) - 1
    abs_start = int(abs_start)
    if abs_start == abs_end_inclusive:
        return _coord_label(abs_start)
    return f"{_coord_label(abs_start)}--{_coord_label(abs_end_inclusive)}"


def build_ig_attention_stage_sankey_edges(records, mod_name, motif_map=None,
                                          group_by='motif_cluster', top_n=20,
                                          min_flow=0.0,
                                          motif_cluster_eps=0.3,
                                          motif_cluster_min_samples=10,
                                          motif_cluster_random_state=666):
    """从 IG 窗口记录构建 IG×Attention 阶段桑基图边列表。
    Build IG×Attention stage Sankey edge list from IG window records.

    完整 input sequence 作为单一整体，先把 raw IG 分配到各个 M2D 绝对坐标区间，
    每个序列区间再在 M2D 阶段被 attention 拆成：
    Input sequence -> M2D [absolute range]       flow = raw IG
    M2D [range] -> Attention [range]              flow = IG kept by local attention
    M2D [range] -> Miss [range]                   flow = IG Miss
    Attention [range] -> Pooling [range]          flow = IG kept by local attention
    Pooling [range] -> motif leaf                 flow = IG kept by local attention

    Parameters
    ----------
    records : list[dict]
        IG 窗口记录，需含 window_score, pred_prob, seq_8nt,
        window_ig_sum, window_attn_sum, window_attention_flow 等。
    mod_name : str
        当前修饰名称。
    motif_map : dict or None
        用户 motif 映射表。
    group_by : str
        叶子分组模式。
    top_n : int
        保留 top-N 聚合单元。
    min_flow : float
        最小 ig_attn_sum 阈值。

    Returns
    -------
    edge_rows : list[dict]
    """
    def _append_edge(edge_rows, source, target, flow_weight, source_stage,
                     target_stage, source_range, target_range, agg,
                     leaf_group='', leaf_label='', leaf_seq=''):
        if flow_weight <= 0:
            return
        edge_rows.append({
            'mod_name': mod_name,
            'source': source,
            'target': target,
            'flow_weight': flow_weight,
            'ig_sum': agg['ig_sum'],
            'attention_sum': agg['attn_sum'],
            'ig_attention_sum': agg['ig_attn_sum'],
            'local_attention_sum': agg['local_attn_sum'],
            'attention_flow': agg['attention_flow'],
            'miss_flow': agg['miss_flow'],
            'raw_window_count': agg['count'],
            'mean_window_score': float(np.mean(agg['scores'])) if agg['scores'] else 0.0,
            'mean_pred_prob': float(np.mean(agg['probs'])) if agg['probs'] else 0.0,
            'source_stage': source_stage,
            'target_stage': target_stage,
            'source_range': source_range,
            'target_range': target_range,
            'leaf_group': leaf_group,
            'leaf_label': leaf_label,
            'leaf_seq': leaf_seq,
            'motif_start_rel': agg['rel_start'],
            'motif_end_rel': agg['rel_end'],
        })

    motif_cluster_map = None
    if group_by == 'motif_cluster':
        motif_cluster_map = build_sankey_motif_cluster_map(
            records, eps=motif_cluster_eps,
            min_samples=motif_cluster_min_samples,
            random_state=motif_cluster_random_state)

    # ---- Step 1: 按 (motif_range, leaf_key) 聚合 motif 单元 ----
    unit_map = OrderedDict()
    for rec in records:
        leaf_key, leaf_label, leaf_seq = assign_sankey_leaf(
            rec, motif_map, group_by, motif_cluster_map=motif_cluster_map)
        m51s = rec['motif_start_51']
        m51e = rec['motif_end_51']
        rel_start = m51s - 25
        rel_end = m51e - 25
        motif_abs_start = rec.get('motif_start', rel_start)
        motif_abs_end = rec.get('motif_end', rel_end)
        range_lbl = _range_label(motif_abs_start, motif_abs_end)
        agg_key = (range_lbl, leaf_key)

        if agg_key not in unit_map:
            unit_map[agg_key] = {
                'range_label': range_lbl,
                'rel_start': rel_start,
                'rel_end': rel_end,
                'leaf_key': leaf_key,
                'leaf_label': leaf_label,
                'leaf_seq': leaf_seq,
                'ig_sum': 0.0,
                'attn_sum': 0.0,
                'ig_attn_sum': 0.0,
                'local_attn_sum': 0.0,
                'attention_flow': 0.0,
                'miss_flow': 0.0,
                'count': 0,
                'scores': [],
                'probs': [],
            }
        agg = unit_map[agg_key]
        ig_sum = rec.get('window_ig_sum', rec['window_score'])
        attention_flow = rec.get('window_attention_flow',
                                 rec.get('window_ig_attn_score', rec['window_score']))
        miss_flow = rec.get('window_miss_flow', max(ig_sum - attention_flow, 0.0))
        agg['ig_sum'] += ig_sum
        agg['attn_sum'] += rec.get('window_attn_sum', 0.0)
        agg['ig_attn_sum'] += rec.get('window_ig_attn_score', attention_flow)
        agg['local_attn_sum'] += rec.get('window_local_attn_sum', 0.0)
        agg['attention_flow'] += attention_flow
        agg['miss_flow'] += miss_flow
        agg['count'] += 1
        agg['scores'].append(rec['window_score'])
        if 'pred_prob' in rec:
            agg['probs'].append(rec['pred_prob'])

    if not unit_map:
        return []

    # ---- Step 2: 按 attention_flow 降序排列，保留 top-N motif 单元 ----
    sorted_aggs = sorted(unit_map.items(), key=lambda x: x[1]['attention_flow'],
                         reverse=True)

    visible = []
    other_ig = 0.0
    other_attn = 0.0
    other_ig_attn = 0.0
    other_local_attn = 0.0
    other_attention_flow = 0.0
    other_miss_flow = 0.0
    other_count = 0
    other_scores = []
    other_probs = []
    for i, (ak, agg) in enumerate(sorted_aggs):
        if agg['attention_flow'] < min_flow:
            other_ig += agg['ig_sum']
            other_attn += agg['attn_sum']
            other_ig_attn += agg['ig_attn_sum']
            other_local_attn += agg['local_attn_sum']
            other_attention_flow += agg['attention_flow']
            other_miss_flow += agg['miss_flow']
            other_count += agg['count']
            other_scores.extend(agg['scores'])
            other_probs.extend(agg['probs'])
            continue
        if i >= top_n:
            other_ig += agg['ig_sum']
            other_attn += agg['attn_sum']
            other_ig_attn += agg['ig_attn_sum']
            other_local_attn += agg['local_attn_sum']
            other_attention_flow += agg['attention_flow']
            other_miss_flow += agg['miss_flow']
            other_count += agg['count']
            other_scores.extend(agg['scores'])
            other_probs.extend(agg['probs'])
            continue
        visible.append((ak, agg))

    if other_attention_flow > 0 or other_miss_flow > 0:
        visible.append(('__other__', {
            'range_label': _OTHER_RANGE_LABEL,
            'rel_start': 0, 'rel_end': 0,
            'leaf_key': '__other__',
            'leaf_label': 'Other motif',
            'leaf_seq': '',
            'ig_sum': other_ig,
            'attn_sum': other_attn,
            'ig_attn_sum': other_ig_attn,
            'local_attn_sum': other_local_attn,
            'attention_flow': other_attention_flow,
            'miss_flow': other_miss_flow,
            'count': other_count,
            'scores': other_scores,
            'probs': other_probs,
        }))

    if not visible:
        return []

    # ---- Step 3: 按 M2D/Attention/Pooling range 汇总阶段流 ----
    range_map = OrderedDict()
    for _, agg in visible:
        range_lbl = agg['range_label']
        if range_lbl not in range_map:
            range_map[range_lbl] = {
                'range_label': range_lbl,
                'rel_start': agg['rel_start'],
                'rel_end': agg['rel_end'],
                'ig_sum': 0.0,
                'attn_sum': 0.0,
                'ig_attn_sum': 0.0,
                'local_attn_sum': 0.0,
                'attention_flow': 0.0,
                'miss_flow': 0.0,
                'count': 0,
                'scores': [],
                'probs': [],
            }
        ragg = range_map[range_lbl]
        ragg['ig_sum'] += agg['ig_sum']
        ragg['attn_sum'] += agg['attn_sum']
        ragg['ig_attn_sum'] += agg['ig_attn_sum']
        ragg['local_attn_sum'] += agg['local_attn_sum']
        ragg['attention_flow'] += agg['attention_flow']
        ragg['miss_flow'] += agg['miss_flow']
        ragg['count'] += agg['count']
        ragg['scores'].extend(agg['scores'])
        ragg['probs'].extend(agg['probs'])

    # ---- Step 4: 构建阶段边 + motif 叶子边 ----
    edge_rows = []
    for range_lbl, ragg in range_map.items():
        src_input = _STAGE_INPUT
        src_m2d = f"{_STAGE_M2D} {range_lbl}"
        tgt_attn = f"{_STAGE_ATTN} {range_lbl}"
        tgt_miss = f"{_STAGE_SUPPRESSED} {range_lbl}"
        tgt_pool = f"{_STAGE_POOL} {range_lbl}"

        _append_edge(
            edge_rows, src_input, src_m2d, ragg['ig_sum'],
            _STAGE_INPUT, _STAGE_M2D, _INPUT_SEQUENCE_RANGE, range_lbl, ragg,
            leaf_group='__input_to_m2d__',
            leaf_label='Whole input sequence to M2D evidence',
        )
        _append_edge(
            edge_rows, src_m2d, tgt_attn, ragg['attention_flow'],
            _STAGE_M2D, _STAGE_ATTN, range_lbl, range_lbl, ragg,
            leaf_group='__attention_kept__',
            leaf_label='Attention-kept IG evidence',
        )
        _append_edge(
            edge_rows, src_m2d, tgt_miss, ragg['miss_flow'],
            _STAGE_M2D, _STAGE_SUPPRESSED, range_lbl, range_lbl, ragg,
            leaf_group='__miss__',
            leaf_label='Miss',
        )
        _append_edge(
            edge_rows, tgt_attn, tgt_pool, ragg['attention_flow'],
            _STAGE_ATTN, _STAGE_POOL, range_lbl, range_lbl, ragg,
            leaf_group='__attention_kept__',
            leaf_label='Attention-kept IG evidence',
        )

    for _, agg in visible:
        range_lbl = agg['range_label']
        src_pool = f"{_STAGE_POOL} {range_lbl}"
        tgt_motif = agg['leaf_label']
        _append_edge(
            edge_rows, src_pool, tgt_motif, agg['attention_flow'],
            _STAGE_POOL, _STAGE_MOTIF, range_lbl, '', agg,
            leaf_group=agg['leaf_key'],
            leaf_label=tgt_motif,
            leaf_seq=agg['leaf_seq'],
        )

    return edge_rows


def write_stage_sankey_edges_tsv(path, edge_rows):
    """写出 IG×Attention 阶段 Sankey 边 TSV。
    Write IG×Attention stage Sankey edges TSV.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(_STAGE_SANKEY_TSV_HDR)
        for r in edge_rows:
            f.write(
                f"{r['mod_name']}\t{r['source']}\t{r['target']}\t"
                f"{r['flow_weight']:.6f}\t"
                f"{r['ig_sum']:.6f}\t{r['attention_sum']:.6f}\t{r['ig_attention_sum']:.6f}\t"
                f"{r['local_attention_sum']:.6f}\t{r['attention_flow']:.6f}\t"
                f"{r['miss_flow']:.6f}\t"
                f"{r['raw_window_count']}\t{r['mean_window_score']:.6f}\t{r['mean_pred_prob']:.6f}\t"
                f"{r['source_stage']}\t{r['target_stage']}\t{r['source_range']}\t{r['target_range']}\t"
                f"{r['leaf_group']}\t{r['leaf_label']}\t{r['leaf_seq']}\t"
                f"{r['motif_start_rel']}\t{r['motif_end_rel']}\n")


def draw_ig_attention_stage_sankey(edge_rows, output_html, output_png=None,
                                   title=None, width=1200, height=700,
                                   font_size=18, node_pad=20,
                                   node_thickness=25):
    """绘制 IG×Attention 阶段桑基图并输出 HTML（可选 PNG）。
    Draw IG×Attention stage Sankey diagram and output HTML (optionally PNG).

    节点按阶段组织：Input sequence -> M2D ranges -> Attention ranges
    -> Miss ranges -> Pooling ranges -> Motif leaves.
    颜色：Input=purple, M2D=blue, Attention=orange, Miss=red,
    Pooling=green, Motif=gray.

    Parameters
    ----------
    edge_rows : list[dict]
        build_ig_attention_stage_sankey_edges 的返回值。
    output_html : str
        HTML 输出路径。
    output_png : str or None
        PNG 输出路径。
    title : str or None
        图标题。
    """
    import plotly.graph_objects as go

    if not edge_rows:
        print("[Stage Sankey] no edges to draw, skipping")
        return

    # ---- 按阶段收集节点，保持顺序 ----
    input_nodes = OrderedDict()
    m2d_nodes = OrderedDict()
    attn_nodes = OrderedDict()
    miss_nodes = OrderedDict()
    pool_nodes = OrderedDict()
    motif_nodes = OrderedDict()

    for r in edge_rows:
        src = r['source']
        tgt = r['target']
        src_stage = r['source_stage']
        tgt_stage = r['target_stage']

        if src_stage == _STAGE_INPUT:
            input_nodes[src] = True
        elif src_stage == _STAGE_M2D:
            m2d_nodes[src] = True
        elif src_stage == _STAGE_ATTN:
            attn_nodes[src] = True
        elif src_stage == _STAGE_SUPPRESSED:
            miss_nodes[src] = True
        elif src_stage == _STAGE_POOL:
            pool_nodes[src] = True

        if tgt_stage == _STAGE_M2D:
            m2d_nodes[tgt] = True
        elif tgt_stage == _STAGE_ATTN:
            attn_nodes[tgt] = True
        elif tgt_stage == _STAGE_SUPPRESSED:
            miss_nodes[tgt] = True
        elif tgt_stage == _STAGE_POOL:
            pool_nodes[tgt] = True
        elif tgt_stage == _STAGE_MOTIF:
            motif_nodes[tgt] = True

    node_order = list(input_nodes.keys()) + list(m2d_nodes.keys()) + \
                 list(attn_nodes.keys()) + list(miss_nodes.keys()) + \
                 list(pool_nodes.keys()) + list(motif_nodes.keys())
    node_idx = {name: i for i, name in enumerate(node_order)}

    # ---- 节点颜色 ----
    color_map = {
        _STAGE_INPUT: 'rgba(117, 112, 179, 0.8)', # purple
        _STAGE_M2D: 'rgba(31, 119, 180, 0.8)',    # blue
        _STAGE_ATTN: 'rgba(255, 127, 14, 0.8)',    # orange
        _STAGE_SUPPRESSED: 'rgba(190, 80, 70, 0.8)', # red
        _STAGE_POOL: 'rgba(44, 160, 44, 0.8)',     # green
        _STAGE_MOTIF: 'rgba(180, 180, 180, 0.8)',  # gray
    }

    link_color_map = {
        _STAGE_INPUT: 'rgba(117, 112, 179, 0.35)', # purple
        _STAGE_M2D: 'rgba(31, 119, 180, 0.4)',     # blue
        _STAGE_ATTN: 'rgba(255, 127, 14, 0.4)',    # orange
        _STAGE_SUPPRESSED: 'rgba(190, 80, 70, 0.35)', # red
        _STAGE_POOL: 'rgba(44, 160, 44, 0.4)',     # green
    }

    node_colors = []
    for name in node_order:
        if name in input_nodes:
            node_colors.append(color_map[_STAGE_INPUT])
        elif name in m2d_nodes:
            node_colors.append(color_map[_STAGE_M2D])
        elif name in attn_nodes:
            node_colors.append(color_map[_STAGE_ATTN])
        elif name in miss_nodes:
            node_colors.append(color_map[_STAGE_SUPPRESSED])
        elif name in pool_nodes:
            node_colors.append(color_map[_STAGE_POOL])
        else:
            node_colors.append(color_map[_STAGE_MOTIF])

    # ---- 构建 link ----
    sources = []
    targets = []
    values = []
    link_labels = []
    link_colors = []

    for r in edge_rows:
        src_idx = node_idx[r['source']]
        tgt_idx = node_idx[r['target']]
        sources.append(src_idx)
        targets.append(tgt_idx)
        values.append(r['flow_weight'])

        src_stage = r['source_stage']
        if r['target_stage'] == _STAGE_SUPPRESSED:
            lc = link_color_map[_STAGE_SUPPRESSED]
        else:
            lc = link_color_map.get(src_stage, 'rgba(127, 127, 127, 0.4)')
        link_colors.append(lc)

        motif_seq = r.get('leaf_seq', '')
        link_labels.append(
            f"{r['source']} → {r['target']}<br>"
            f"Displayed flow: {r['flow_weight']:.4f}<br>"
            f"IG sum: {r['ig_sum']:.4f}<br>"
            f"Raw attention sum: {r['attention_sum']:.4f}<br>"
            f"Local attention mass: {r['local_attention_sum']:.4f}<br>"
            f"Raw IG × attention: {r['ig_attention_sum']:.4f}<br>"
            f"Attention-kept flow: {r['attention_flow']:.4f}<br>"
            f"Miss flow: {r['miss_flow']:.4f}<br>"
            f"Windows: {r['raw_window_count']}<br>"
            f"Mean pred_prob: {r['mean_pred_prob']:.4f}<br>"
            f"Motif: {motif_seq}<br>"
            f"Range: {r['source_range']}"
        )

    if title is None:
        mod_name = edge_rows[0]['mod_name'] if edge_rows else ''
        title = f"IG×Attention Stage Sankey — {mod_name}"

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=node_pad,
            thickness=node_thickness,
            line=dict(color='black', width=0.5),
            label=node_order,
            color=node_colors,
            hovertemplate='%{label}<br>Total flow: %{value:.4f}<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate='%{label}<extra></extra>',
        ),
    )])

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=font_size),
        font=dict(size=font_size),
        width=width,
        height=height,
        annotations=[
            dict(
                text="Stage Sankey: the whole input sequence distributes raw IG "
                     "to M2D absolute-coordinate ranges; M2D raw IG then "
                     "splits into attention-kept and Miss flow.",
                xref="paper", yref="paper",
                x=0.5, y=-0.08, showarrow=False,
                font=dict(size=font_size, color="gray"),
            )
        ],
    )

    os.makedirs(os.path.dirname(output_html) or '.', exist_ok=True)
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"[Stage Sankey] wrote HTML: {output_html}")

    if output_png:
        try:
            fig.write_image(output_png, scale=2)
            print(f"[Stage Sankey] wrote PNG: {output_png}")
        except Exception as e:
            print(f"[Stage Sankey] PNG export failed (kaleido may not be installed): {e}")


# ---------------------------------------------------------------------------
# Motif/Tomtom 汇总解析：读取 STREME、模型 MEME 和 Tomtom，生成全局 Excel。
# Motif/Tomtom summary parsing: read STREME, model MEME, and Tomtom to build Excel.
# ---------------------------------------------------------------------------

def _consensus_from_pwm_rows(rows):
    """从 PWM 行生成 A/C/G/U consensus。
    Build an A/C/G/U consensus string from PWM rows.
    """
    letters = ['A', 'C', 'G', 'U']
    consensus = []
    for row in rows:
        if not row:
            continue
        consensus.append(letters[int(np.argmax(row))])
    return ''.join(consensus)


def parse_meme_like_motifs(path, source):
    """解析 MEME/STREME 文本中的 motif ID、consensus、宽度和 p-value。
    Parse motif ID, consensus, width, and p-value from MEME/STREME text files.
    """
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
                    'source': source,
                    'motif_id': motif_id,
                    'motif_name': motif_text,
                    'consensus': consensus,
                    'width': None,
                    'nsites': None,
                    'p_value': None,
                    '_rows': [],
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
    """读取 Tomtom TSV，跳过注释行。
    Read Tomtom TSV while skipping comment lines.
    """
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
    """为每个模型 motif 选择最佳 Tomtom p-value，用于 logo 标题。
    Select the best Tomtom p-value per model motif for logo titles.
    """
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
    """过滤没有 Tomtom p-value 的模型 motif，避免最终 logo 中出现 p=NA。
    Filter model motifs without Tomtom p-values so final logos never show p=NA.
    """
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
    """删除当前修饰的 motif logo，防止旧的 p=NA 图残留。
    Remove motif logo files for a modification to avoid stale p=NA figures.
    """
    for ext in ('png', 'pdf'):
        path = os.path.join(mod_dir, f"{mod_name}_motif_motif.{ext}")
        if os.path.exists(path):
            os.remove(path)


def build_mod_summary(mod_name, mod_dir):
    """构建单个修饰的 STREME、模型 motif 和 Tomtom 匹配汇总。
    Build STREME, model motif, and Tomtom match summaries for one modification.
    """
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
            'mod_name': mod_name,
            'matched': True,
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
                'mod_name': mod_name,
                'matched': False,
                'model_motif_id': motif['motif_id'],
                'model_consensus': motif.get('consensus', ''),
                'streme_motif_id': '',
                'streme_consensus': '',
                'streme_p_value': '',
                'tomtom_p_value': '',
                'tomtom_E_value': '',
                'tomtom_q_value': '',
                'overlap': '',
                'orientation': '',
                'optimal_offset': '',
            })

    for motif in streme_motifs:
        if motif['motif_id'] not in matched_streme_ids:
            match_rows.append({
                'mod_name': mod_name,
                'matched': False,
                'model_motif_id': '',
                'model_consensus': '',
                'streme_motif_id': motif['motif_id'],
                'streme_consensus': motif.get('consensus', ''),
                'streme_p_value': motif.get('p_value', ''),
                'tomtom_p_value': '',
                'tomtom_E_value': '',
                'tomtom_q_value': '',
                'overlap': '',
                'orientation': '',
                'optimal_offset': '',
            })

    streme_rows = [
        {
            'mod_name': mod_name,
            'streme_motif_id': m.get('motif_id', ''),
            'streme_consensus': m.get('consensus', ''),
            'streme_consensus_from_pwm': m.get('consensus_from_pwm', ''),
            'width': m.get('width', ''),
            'nsites': m.get('nsites', ''),
            'streme_p_value': m.get('p_value', ''),
        }
        for m in streme_motifs
    ]
    model_rows = [
        {
            'mod_name': mod_name,
            'model_motif_id': m.get('motif_id', ''),
            'model_consensus': m.get('consensus', ''),
            'width': m.get('width', ''),
            'nsites': m.get('nsites', ''),
        }
        for m in model_motifs
    ]
    return match_rows, streme_rows, model_rows, tomtom_rows


def write_tomtom_summary_excel(output_dir, match_rows, streme_rows, model_rows):
    """所有修饰跑完后写出 Tomtom/STREME/model motif 汇总 Excel。
    Write an all-modification Tomtom/STREME/model motif summary Excel file.
    """
    if not match_rows and not streme_rows and not model_rows:
        print("[Summary] no motif summary rows to write")
        return None
    import pandas as pd

    out_path = os.path.join(output_dir, 'tomtom_summary.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        pd.DataFrame(match_rows).to_excel(writer, sheet_name='tomtom_matches',
                                          index=False)
        pd.DataFrame(streme_rows).to_excel(writer, sheet_name='streme_motifs',
                                           index=False)
        pd.DataFrame(model_rows).to_excel(writer, sheet_name='model_motifs',
                                          index=False)
    print(f"[Summary] wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 主流程：加载模型和全量 human 数据，逐修饰提取 IG motif 并运行 STREME/Tomtom。
# Main pipeline: load model and full human data, then extract IG motifs per modification
# and run STREME/Tomtom.
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """执行完整 localized IG motif 分析流程（含 IG-only / IG×Attention Stage Sankey）。
    Run the full localized IG motif analysis pipeline (with IG-only / IG×Attention Stage Sankey).
    """
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
        output_root = os.path.join(PROJECT_ROOT, 'ipynb', 'motif_apply', 'checkpoints')
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

    # ---- Dataset via PlantDataset ----
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

    # ---- Use train_indices only for motif analysis ----
    if train_indices is not None:
        indices = train_indices
        print(f"[Data] using train indices only: {len(indices)} samples "
              f"(val={len(val_indices)} held out)")
    else:
        indices = list(range(len(dataset)))
        print(f"[Data] using all plant samples: {len(indices)}")

    target_mods = parse_mod_list(args.mods)
    if not target_mods:
        print("[Error] No valid mods. Exiting.")
        return
    print(f"[Mods] {[m for m, _ in target_mods]}")

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    run_cfg = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'project_root': PROJECT_ROOT,
        'seed': args.seed,
        'device': str(device),
        'data_usage': 'train_only',
        'dataset_name': 'plant',
        'init_checkpoint': getattr(args, 'init_checkpoint', _DEFAULT_INIT_CKPT),
        'trained_checkpoint': checkpoint_path,
        'training_mode': 'train_90_val_10_early_stop',
        'motif_event_split': 'train_only',
        'n_train': len(train_indices) if train_indices else len(indices),
        'n_val': len(val_indices) if val_indices else 0,
        'train_history_path': train_history_path,
        'data_dir': data_dir,
        'cache_dir': cache_dir,
        'n_used': len(indices),
        'negative_dir': args.negative_dir,
        'negative_mode': args.negative_mode,
        'negative_ratio': args.negative_ratio,
        'streme_minw': args.streme_minw,
        'streme_maxw': args.streme_maxw,
        'default_cluster_eps_by_mod': DEFAULT_CLUSTER_EPS_BY_MOD,
        'cluster_eps_override': args.cluster_eps,
        'cluster_min_samples': args.cluster_min_samples,
        'sequence_alphabet': 'RNA_ACGU_T_normalized_to_U',
        'ig_sankey_enabled': args.enable_ig_sankey,
        'sankey_mode': args.sankey_mode,
        'sankey_group_by': args.sankey_group_by,
        'sankey_top_n': args.sankey_top_n,
        'sankey_min_flow': args.sankey_min_flow,
        'sankey_cluster_method': 'dbscan',
        'sankey_cluster_eps': args.sankey_cluster_eps,
        'sankey_cluster_min_samples': args.sankey_cluster_min_samples,
        'sankey_width': args.sankey_width,
        'sankey_height': args.sankey_height,
        'sankey_font_size': args.sankey_font_size,
        'sankey_node_pad': args.sankey_node_pad,
        'sankey_node_thickness': args.sankey_node_thickness,
    }
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # ---- Load Sankey motif map if provided ----
    sankey_motif_map = None
    if args.sankey_motif_map:
        sankey_motif_map = load_sankey_motif_map(args.sankey_motif_map)
        print(f"[Sankey] loaded motif map from {args.sankey_motif_map} "
              f"({len(sankey_motif_map)} entries)")

    # ---- Collect events from dataset ----
    all_events = collect_events_from_dataset(dataset, indices, target_mods)
    print(f"[Events] total: {len(all_events)}")

    events_by_mod = defaultdict(list)
    for ev in all_events:
        events_by_mod[MOD_NAMES[ev[2]]].append(ev)

    # ---- Global summary buffers ----
    all_match_rows, all_streme_rows, all_model_rows = [], [], []
    # Sankey 全局汇总
    sankey_summary_rows = []

    # ---- Negative sampling setup ----
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
        sankey_cluster_eps = (args.sankey_cluster_eps
                              if args.sankey_cluster_eps is not None
                              else cluster_eps)
        sankey_cluster_min_samples = (args.sankey_cluster_min_samples
                                      if args.sankey_cluster_min_samples is not None
                                      else args.cluster_min_samples)
        print(f"[{mod_name}] cluster_eps={cluster_eps}, "
              f"cluster_min_samples={args.cluster_min_samples}")
        if args.enable_ig_sankey and args.sankey_group_by == 'motif_cluster':
            print(f"[{mod_name}] sankey motif_cluster=DBSCAN, "
                  f"eps={sankey_cluster_eps}, "
                  f"min_samples={sankey_cluster_min_samples}")

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
                inp = torch.FloatTensor(bytes_to_onehot(dataset.sequences[sidx]))  # (1001, 4)
                edge_index = data.edge_index                                      # (2, E) long tensor
                n_edges = edge_index.size(1)

                if ei == 0:
                    print(f"[{mod_name}] sample edge_index size: {n_edges} edges "
                          f"({'STRUCTURE' if n_edges > 2000 else 'SEQUENTIAL-ONLY'})")

                batch_vec = torch.zeros(inp.size(0), dtype=torch.long)
                seq_str = bytes_to_seqstr(dataset.sequences[sidx])

                score51, pprob, logit = compute_localized_ig(
                    model, inp, edge_index, batch_vec, cidx, spos,
                    ig_steps=args.ig_steps, device=device)

                # 提取 attention（ig_attention_stage 模式需要）
                # Extract attention (needed for ig_attention_stage mode)
                attn_51 = None
                if args.enable_ig_sankey and args.sankey_mode == 'ig_attention_stage':
                    attn_51 = extract_target_attention_51(
                        model, inp, edge_index, batch_vec, cidx, spos, device)
                local_attn_51 = None
                if attn_51 is not None:
                    attn_51 = np.asarray(attn_51, dtype=np.float64)
                    attn_51_sum = float(np.sum(attn_51))
                    if np.isfinite(attn_51_sum) and attn_51_sum > 0:
                        local_attn_51 = attn_51 / attn_51_sum
                    else:
                        local_attn_51 = np.ones_like(attn_51, dtype=np.float64) / len(attn_51)

                if args.pred_threshold > 0 and pprob < args.pred_threshold:
                    continue

                wins = extract_windows(score51, seq_str, spos,
                                       motif_w=args.motif_w, top_k=args.top_k,
                                       skip_n=args.skip_n_windows)
                for w in wins:
                    rec = dict(sample_idx=sidx, site_pos=spos, class_idx=cidx,
                               mod_name=mod_name, pred_prob=pprob, logit=logit, **w)

                    # 计算 attention 相关字段
                    # Compute attention-related fields
                    m51s = w['motif_start_51']
                    m51e = w['motif_end_51']
                    rec['window_ig_sum'] = float(np.sum(score51[m51s:m51e]))
                    if attn_51 is not None:
                        rec['window_attn_sum'] = float(np.sum(attn_51[m51s:m51e]))
                        rec['window_attn_mean'] = float(np.mean(attn_51[m51s:m51e]))
                        rec['window_ig_attn_score'] = float(
                            np.sum(score51[m51s:m51e] * attn_51[m51s:m51e]))
                        rec['window_local_attn_sum'] = float(
                            np.sum(local_attn_51[m51s:m51e]))
                        rec['window_local_attn_mean'] = float(
                            np.mean(local_attn_51[m51s:m51e]))
                        rec['window_attention_flow'] = float(
                            rec['window_ig_sum'] * rec['window_local_attn_sum'])
                        rec['window_miss_flow'] = float(
                            max(rec['window_ig_sum'] - rec['window_attention_flow'], 0.0))
                    else:
                        rec['window_attn_sum'] = 0.0
                        rec['window_attn_mean'] = 0.0
                        rec['window_ig_attn_score'] = rec['window_ig_sum']
                        rec['window_local_attn_sum'] = 1.0
                        rec['window_local_attn_mean'] = 1.0 / max(m51e - m51s, 1)
                        rec['window_attention_flow'] = rec['window_ig_sum']
                        rec['window_miss_flow'] = 0.0

                    records.append(rec)
                    fasta_seqs.append(w['seq_8nt'])
                    fasta_hdrs.append(f"{sidx}_{spos}_{w['rank']}")
            except Exception as e:
                errors.append(dict(sample_idx=sidx, site_pos=spos,
                                   class_idx=cidx, error=str(e)))
            if (ei + 1) % 100 == 0:
                print(f"  [{mod_name}] {ei+1}/{len(mod_events)} "
                      f"({time.time()-t0:.1f}s), wins={len(records)}")

        print(f"[{mod_name}] done: {len(records)} windows, "
              f"{len(errors)} errors, {time.time()-t0:.1f}s")

        write_windows_tsv(os.path.join(mod_dir, 'windows.tsv'), records)

        if errors:
            with open(os.path.join(mod_dir, 'errors.tsv'), 'w') as f:
                f.write("sample_idx\tsite_pos\tclass_idx\terror\n")
                for e in errors:
                    f.write(f"{e['sample_idx']}\t{e['site_pos']}\t"
                            f"{e['class_idx']}\t{e['error']}\n")

        # ===========================================================
        # Sankey 可视化（按模式分支）
        # Sankey visualization (branch by mode)
        # ===========================================================
        if args.enable_ig_sankey and records:
            sankey_group_by = args.sankey_group_by

            if args.sankey_mode == 'legacy_ig_only':
                # ---- 旧模式：IG-only Sankey ----
                print(f"\n[Sankey] building IG-only Sankey for {mod_name}...")

                edge_rows = build_ig_only_sankey_edges(
                    records, mod_name,
                    motif_map=sankey_motif_map,
                    group_by=sankey_group_by,
                    top_n=args.sankey_top_n,
                    min_flow=args.sankey_min_flow,
                    motif_cluster_eps=sankey_cluster_eps,
                    motif_cluster_min_samples=sankey_cluster_min_samples,
                    motif_cluster_random_state=args.seed,
                )

                if edge_rows:
                    sankey_tsv_path = os.path.join(mod_dir, f'{mod_name}_ig_only_sankey.tsv')
                    write_sankey_edges_tsv(sankey_tsv_path, edge_rows)
                    print(f"[Sankey] wrote {sankey_tsv_path}")

                    sankey_html_path = os.path.join(mod_dir, f'{mod_name}_ig_only_sankey.html')
                    sankey_png_path = None
                    if args.sankey_export_png:
                        sankey_png_path = os.path.join(mod_dir, f'{mod_name}_ig_only_sankey.png')

                    draw_ig_only_sankey(
                        edge_rows, sankey_html_path,
                        output_png=sankey_png_path,
                        title=f"IG-only Sankey — {mod_name} "
                              f"(flow = sum of IG window scores)",
                        width=args.sankey_width,
                        height=args.sankey_height,
                        font_size=args.sankey_font_size,
                        node_pad=args.sankey_node_pad,
                        node_thickness=args.sankey_node_thickness,
                    )

                    # 汇总信息
                    pipeline_flows = [r['flow_weight'] for r in edge_rows
                                      if r['leaf_group'] == '__pipeline__']
                    leaf_flows = [r['flow_weight'] for r in edge_rows
                                  if r['leaf_group'] != '__pipeline__']
                    total_pipeline = pipeline_flows[0] if pipeline_flows else 0.0
                    total_leaf = sum(leaf_flows)
                    n_leaves = len(leaf_flows)
                    n_windows = len(records)

                    sankey_summary_rows.append({
                        'mod_name': mod_name,
                        'sankey_mode': 'legacy_ig_only',
                        'total_flow': total_pipeline,
                        'total_ig_flow': total_pipeline,
                        'total_ig_attention_flow': 0.0,
                        'leaf_flow_sum': total_leaf,
                        'n_leaves': n_leaves,
                        'n_windows': n_windows,
                        'group_by': sankey_group_by,
                    })

                    # 验收检查 / Acceptance checks
                    assert len(set(round(f, 10) for f in pipeline_flows)) <= 1, \
                        f"[Sankey] pipeline edge flows not equal: {pipeline_flows}"
                    assert abs(total_leaf - total_pipeline) < 1e-6 * max(total_pipeline, 1e-10), \
                        f"[Sankey] leaf flow sum ({total_leaf}) != pipeline flow ({total_pipeline})"

                    print(f"[Sankey] {mod_name}: total_flow={total_pipeline:.4f}, "
                          f"leaves={n_leaves}, windows={n_windows}")
                else:
                    print(f"[Sankey] {mod_name}: no edges generated")

            else:
                # ---- 新模式：IG×Attention Stage Sankey ----
                print(f"\n[Stage Sankey] building IG×Attention stage Sankey for {mod_name}...")

                edge_rows = build_ig_attention_stage_sankey_edges(
                    records, mod_name,
                    motif_map=sankey_motif_map,
                    group_by=sankey_group_by,
                    top_n=args.sankey_top_n,
                    min_flow=args.sankey_min_flow,
                    motif_cluster_eps=sankey_cluster_eps,
                    motif_cluster_min_samples=sankey_cluster_min_samples,
                    motif_cluster_random_state=args.seed,
                )

                if edge_rows:
                    stage_tsv_path = os.path.join(
                        mod_dir, f'{mod_name}_ig_attention_stage_sankey.tsv')
                    write_stage_sankey_edges_tsv(stage_tsv_path, edge_rows)
                    print(f"[Stage Sankey] wrote {stage_tsv_path}")

                    stage_html_path = os.path.join(
                        mod_dir, f'{mod_name}_ig_attention_stage_sankey.html')
                    stage_png_path = None
                    if args.sankey_export_png:
                        stage_png_path = os.path.join(
                            mod_dir, f'{mod_name}_ig_attention_stage_sankey.png')

                    draw_ig_attention_stage_sankey(
                        edge_rows, stage_html_path,
                        output_png=stage_png_path,
                        title=f"IG×Attention Stage Sankey — {mod_name}",
                        width=args.sankey_width,
                        height=args.sankey_height,
                        font_size=args.sankey_font_size,
                        node_pad=args.sankey_node_pad,
                        node_thickness=args.sankey_node_thickness,
                    )

                    # 汇总 + 验收
                    input_flows = [r['flow_weight'] for r in edge_rows
                                   if r['source_stage'] == _STAGE_INPUT]
                    m2d_flows = [r['flow_weight'] for r in edge_rows
                                 if r['source_stage'] == _STAGE_M2D]
                    attn_flows = [r['flow_weight'] for r in edge_rows
                                  if r['source_stage'] == _STAGE_ATTN]
                    pool_flows = [r['flow_weight'] for r in edge_rows
                                  if r['source_stage'] == _STAGE_POOL]
                    miss_flows = [r['flow_weight'] for r in edge_rows
                                  if r['target_stage'] == _STAGE_SUPPRESSED]

                    total_input_flow = sum(input_flows)
                    total_ig_flow = sum(m2d_flows)
                    total_attn_pool_flow = sum(attn_flows)
                    total_pool_motif_flow = sum(pool_flows)
                    total_miss_flow = sum(miss_flows)

                    sankey_summary_rows.append({
                        'mod_name': mod_name,
                        'sankey_mode': 'ig_attention_stage',
                        'total_flow': total_ig_flow,
                        'total_ig_flow': total_ig_flow,
                        'total_ig_attention_flow': total_attn_pool_flow,
                        'leaf_flow_sum': total_pool_motif_flow,
                        'n_leaves': len(pool_flows),
                        'n_windows': len(records),
                        'group_by': sankey_group_by,
                    })

                    # 验收检查：
                    # 1) M2D->Attention flow >= 0
                    assert total_ig_flow >= -1e-10, \
                        f"[Stage Sankey] M2D->Attention flow negative: {total_ig_flow}"
                    # 2) Input->M2D raw flow should equal M2D kept+Miss flow.
                    assert abs(total_input_flow - total_ig_flow) < \
                        1e-6 * max(total_input_flow, 1e-10), \
                        f"[Stage Sankey] Input->M2D ({total_input_flow}) " \
                        f"!= M2D split flow ({total_ig_flow})"
                    # 3) Attention-kept flow should not exceed raw M2D IG flow.
                    assert total_attn_pool_flow <= total_ig_flow + 1e-8, \
                        f"[Stage Sankey] kept flow ({total_attn_pool_flow}) " \
                        f"> raw M2D flow ({total_ig_flow})"
                    # 4) Attention->Pooling == Pooling->Motif
                    assert abs(total_attn_pool_flow - total_pool_motif_flow) < \
                        1e-6 * max(total_attn_pool_flow, 1e-10), \
                        f"[Stage Sankey] Attention->Pooling ({total_attn_pool_flow}) " \
                        f"!= Pooling->Motif ({total_pool_motif_flow})"

                    print(f"[Stage Sankey] {mod_name}: "
                          f"IG_flow={total_ig_flow:.4f}, "
                          f"Attention_kept={total_attn_pool_flow:.4f}, "
                          f"Miss={total_miss_flow:.4f}, "
                          f"leaves={len(pool_flows)}, windows={len(records)}")
                else:
                    print(f"[Stage Sankey] {mod_name}: no edges generated")

        # ===========================================================
        # 原始流程继续：正负样本 FASTA、STREME、Tomtom 等
        # Original pipeline continues: positive/negative FASTA, STREME, Tomtom, etc.
        # ===========================================================

        # ---- Positives FASTA ----
        fasta_path = None
        if fasta_seqs:
            fasta_path = os.path.join(mod_dir, 'positives.fasta')
            write_fasta(fasta_path, fasta_seqs, fasta_hdrs)
            print(f"[{mod_name}] wrote {fasta_path} ({len(fasta_seqs)} seqs)")

        # ---- Negatives FASTA ----
        negative_fasta_path = None
        if has_negative_source and records:
            neg_seqs, neg_hdrs, neg_recs = generate_negatives_matched_offsets(
                records, neg_zero_seq_path,
                motif_w=args.motif_w,
                negative_ratio=args.negative_ratio,
                seed=args.negative_seed,
                allow_no_negative=args.allow_no_negative,
            )
            if neg_seqs:
                negative_fasta_path = os.path.join(mod_dir, 'negatives.fasta')
                write_fasta(negative_fasta_path, neg_seqs, neg_hdrs)
                print(f"[{mod_name}] wrote {negative_fasta_path} ({len(neg_seqs)} neg seqs)")
                neg_tsv_path = os.path.join(mod_dir, 'negative_windows.tsv')
                write_negative_windows_tsv(neg_tsv_path, neg_recs)
                print(f"[{mod_name}] wrote {neg_tsv_path}")
                if len(neg_seqs) != len(fasta_seqs):
                    print(f"[{mod_name}] WARNING: positives={len(fasta_seqs)} "
                          f"vs negatives={len(neg_seqs)}")
        elif not has_negative_source and not args.allow_no_negative:
            print(f"[{mod_name}] no negative source available")

        if args.skip_external_tools:
            print(f"[{mod_name}] skipping external tools")
            continue
        if len(fasta_seqs) < args.min_events_per_mod:
            print(f"[{mod_name}] too few for STREME, skipping")
            continue

        # ---- STREME ----
        try:
            streme_out = os.path.join(mod_dir, 'streme_out')
            run_streme(
                streme_out,
                fasta_path,
                negative_fasta=negative_fasta_path,
                minw=args.streme_minw,
                maxw=args.streme_maxw,
                rna=True,
            )
        except Exception as e:
            print(f"[{mod_name}] STREME failed: {e}")

        # ---- Consensus motif ----
        try:
            if len(fasta_seqs) >= 4:
                cm, ig_sc = cal_consensus_motif_2(
                    fasta_seqs, [r['window_score'] for r in records],
                    eps=cluster_eps,
                    min_samples=args.cluster_min_samples,
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

        # ---- Tomtom ----
        streme_txt = os.path.join(streme_out, 'streme.txt')
        meme_p = os.path.join(mod_dir, 'model_motifs.meme')
        if os.path.isfile(streme_txt) and os.path.isfile(meme_p):
            try:
                tomtom_out = os.path.join(mod_dir, 'tomtom_out')
                tsv = run_tomtom_validation(meme_p, streme_txt, tomtom_out)
                print(f"[{mod_name}] Tomtom done: {tsv}")

                tomtom_rows_for_logo = parse_tomtom_tsv(tsv)
                if cm is not None and ig_sc is not None:
                    p_values = best_tomtom_pvalues_by_query(
                        tomtom_rows_for_logo, cm.shape[0])
                    cm_plot, ig_plot, p_plot, keep_idx = filter_motifs_with_pvalues(
                        cm, ig_sc, p_values)
                    try:
                        if cm_plot is None:
                            remove_logo_files(mod_dir, mod_name)
                            print(f"[{mod_name}] no Tomtom p-values for logos; "
                                  "removed motif logo files")
                        else:
                            motif_labels = [f"Model_Motif_{i + 1}" for i in keep_idx]
                            print(f"[{mod_name}] logo motifs kept after p-value filter: "
                                  f"{motif_labels}")
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif",
                                             file_format='png', p_values=p_plot,
                                             motif_labels=motif_labels)
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif",
                                             file_format='pdf', p_values=p_plot,
                                             motif_labels=motif_labels)
                    except Exception as e_logo:
                        print(f"[{mod_name}] p-value motif logo failed: {e_logo}")

                rp = os.path.join(mod_dir, 'results.txt')
                with open(rp, 'w') as f:
                    f.write(f"Mod: {mod_name}\nWindows: {len(records)}\n\n")
                    if os.path.isfile(tsv):
                        with open(tsv, 'r') as tt:
                            for row in csv.DictReader(tt, delimiter='\t'):
                                f.write(f"  {row.get('Query_ID','')} vs "
                                        f"{row.get('Target_ID','')}: "
                                        f"p={row.get('p-value','')} "
                                        f"E={row.get('E-value','')} "
                                        f"q={row.get('q-value','')}\n")
            except Exception as e:
                print(f"[{mod_name}] Tomtom failed: {e}")

        # ---- Per-mod summary collection ----
        match_rows, streme_rows, model_rows, _ = build_mod_summary(mod_name, mod_dir)
        all_match_rows.extend(match_rows)
        all_streme_rows.extend(streme_rows)
        all_model_rows.extend(model_rows)

    # ---- 全局汇总输出 ----
    write_tomtom_summary_excel(args.output_dir, all_match_rows,
                               all_streme_rows, all_model_rows)

    # ---- Sankey 全局汇总 TSV ----
    if args.enable_ig_sankey and sankey_summary_rows:
        summary_path = os.path.join(args.output_dir, 'ig_sankey_summary.tsv')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("mod_name\tsankey_mode\ttotal_flow\ttotal_ig_flow\t"
                    "total_ig_attention_flow\tleaf_flow_sum\tn_leaves\tn_windows\tgroup_by\n")
            for r in sankey_summary_rows:
                f.write(f"{r['mod_name']}\t{r['sankey_mode']}\t"
                        f"{r['total_flow']:.6f}\t{r['total_ig_flow']:.6f}\t"
                        f"{r['total_ig_attention_flow']:.6f}\t"
                        f"{r['leaf_flow_sum']:.6f}\t{r['n_leaves']}\t"
                        f"{r['n_windows']}\t{r['group_by']}\n")
        print(f"[Sankey] wrote summary: {summary_path}")

    print(f"\n[Done] Outputs in {args.output_dir}")


def main():
    # 中文：命令行参数定义，覆盖输入模型、输出目录、采样规模和外部工具选项。
    # English: CLI definition for model inputs, output paths, sampling scale, and external tools.
    p = argparse.ArgumentParser(description='Plant localized IG motif analysis with Sankey visualization (v2)')
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
    p.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'ipynb', 'motif_apply', 'output', 'plant', 'ig_attn_stage_sankey'))
    p.add_argument('--data_dir', default=None)
    p.add_argument('--mods', default=PLANT_DEFAULT_MODS)
    p.add_argument('--sample_size', type=int, default=5000)
    p.add_argument('--window_51', type=int, default=51)
    p.add_argument('--motif_w', type=int, default=8)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--ig_steps', type=int, default=32)
    p.add_argument('--baseline', default='zero')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--pred_threshold', type=float, default=0.0)
    p.add_argument('--positive_only', action='store_true')
    p.add_argument('--skip_n_windows', action='store_true', default=True)
    p.add_argument('--min_events_per_mod', type=int, default=10)
    p.add_argument('--skip_external_tools', action='store_true')
    p.add_argument('--alignment_r_script', default=None)
    p.add_argument('--conda_env', default='meme_env')

    # 原始负样本和聚类参数
    p.add_argument('--negative_dir', default='npy/zero',
                   help='Directory containing zero_seq.npy for negative sampling')
    p.add_argument('--negative_mode', default='matched_offsets',
                   choices=['matched_offsets'],
                   help='Negative sampling strategy')
    p.add_argument('--negative_ratio', type=float, default=1.0,
                   help='Ratio of negatives to positives (default 1.0)')
    p.add_argument('--negative_seed', type=int, default=666,
                   help='Random seed for negative sampling')
    p.add_argument('--allow_no_negative', action='store_true',
                   help='Continue even if zero_seq.npy is missing')
    p.add_argument('--streme_minw', type=int, default=5,
                   help='STREME minimum motif width (default 5)')
    p.add_argument('--streme_maxw', type=int, default=15,
                   help='STREME maximum motif width (default 15)')
    p.add_argument('--cluster_eps', type=float, default=None,
                   help='Override per-modification DBSCAN eps for consensus motifs')
    p.add_argument('--cluster_min_samples', type=int, default=10,
                   help='DBSCAN min_samples for consensus motifs (default 10)')

    # ---- Sankey 参数 ----
    p.add_argument('--enable_ig_sankey', action='store_true', default=False,
                   help='Enable Sankey visualization per modification')
    p.add_argument('--sankey_mode', default='ig_attention_stage',
                   choices=['legacy_ig_only', 'ig_attention_stage'],
                   help='Sankey mode: legacy_ig_only (IG-only) or ig_attention_stage (IG×Attention stage)')
    p.add_argument('--sankey_group_by', default='motif_cluster',
                   choices=['motif_map', 'motif_cluster', 'seq_8nt'],
                   help='Leaf grouping mode for Sankey (default: motif_cluster)')
    p.add_argument('--sankey_motif_map', default=None,
                   help='Path to user motif mapping TSV for Sankey leaf labels')
    p.add_argument('--sankey_cluster_eps', type=float, default=None,
                   help='DBSCAN eps for Sankey motif_cluster; default uses per-mod cluster_eps')
    p.add_argument('--sankey_cluster_min_samples', type=int, default=None,
                   help='DBSCAN min_samples for Sankey motif_cluster; default uses --cluster_min_samples')
    p.add_argument('--sankey_top_n', type=int, default=20,
                   help='Keep top-N leaves in Sankey; rest merged to Other (default: 20)')
    p.add_argument('--sankey_min_flow', type=float, default=0.0,
                   help='Minimum flow threshold for Sankey leaves (default: 0.0)')
    p.add_argument('--sankey_export_png', action='store_true', default=False,
                   help='Export Sankey as PNG (requires kaleido)')
    p.add_argument('--sankey_width', type=int, default=1920,
                   help='Sankey figure width in pixels (default: 1200)')
    p.add_argument('--sankey_height', type=int, default=1200,
                   help='Sankey figure height in pixels (default: 700)')
    p.add_argument('--sankey_font_size', type=int, default=18,
                   help='Unified Sankey font size in pt (default: 18)')
    p.add_argument('--sankey_node_pad', type=int, default=20,
                   help='Sankey node padding in pixels (default: 20)')
    p.add_argument('--sankey_node_thickness', type=int, default=25,
                   help='Sankey node thickness in pixels (default: 25)')

    args = p.parse_args()

    # 如果提供了 motif_map 但用户没显式改 group_by，自动切换
    # Auto-switch group_by to motif_map if motif_map file is provided
    if args.sankey_motif_map and args.sankey_group_by != 'motif_map':
        print(f"[Sankey] motif_map provided, but group_by={args.sankey_group_by}. "
              f"Use --sankey_group_by motif_map to activate motif map grouping.")

    if args.checkpoint and not args.trained_checkpoint:
        args.trained_checkpoint = args.checkpoint
    run_pipeline(args)


if __name__ == '__main__':
    main()
