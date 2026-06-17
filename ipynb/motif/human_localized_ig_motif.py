#!/usr/bin/env python3
"""
human_localized_ig_motif.py
Localized Integrated Gradients motif analysis for human 12-class RNA modification model.

v2 — Unified full-data loading via Mer100Dataset,
     real structure edge_index from LinearFold cache, negative sampling from npy/zero,
     and per-modification DBSCAN eps for consensus motifs.

多标签处理说明：
1. IG 侧：human 数据是 12 类修饰多标签任务，同一条 1001nt 序列可以有多个修饰位点、
   同一个样本也可以同时包含多个修饰类别。本脚本不会把样本强行改成单标签；它直接读取
   Mer100Dataset.full_labels 中的位点级标签，对用户指定的每一种修饰分别收集
   (sample_idx, site_pos, class_idx) 事件。计算 localized IG 时，每次只固定一个事件的
   class_idx 作为目标 logit，并只遮蔽该位点中心 51nt 作为 baseline，因此一个多标签样本
   会按修饰类别和位点被拆成多个独立解释事件。
2. STREME 侧：STREME 本身只接收 positive/negative 序列集合，不理解模型的 12 类多标签。
   因此脚本按修饰类别分别运行 STREME：某个修饰的 positive 是该修饰事件中 IG top-k 选出的
   短窗口，negative 来自 npy/zero/zero_seq.npy 中按相对位置匹配抽取的背景窗口。这样得到的
   de novo motif 表示“该修饰类别的 IG 高贡献窗口相对于 zero 背景富集了什么序列模式”。
3. 字母表：所有输出序列统一使用 RNA 字母 A/C/G/U；输入中若出现 T，会在字符串输出层转成 U。
   one-hot 编码层仍保持 4 通道 A/C/G/U，其中 T 和 U 都映射到第 4 个通道，确保历史 T 数据
   与 RNA U 数据等价。

Multi-label handling:
1. IG side: the human dataset is a 12-class multi-label modification task. One 1001nt
   sequence may contain multiple modification sites and multiple modification classes.
   This script does not collapse samples into single-label examples. It reads
   Mer100Dataset.full_labels directly, collects one (sample_idx, site_pos, class_idx)
   event per requested modification site, and computes localized IG for exactly one
   target class logit at a time while masking only the 51nt window around that site.
2. STREME side: STREME receives only positive/negative sequence sets and has no notion
   of the model's 12 labels. The script therefore runs STREME separately for each
   modification. Positives are IG top-k short windows from that modification's events;
   negatives are position-matched windows sampled from npy/zero/zero_seq.npy. The
   resulting de novo motifs describe what sequence patterns are enriched in high-IG
   windows for that modification against the zero background.
3. Alphabet: all written sequences use RNA A/C/G/U. Any T in input strings is normalized
   to U for output, while the one-hot encoder maps both T and U to the same fourth
   channel so legacy T-encoded data and RNA U-encoded data remain equivalent.

Smoke test:
    python ipynb/motif/human_localized_ig_motif.py \
      --config json/human.json \
      --checkpoint logs/old/rna_classification_20260129_195404/checkpoints/epoch_090.pt \
      --output_dir ipynb/motif/output/debug_localized_ig \
      --mods m6A --sample_size 20 --top_k 2 --ig_steps 8 \
      --skip_external_tools --batch_size 1 --device cuda
"""

import argparse, json, os, sys, time, warnings, csv, re
from collections import defaultdict
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
from dataset.human import Mer100Dataset, MOD_NAMES, LABEL_MAPPING
from motif_utils import (
    highest_x, cal_consensus_motif_2, export_to_meme,
    draw_motif_logos,
    run_streme, run_tomtom_validation, run_alignment_r,
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
        byte_array = raw.view(np.uint8).reshape(-1)
    else:
        byte_array = raw.astype(np.uint8, copy=False).reshape(-1)
    return _BYTE_TO_ONEHOT_MAPPING[byte_array].copy()

# ---------------------------------------------------------------------------
# 修饰名称映射：把用户输入的修饰名映射到模型的 0-11 类别索引。
# Modification mapping: map user-facing names to model class indices 0-11.
# ---------------------------------------------------------------------------
MOD_NAME_TO_CLASS = {v: k for k, v in MOD_NAMES.items()}  # mod_name -> class_idx (0-11)

# ---------------------------------------------------------------------------
# 每类修饰的 DBSCAN eps 默认值：用于 IG 片段聚类生成模型侧共识 motif。
# Default DBSCAN eps per modification: used to cluster IG windows into model motifs.
#
# 说明：8nt one-hot 展平后，1 个碱基错配的欧氏距离约为 sqrt(2)=1.414，
#      2 个错配约为 2.0。原来的 eps=0.3 几乎只允许完全相同序列成团，
#      对 Tm/Y 这类较分散、U-rich 的窗口会过严，容易全部变成 noise 后退化成 1 个 cluster。
# Note: after flattening 8nt one-hot encodings, one mismatch has Euclidean distance
#       about sqrt(2)=1.414, and two mismatches about 2.0. The old eps=0.3 only
#       groups nearly identical sequences, which is too strict for diffuse U-rich
#       classes such as Tm/Y and can collapse all noise into one fallback cluster.
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
# 配置加载：读取 json/human.json 里的模型、数据和训练配置。
# Config loading: read model, data, and training settings from json/human.json.
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
# 序列辅助函数：从 Mer100Dataset 的字节数组恢复原始 RNA 序列字符串。
# Sequence helpers: recover raw RNA strings from Mer100Dataset byte arrays.
# ---------------------------------------------------------------------------

def bytes_to_seqstr(row):
    """将 (1001,) |S1/byte 数组转换为 RNA 序列字符串，T 统一写成 U。
    Convert a (1001,) |S1/byte array to an RNA sequence string, normalizing T to U.
    """
    raw = np.asarray(row)
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
    # 中文：模型类别索引 0-11 需要映射回 1001loc.npy 里存储的修饰 ID 1-12。
    # English: map model class index 0-11 back to modification ID 1-12 in 1001loc.npy.
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

    中文：对每个 positive window，读取它在中心 51nt 内的 ``motif_start_51``，
    然后从随机 zero 样本的中心 51nt 区域同一相对位置抽取等长片段。

    English: for each positive window, use its ``motif_start_51`` and extract
    an equal-length subsequence at the same relative offset from the center
    51nt region of a randomly selected zero sample.

    Returns
    -------
    neg_seqs : list[str]
        Negative sequences (RNA alphabet, U not T).
    neg_hdrs : list[str]
        FASTA headers for tracking.
    neg_records : list[dict]
        Metadata for ``negative_windows.tsv``.
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
        # 中文：当 negative_ratio > 1 时循环复用 positive 的 offset 模板。
        # English: cycle through positive offset templates when negative_ratio > 1.
        rec = records[i % n_pos]
        motif_start_51 = rec['motif_start_51']
        m51e = motif_start_51 + motif_w
        if m51e > 51:
            m51e = 51
        # 中文：1001nt 序列的中心 51nt 对应绝对位置 475..525。
        # English: the center 51nt of a 1001nt sequence spans absolute positions 475..525.
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
        # 中文：Captum 可能传入单样本或 batch，这里统一成 (N, 1001, 4)。
        # English: Captum may pass one sample or a batch; normalize to (N, 1001, 4).
        if x.dim() == 2:
            x = x.unsqueeze(0)
        bs = x.size(0)
        # 中文：为 batch 中每个图复制 edge_index，并按节点偏移修正索引。
        # English: replicate edge_index for each graph and offset node indices per batch item.
        offsets = torch.arange(bs, device=x.device).view(1, -1, 1) * self.seq_len
        ei = self.single_edge_index.unsqueeze(1)  # (2, 1, E)
        ei = ei + offsets  # (2, N, E)
        ei = ei.permute(1, 0, 2).reshape(2, -1)  # (2, N*E)
        # 中文：构造 PyG batch 向量，标记每个节点属于哪个样本图。
        # English: build the PyG batch vector to mark graph membership for each node.
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

    # 中文：先做一次前向推理，记录目标修饰类别的 logit 和 sigmoid 概率。
    # English: run a forward pass to record the target class logit and sigmoid probability.
    with torch.no_grad():
        out = model(input_full.unsqueeze(0), edge_index_dev, batch_vec_dev)
        logits_12 = out[0] if isinstance(out, tuple) else out
        logit_val = (logits_12[0, class_idx] if logits_12.dim() == 2
                     else logits_12[class_idx]).item()
    pred_prob = torch.sigmoid(torch.tensor(logit_val)).item()

    # 中文：baseline 只清零目标位点周围 51nt，以得到局部窗口贡献。
    # English: baseline zeros out only the 51nt site-centered window for localized attribution.
    s0, s1 = site_pos - 25, site_pos + 26
    baseline = input_full.clone()
    baseline[s0:s1, :] = 0.0

    wrapper = LocalizedIGWrapper(model, edge_index_dev, 1001, class_idx).to(device)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(input_full.unsqueeze(0), baselines=baseline.unsqueeze(0),
                        n_steps=ig_steps, target=0)
    attr_np = attr.squeeze(0).detach().cpu().numpy()  # (1001, 4)
    # 中文：通道维取绝对值求和，得到每个位置的贡献强度。
    # English: sum absolute attributions across nucleotide channels to get per-position strength.
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
    "rank\tpred_prob\tlogit\n"
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
                    f"{r['rank']}\t{r['pred_prob']:.6f}\t{r['logit']:.6f}\n")


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
    """执行完整 localized IG motif 分析流程。
    Run the full localized IG motif analysis pipeline.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Init] device={device}")

    cfg = load_config(args.config)
    print(f"[Config] loaded from {args.config}")

    model = load_model(cfg, args.checkpoint, device)
    print(f"[Model] loaded from {args.checkpoint}")

    # ---- Dataset via Mer100Dataset ----
    # 中文：复用 dataset/human.py 的 Mer100Dataset，确保 one-hot、LinearFold 结构边、
    #      y_site 标签和训练/推理流程保持一致。
    # English: reuse Mer100Dataset from dataset/human.py so one-hot encoding,
    #          LinearFold structure edges, y_site labels, and model inputs match training.
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, cfg['data']['human_data_dir'])
    cache_dir = os.path.join(PROJECT_ROOT, cfg['data'].get('cache_dir', 'cache'))
    print(f"[Data] loading Mer100Dataset from {data_dir}")
    dataset = Mer100Dataset(
        # 中文：use_human3=True 时 mode 不切分样本，只影响结构缓存名；保留 train 以复用已有缓存。
        # English: with use_human3=True, mode does not subset samples; it only names the cache.
        mode='train',
        data_dir=data_dir,
        cache_dir=cache_dir,
        use_human3=True,
        use_cache=True,
        preload_cache=True,
    )
    print(f"[Data] dataset size: {len(dataset)}")

    # ---- Full-data analysis ----
    # 中文：motif 分析使用全量 human 数据，不再先构造 train/test split。
    # English: motif analysis uses the full human dataset; no train/test split is created here.
    indices = list(range(len(dataset)))
    print(f"[Data] using all human samples: {len(indices)}")

    target_mods = parse_mod_list(args.mods)
    if not target_mods:
        print("[Error] No valid mods. Exiting.")
        return
    print(f"[Mods] {[m for m, _ in target_mods]}")

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    # 中文：保存运行配置，便于之后复现实验和追踪输出来源。
    # English: save the run configuration to make outputs reproducible and traceable.
    run_cfg = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'project_root': PROJECT_ROOT,
        'seed': args.seed,
        'device': str(device),
        'data_usage': 'all_human',
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
    }
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # ---- Collect events from dataset ----
    # 中文：按修饰类别从全量 y_site 标签中收集所有可解释的位点事件。
    # English: collect all interpretable site events for the requested modifications from y_site labels.
    all_events = collect_events_from_dataset(dataset, indices, target_mods)
    print(f"[Events] total: {len(all_events)}")

    events_by_mod = defaultdict(list)
    for ev in all_events:
        events_by_mod[MOD_NAMES[ev[2]]].append(ev)

    # ---- Global summary buffers ----
    # 中文：跨修饰累计 STREME、模型 motif 和 Tomtom 匹配结果，最后写入 Excel。
    # English: accumulate STREME, model motif, and Tomtom match rows across mods for Excel.
    all_match_rows, all_streme_rows, all_model_rows = [], [], []

    # ---- Negative sampling setup ----
    # 中文：负样本默认来自 npy/zero/zero_seq.npy，用于 STREME 的 --n 背景文件。
    # English: negatives default to npy/zero/zero_seq.npy and are passed to STREME via --n.
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

        # 中文：每个修饰使用独立 eps；命令行 --cluster_eps 可临时覆盖所有修饰。
        # English: each modification uses its own eps; --cluster_eps can override all mods.
        cluster_eps = (args.cluster_eps if args.cluster_eps is not None
                       else DEFAULT_CLUSTER_EPS_BY_MOD.get(mod_name, 2.0))
        print(f"[{mod_name}] cluster_eps={cluster_eps}, "
              f"cluster_min_samples={args.cluster_min_samples}")

        # 中文：为控制运行时间，可按事件数随机下采样；0 或负数表示不下采样。
        # English: optionally downsample events for runtime control; <=0 disables sampling.
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
                # 中文：从 Mer100Dataset 取样本和真实 LinearFold 结构边。
                # English: fetch the sample and real LinearFold structure edges from Mer100Dataset.
                data = dataset[sidx]
                # 中文：edge_index/标签仍复用 Mer100Dataset；x 用本脚本的 T/U 等价映射显式重建。
                # English: edge_index/labels still come from Mer100Dataset; x is rebuilt with this script's T/U-equivalent mapping.
                inp = torch.FloatTensor(bytes_to_onehot(dataset.sequences[sidx]))  # (1001, 4)
                edge_index = data.edge_index                                      # (2, E) long tensor
                n_edges = edge_index.size(1)

                if ei == 0:
                    print(f"[{mod_name}] sample edge_index size: {n_edges} edges "
                          f"({'STRUCTURE' if n_edges > 2000 else 'SEQUENTIAL-ONLY'})")

                batch_vec = torch.zeros(inp.size(0), dtype=torch.long)
                seq_str = bytes_to_seqstr(dataset.sequences[sidx])

                # 中文：计算目标修饰类别在该位点中心 51nt 内的 localized IG 分数。
                # English: compute localized IG scores for the target class in the 51nt site window.
                score51, pprob, logit = compute_localized_ig(
                    model, inp, edge_index, batch_vec, cidx, spos,
                    ig_steps=args.ig_steps, device=device)

                if args.pred_threshold > 0 and pprob < args.pred_threshold:
                    continue

                # 中文：从 51nt 分数里选 top-k 个短 motif 窗口，作为 STREME positive。
                # English: select top-k short motif windows from the 51nt scores as STREME positives.
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

        print(f"[{mod_name}] done: {len(records)} windows, "
              f"{len(errors)} errors, {time.time()-t0:.1f}s")

        write_windows_tsv(os.path.join(mod_dir, 'windows.tsv'), records)

        if errors:
            with open(os.path.join(mod_dir, 'errors.tsv'), 'w') as f:
                f.write("sample_idx\tsite_pos\tclass_idx\terror\n")
                for e in errors:
                    f.write(f"{e['sample_idx']}\t{e['site_pos']}\t"
                            f"{e['class_idx']}\t{e['error']}\n")

        # ---- Positives FASTA ----
        # 中文：写出 IG 选中的正样本短片段，供 STREME --p 使用。
        # English: write IG-selected positive short windows for STREME --p.
        fasta_path = None
        if fasta_seqs:
            fasta_path = os.path.join(mod_dir, 'positives.fasta')
            write_fasta(fasta_path, fasta_seqs, fasta_hdrs)
            print(f"[{mod_name}] wrote {fasta_path} ({len(fasta_seqs)} seqs)")

        # ---- Negatives FASTA ----
        # 中文：按 positive 窗口的相对位置抽取 zero 负样本，供 STREME --n 使用。
        # English: sample zero negatives at positive-matched offsets for STREME --n.
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
                # Verify counts match
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
        # 中文：用显式负样本运行 STREME，避免只用 positive shuffle 作为背景。
        # English: run STREME with explicit negatives instead of relying only on shuffled positives.
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
        # 中文：对模型 IG 短窗口聚类，生成模型侧共识 PWM，再导出为 MEME 给 Tomtom。
        # English: cluster model IG short windows into consensus PWMs and export MEME for Tomtom.
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

                # 中文：不在 Tomtom 之前绘制 logo，避免生成 p=NA 的 motif 图。
                # English: do not draw logos before Tomtom, avoiding p=NA motif figures.
                remove_logo_files(mod_dir, mod_name)
        except Exception as e:
            print(f"[{mod_name}] consensus motif failed: {e}")

        # ---- Tomtom ----
        # 中文：比较模型侧 motif 与 STREME de novo motif 的相似性。
        # English: compare model-side motifs against STREME de novo motifs.
        streme_txt = os.path.join(streme_out, 'streme.txt')
        meme_p = os.path.join(mod_dir, 'model_motifs.meme')
        if os.path.isfile(streme_txt) and os.path.isfile(meme_p):
            try:
                tomtom_out = os.path.join(mod_dir, 'tomtom_out')
                tsv = run_tomtom_validation(meme_p, streme_txt, tomtom_out)
                print(f"[{mod_name}] Tomtom done: {tsv}")

                # 中文：仅保留有 Tomtom p-value 的模型 motif 绘图，过滤 p=NA motif。
                # English: draw only model motifs with Tomtom p-values, filtering p=NA motifs.
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
                            print(f"[{mod_name}] logo motifs kept after p-value filter: "
                                  f"{[i + 1 for i in keep_idx]}")
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif",
                                             file_format='png', p_values=p_plot)
                            draw_motif_logos(cm_plot, ig_plot, mod_name, res_dir=mod_dir,
                                             filename=f"{mod_name}_motif",
                                             file_format='pdf', p_values=p_plot)
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
        # 中文：无论 Tomtom 是否有显著匹配，都汇总 STREME 捕获序列、模型捕获序列和匹配情况。
        # English: summarize STREME-captured sequences, model-captured sequences, and matches regardless of significance.
        match_rows, streme_rows, model_rows, _ = build_mod_summary(mod_name, mod_dir)
        all_match_rows.extend(match_rows)
        all_streme_rows.extend(streme_rows)
        all_model_rows.extend(model_rows)

    write_tomtom_summary_excel(args.output_dir, all_match_rows,
                               all_streme_rows, all_model_rows)

    print(f"\n[Done] Outputs in {args.output_dir}")


def main():
    # 中文：命令行参数定义，覆盖输入模型、输出目录、采样规模和外部工具选项。
    # English: CLI definition for model inputs, output paths, sampling scale, and external tools.
    p = argparse.ArgumentParser(description='Localized IG motif analysis (v2)')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'ipynb', 'motif', 'output'))
    p.add_argument('--data_dir', default=None)
    p.add_argument('--mods', default=','.join(MOD_NAMES.values()))
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

    # 中文：以下参数控制 zero 负样本、STREME 宽度和模型侧 motif 聚类。
    # English: these arguments control zero negatives, STREME widths, and model-side clustering.
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

    args = p.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
